"""Permanent AS2 event-budget measurement; never imports the retired Go2 driver."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from . import state as st
from .task import configure_reward_budget, reward_budget


def build_budget(episodes, source):
    """Use measured mature AS2 motion return without shrinking its 15 coefficients."""
    complete = [row for row in episodes if row['qualified_complete']]
    if not complete:
        raise RuntimeError('No complete tracking-qualified AS2 episode; no reward budget can be invented')
    ref = max(row['net_return'] for row in complete)
    if ref <= 0:
        raise RuntimeError('Measured AS2 reference return must be positive')
    cap = .9*ref
    event_seconds, positive_rate = 8., 3.
    bonus = cap-positive_rate*event_seconds
    if bonus <= 0:
        raise RuntimeError('Measured AS2 return cannot support an 8 s task and positive success bonus')
    tasks = []
    for task_id in range(4):
        special = task_id == 3
        length = event_seconds if special else 20.
        tasks.append({'task_id':task_id, 'episode_seconds':length,
            'base_positive_rate_bound':positive_rate,
            'base_positive_return_bound':positive_rate*length,
            'success_bonus':bonus if special else 0.,
            'unfinished_rate':.002*ref if special else 0.})
    values = torch.tensor([row['net_return'] for row in complete], dtype=torch.float64)
    result = {'schema':'as2_depth_crossing_budget_v2', 'robot':'as2',
        'reference_net_max':ref, 'positive_cap':cap, 'failure_penalty':.1*ref,
        'source':source, 'qualified_episodes':len(complete), 'total_episodes':len(episodes),
        'reference_statistics':{'mean':float(values.mean()), 'median':float(values.median()),
            'min':float(values.min()), 'max':float(values.max())},
        'tasks':tasks, 'episodes':episodes,
        'accounting':('AS2 continuous coefficients unchanged; rate rewards multiply dt=.02. '
            'Hard success and failure bonuses occur once. Positive cap applies to crossing task 3; '
            'ordinary AS2 tasks keep their original 20 s horizon. Tracking <=3/s before and after completion.'),
        'design_status':'Measured scale plus explicitly chosen event coefficients; not AS2 upstream reward facts'}
    configure_reward_budget(result)
    return result


@torch.no_grad()
def calibrate_as2_reward(env, action_fn, output, source_policy, *, branch='student',
                         reference_seconds=20., reset_fn=None):
    """Measure the first complete flat episode of every dedicated calibration env.

    action_fn(snapshot) returns raw AS2 actions in canonical FL/FR/RL/RR order.
    The caller loads the mature *blind AS2* checkpoint with its own source model;
    this function does not reinterpret checkpoint keys or robot parameters.
    Use a dedicated environment: this measurement deliberately resets physics.
    Optional reset_fn(done) resets a stateful inference wrapper.
    """
    if action_fn is None or branch not in ('teacher', 'student'):
        raise ValueError('A mature AS2 action function and explicit branch are required')
    if (abs(reference_seconds-20.) > 1e-8 or abs(env.cfg.dt-.02) > 1e-8
            or abs(env.cfg.episode_seconds-20.) > 1e-8):
        raise ValueError('Calibration must use the AS2 20 s / 50 Hz reference')
    policy_path = Path(source_policy)
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    n, device = env.num_envs, env.device
    original = {'iteration':env.iteration, 'variant':env.cfg.variant,
        'evaluation_mode':env.evaluation_mode, 'calibration_flat':env.calibration_flat,
        'command_override':env.command_override, 'budget':reward_budget(),
        'curriculum':copy.deepcopy(env.curriculum_state_dict())}
    result = None
    try:
        configure_reward_budget(None)
        env.cfg.variant = 'simple'
        env.evaluation_mode = True
        env.iteration = 0
        env.command_override = None
        env.place_calibration_flat()
        env.iteration = 3000
        commands = torch.tensor([[0,0,0],[.5,0,0],[.5,.2,.4],[1,0,0],[-.5,0,0]],
                                device=device, dtype=torch.float32)
        rows = torch.arange(n, device=device)
        command = commands[rows%len(commands)]
        payload = (rows//len(commands)%3).float()  # AS2 baseline 0 / 1 / 2 kg
        env.set_payload(env.ids, payload)
        env.command_override = command.clone()
        env._command_views()
        env._snapshot = env._observe()
        env.hist.zero_()
        env.hist.copy_(st.push_history(env.hist,st.frame(env._snapshot['s'],env.last,command,noise=True)))
        env._snapshot['obs'] = env.hist.clone()
        if reset_fn is not None:
            reset_fn(torch.ones(n, device=device, dtype=torch.bool))
        live = torch.ones(n, device=device, dtype=torch.bool)
        positive = torch.zeros(n, device=device)
        negative, xy, yaw, count = (positive.clone() for _ in range(4))
        complete, failed = torch.zeros_like(live), torch.zeros_like(live)
        steps = round(reference_seconds/env.cfg.dt)
        for _ in range(steps):
            action = action_fn(env.snapshot())
            if action.shape != (n, 12) or not torch.isfinite(action).all():
                raise RuntimeError('Mature AS2 inference returned invalid actions')
            terminal, done, _ = env.step(action)
            for key in ('positive_reward','negative_reward','terminated','timeout'):
                if key not in terminal:
                    raise RuntimeError('Calibration requires pre-reset terminal accounting: '+key)
            velocity = st.velocity_body(terminal['s'])
            positive += terminal['positive_reward']*live
            negative += terminal['negative_reward']*live
            xy += (velocity[:,:2]-command[:,:2]).square().sum(-1)*live
            yaw += (terminal['s'][:,9]-command[:,2]).square()*live
            count += live
            failure = terminal['terminated'].bool()
            failed |= live&failure
            complete |= live&terminal['timeout'].bool()&~failure
            live &= ~done
            if reset_fn is not None:
                reset_fn(done)
        complete |= live&(count==steps)
        xy = (xy/count.clamp_min(1)).sqrt()
        yaw = (yaw/count.clamp_min(1)).sqrt()
        qualified = (complete&(count==steps)&~failed
            &(xy<torch.maximum(torch.full_like(xy,.2),.2*command[:,:2].norm(dim=-1)))&(yaw<.3))
        episodes = [{'env':i, 'command':command[i].tolist(), 'payload_kg':float(payload[i]),
            'steps':int(count[i]), 'positive_return':float(positive[i]),
            'negative_return':float(negative[i]), 'net_return':float(positive[i]+negative[i]),
            'complete':bool(complete[i]), 'failure':bool(failed[i]), 'xy_rmse':float(xy[i]),
            'yaw_rmse':float(yaw[i]), 'qualified_complete':bool(qualified[i])} for i in range(n)]
        source = {'policy_path':str(policy_path),
            'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            'branch':branch, 'robot':'as2', 'baseline_commit':'2061be6',
            'backend':'IsaacSim/PhysX', 'reward_iteration':3000,
            'duration':reference_seconds, 'num_envs':n, 'startup_included':True,
            'reference':'maximum net return among complete tracking-qualified episodes'}
        result = build_budget(episodes, source)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        configure_reward_budget(original['budget'])
        env.cfg.variant = original['variant']
        env.iteration = original['iteration']
        env.command_override = original['command_override']
        env.calibration_flat = original['calibration_flat']
        env.evaluation_mode = True
        env.load_curriculum_state_dict(original['curriculum'])
        env.reset(env.ids, advance_curriculum=False)
        env.evaluation_mode = original['evaluation_mode']
        env._snapshot = env._observe()
