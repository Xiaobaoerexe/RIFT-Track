"""Permanent reward-budget calibration and accounting, not a training variant."""
import hashlib,json,math
from pathlib import Path
import torch
from . import state as st
from .task import configure_reward_budget

def build_budget(episodes,source):
    complete=[e for e in episodes if e['qualified_complete']]
    if not complete:raise RuntimeError('No qualified complete official-policy flat episode; cannot invent reward budget')
    ref=max(e['net_return'] for e in complete)
    if ref<=0:raise RuntimeError('Official controller reference return is not positive')
    cap=.9*ref
    rates=[3.,3.,3.,2.25,2.25,3.,3.2,1.5,1.5,1.5]
    nominal=[20.,20.,20.,14.,14.,6.,8.,12.,10.,10.]
    tasks=[]
    for i,(rate,length) in enumerate(zip(rates,nominal)):
        if i:
            length=min(length,cap/rate)
            if i==5:length=min(length,.45*cap/rate)
            length=math.floor((length+1e-8)/.02)*.02
        if length<3:raise RuntimeError('Measured reference cannot support feasible unscaled task budgets')
        allowance=max(0.,cap-rate*length) if i else 0.
        tasks.append({'task_id':i,'episode_seconds':length,'base_positive_rate_bound':rate,
                      'base_positive_return_bound':rate*length,
                      'success_bonus':allowance if i in (3,4,5,6,8,9) else 0.,
                      'progress_budget':allowance if i==7 else 0.})
    values=torch.tensor([e['net_return'] for e in complete],dtype=torch.float64)
    result={'schema':'depth_reward_budget_v1','reference_net_max':ref,'positive_cap':cap,
            'failure_penalty':.1*ref,'obstacle_head_failure':18.,'source':source,
            'qualified_episodes':len(complete),'total_episodes':len(episodes),
            'reference_statistics':{'mean':float(values.mean()),'median':float(values.median()),
                                    'min':float(values.min()),'max':float(values.max())},
            'tasks':tasks,'episodes':episodes,
            'accounting':'Rates multiply .02; one-time success/failure/progress events do not. Negative costs never offset positive caps. Shared coefficients unchanged.'}
    configure_reward_budget(result)
    return result


@torch.no_grad()
def calibrate_official_reward(env,driver,output,source_policy):
    """One complete first episode per env, startup included, no reset mixing.

    The environment remains on its assigned initial flat map during the cohort.
    Final reward courses are evaluated at iteration3000, independent of resets.
    """
    if driver is None:raise ValueError('Official policy driver is required')
    n=env.num_envs;dev=env.device;old_iteration=env.iteration
    env.evaluation_mode=True;env.iteration=0;env.command_override=None
    env.reset(env.ids)
    env.place_calibration_flat()
    env.iteration=3000
    commands=torch.tensor([[0,0,0],[.5,0,0],[.5,.2,.4],[1,0,0],[-.5,0,0]],device=dev)
    row=torch.arange(n,device=dev);cmd=commands[row%len(commands)]
    loads=(row//len(commands)%3).float()*.6
    env.set_payload(env.ids,loads)
    env.command_override=cmd.clone();env.command=cmd.clone();env._snapshot=env._observe()
    driver.reset(torch.ones(n,device=dev,dtype=torch.bool))
    live=torch.ones(n,device=dev,dtype=torch.bool)
    pos=torch.zeros(n,device=dev);neg=pos.clone();xy=pos.clone();yaw=pos.clone();count=pos.clone()
    complete=torch.zeros_like(live);failed=torch.zeros_like(live)
    duration=20.;steps=round(duration/env.cfg.dt)
    try:
        for i in range(steps):
            action=driver.act(env.command)
            after,done,info=env.step(action)
            if 'reward' not in after or 'positive_reward' not in after:
                raise RuntimeError('Calibration needs per-environment terminal reward accounting')
            terminal=after
            v=st.velocity_body(terminal['s'])
            pos+=after['positive_reward']*live;neg+=after['negative_reward']*live
            xy+=(v[:,:2]-cmd[:,:2]).square().sum(-1)*live
            yaw+=(terminal['s'][:,9]-cmd[:,2]).square()*live;count+=live
            fail=after['terminated'].bool()
            failed|=live&fail
            complete|=live&after['timeout'].bool()&(~fail)
            live&=~done
            driver.reset(done)
        complete|=live&(count>=steps)
        xy=(xy/count.clamp_min(1)).sqrt();yaw=(yaw/count.clamp_min(1)).sqrt()
        qualified=complete&(count==steps)&(~failed)&(xy<torch.maximum(torch.full_like(xy,.2),.2*cmd[:,:2].norm(dim=-1)))&(yaw<.3)
        episodes=[]
        for i in range(n):
            episodes.append({'env':i,'command':cmd[i].tolist(),'payload_kg':float(loads[i]),
                'steps':int(count[i]),'positive_return':float(pos[i]),'negative_return':float(neg[i]),
                'net_return':float(pos[i]+neg[i]),'complete':bool(complete[i]),'failure':bool(failed[i]),
                'xy_rmse':float(xy[i]),'yaw_rmse':float(yaw[i]),'qualified_complete':bool(qualified[i])})
        path=Path(source_policy)
        source={'policy_path':str(path),'policy_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'backend':'IsaacSim/PhysX','reward_iteration':3000,'duration':duration,
                'num_envs':n,'startup_included':True,'reference':'maximum net return among complete tracking-qualified episodes'}
        result=build_budget(episodes,source)
        Path(output).write_text(json.dumps(result,indent=2))
        return result
    finally:
        env.command_override=None;env.iteration=old_iteration;env.evaluation_mode=False
        env.reset(env.ids);env._snapshot=env._observe()
