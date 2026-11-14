"""Pinned AS2 motion rewards and real events; no imagined soft success labels."""
import json
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from . import state as st
from .terrain import ground

DT = .02
TASK_CONTEXT_DIM = 21
GOAL = slice(0, 3)
COMPLETED, STABLE, ELAPSED = 3, 4, 5
ROUTE = slice(6, 8)
ROUTE_YAW, PROGRESS = 8, 9
ORIGIN = slice(10, 12)
EPISODE_LIMIT, RECOVERY_DEADLINE, TARGET_RADIUS, ROUTE_LENGTH = 12, 13, 14, 15
PRIOR_SUPPORT = slice(16, 20)
REQUIRED_STABLE_SECONDS = 20
FLAT_TERRAIN_ID, GAP_TERRAIN_ID = 7, 10
BASE_HEIGHT_TARGET = .30
BASE_HEIGHT_CROP_X = (-.2, -.1, 0., .1, .2)
BASE_HEIGHT_CROP_Y = (-.1, 0., .1)
OFFICIAL_REWARD_SOURCE = 'wty-yy/go2_rl_lab@6c95d9feef4814c25eb6536e61bb38e1d1d718d4'
AS2_BASELINE_COMMIT = '2061be6'
WEIGHTS = {'track_xy':2., 'track_yaw':1., 'lin_vel_z_l2':-2., 'ang_vel_xy_l2':-.05,
    'joint_acc_l2':-1e-7, 'joint_power':-2e-5, 'joint_torques_l2':-1e-4,
    'base_height_l2':-1., 'action_rate_l2':-.01, 'action_smoothness_l2':-.01,
    'undesired_contacts':-1., 'joint_pos_limits':-2., 'feet_regulation':-.05,
    'hip_pos_penalty_l1':-.05, 'joint_pos_penalty_l1':-.01}
_BUDGET = None


@dataclass
class RobotConsts:
    default_q: torch.Tensor
    action_scale: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor
    effort: torch.Tensor
    soft_limits: torch.Tensor


def reward_weights(iteration, stage='flat'):
    progress = ((iteration / 3000.).clamp(0., 1.) if torch.is_tensor(iteration)
                else min(1., max(0., float(iteration) / 3000.)))
    return dict(WEIGHTS, lin_vel_z_l2=-2.+2.*progress, base_height_l2=-1.-9.*progress)


def advance_episode_time(elapsed):
    return (torch.round(elapsed.float()/DT)+1)*DT


def episode_time_reached(elapsed, limit):
    return torch.round(elapsed.float()/DT)*DT >= limit


def pose_yaw(pose):
    return st.yaw_from_pose(pose)


def initial_task_context(pose, goal_xyz=None, episode_limit=20., recovery_deadline=8.,
                         target_radius=.8, route_length=5., stable_seconds=.32):
    ctx = pose.new_zeros((len(pose), TASK_CONTEXT_DIM))
    ctx[:, GOAL] = pose[:, :3] if goal_xyz is None else goal_xyz
    ctx[:, ORIGIN] = pose[:, :2]
    ctx[:, ROUTE] = pose[:, :2]
    ctx[:, ROUTE_YAW] = pose_yaw(pose)
    for key, value in ((EPISODE_LIMIT, episode_limit), (RECOVERY_DEADLINE, recovery_deadline),
                       (TARGET_RADIUS, target_radius), (ROUTE_LENGTH, route_length),
                       (REQUIRED_STABLE_SECONDS, stable_seconds)):
        ctx[:, key] = torch.as_tensor(value, device=pose.device, dtype=pose.dtype)
    return ctx


def configure_reward_budget(path_or_record):
    """Reject the old Go2 ten-task calibration rather than silently reusing it."""
    global _BUDGET
    if path_or_record is None:
        _BUDGET = None
        return None
    value = (json.loads(Path(path_or_record).read_text(encoding='utf-8'))
             if isinstance(path_or_record, (str, Path)) else dict(path_or_record))
    if value.get('schema') != 'as2_depth_event_budget_v1' or value.get('robot') != 'as2':
        raise ValueError('A measured AS2 five-task event calibration is required')
    if len(value.get('tasks', [])) != 5 or value.get('reference_net_max', 0.) <= 0:
        raise ValueError('Invalid AS2 reward calibration')
    if not value.get('source', {}).get('policy_sha256'):
        raise ValueError('Calibration must identify its mature AS2 policy')
    for task in (3, 4):
        row = value['tasks'][task]
        if row['success_bonus'] <= 0 or row['episode_seconds'] < 6.:
            raise ValueError('Measured budget cannot support the event task')
        if row['base_positive_rate_bound']*row['episode_seconds']+row['success_bonus'] > value['positive_cap']+1e-5:
            raise ValueError('Event task allowance exceeds the measured AS2 cap')
    _BUDGET = value
    return value


def reward_budget():
    return _BUDGET


def crop_height_from_map(s, pose, field=None):
    """Same AS2 fifteen yaw-aligned rays, adapted only to xyz/quaternion pose."""
    xs, ys = torch.meshgrid(pose.new_tensor(BASE_HEIGHT_CROP_X), pose.new_tensor(BASE_HEIGHT_CROP_Y), indexing='ij')
    x, y = xs.flatten(), ys.flatten()
    yaw = pose_yaw(pose)[:, None]
    xy = torch.stack((pose[:, 0:1]+yaw.cos()*x-yaw.sin()*y,
                      pose[:, 1:2]+yaw.sin()*x+yaw.cos()*y), -1)
    return pose[:, 2]-ground(xy, field).mean(-1)


def fall_probability(s):
    tilt = torch.acos((-s[:, 3]).clamp(-1+1e-4, 1-1e-4))
    return torch.sigmoid((tilt-math.radians(70))/math.radians(5))


def fell(s):
    return -s[:, 3] < math.cos(math.radians(70))


def _capped(x, maximum):
    return torch.where(torch.isfinite(x), x.clamp_max(maximum), x)


def motion_terms(s0, s1, a, previous, command, c, *, pose, telemetry, previous2,
                 on_flat, iteration=0, field=None, height_above_crop=None, soft_contacts=False,
                 base_height_target=BASE_HEIGHT_TARGET):
    """Exact fifteen AS2 formulas; no task-specific hidden rescaling."""
    if previous2 is None:
        raise ValueError('AS2 action smoothness requires two previous actions')
    required = ('applied_torque', 'joint_acceleration', 'foot_velocity', 'log_contact_force')
    missing = [key for key in required if key not in telemetry]
    if missing:
        raise ValueError(f'Missing AS2 reward telemetry: {missing}')
    tau, acc = telemetry['applied_torque'], telemetry['joint_acceleration']
    feet_v = telemetry['foot_velocity'].reshape(-1, 4, 3)[..., :2]
    logforce = telemetry['log_contact_force']
    contact = (torch.sigmoid((logforce-math.log1p(5.))/.15) if soft_contacts
               else (logforce > math.log1p(5.)).to(s1.dtype))
    crop = height_above_crop if height_above_crop is not None else crop_height_from_map(s1, pose, field)
    q = s1[:, st.Q]+c.default_q
    lim = c.soft_limits
    vb = st.velocity_body(s1)
    feet_h = (crop[:, None]+s1[:, st.FEET].reshape(-1, 4, 3)[..., 2]).clamp_min(0.)
    smooth = (a-2*previous+previous2).square()*(previous != 0)*(previous2 != 0)
    hip = s1[:, st.Q].reshape(-1, 4, 3)[:, :, 0].abs().sum(-1)
    legs = s1[:, st.Q].reshape(-1, 4, 3)[:, :, 1:].abs().sum((-1, -2))
    hip = hip*torch.where((command[:, 1:].norm(dim=-1)<1e-3)&on_flat, 10., 1.)
    legs = legs*torch.where((command.norm(dim=-1)<1e-3)&on_flat, 10., 1.)
    error = (vb[:, :2]-command[:, :2]).square().sum(-1)
    alpha = .25*((command[:, 0].abs()-2.)/1.).clamp(0., 1.)*on_flat.to(s1.dtype)
    tracking = (1-alpha)*torch.exp(-error/.25)+alpha*torch.exp(-error/1.)
    terms = {'track_xy':tracking, 'track_yaw':torch.exp(-(s1[:, 9]-command[:, 2]).square()/.25),
        'lin_vel_z_l2':vb[:, 2].square(), 'ang_vel_xy_l2':s1[:, 7:9].square().sum(-1),
        'joint_acc_l2':_capped(acc.square().sum(-1), 1e8), 'joint_power':(tau*s1[:, st.QD]).abs().sum(-1),
        'joint_torques_l2':tau.square().sum(-1), 'base_height_l2':(crop-base_height_target).square(),
        'action_rate_l2':_capped((a-previous).square().sum(-1), 20.),
        'action_smoothness_l2':_capped(smooth.sum(-1), 50.), 'undesired_contacts':contact.sum(-1),
        'joint_pos_limits':((lim[..., 0]-q).clamp_min(0)+(q-lim[..., 1]).clamp_min(0)).sum(-1),
        'feet_regulation':(feet_v.square().sum(-1)*torch.exp(-feet_h/(.025*base_height_target))).sum(-1),
        'hip_pos_penalty_l1':hip, 'joint_pos_penalty_l1':legs}
    weights = reward_weights(iteration)
    return terms, {key:weights[key]*value*DT for key, value in terms.items()}, crop


def finite_horizon_mask(task_ids, terminated, timeout, variant='simple'):
    """AS2 time limits bootstrap; crossing/recovery deadlines are finite."""
    finite = (task_ids.long() >= 3) if variant != 'simple' else torch.zeros_like(timeout, dtype=torch.bool)
    return (1-terminated.to(torch.float32))*(~(timeout.bool()&finite)).to(torch.float32)


def task_reward(s0, s1, a, previous, command, c, *, task_id, terrain_id, context, pose0,
                pose, telemetry, previous2=None, history_valid=None, iteration=0, field=None,
                soft=False, height_above_crop=None, variant='simple', base_height_target=BASE_HEIGHT_TARGET):
    """Return total, terms, context, failure, first-success.

    soft=True is motion imagination. In event variants completed/stable remain
    real Replay memory: no predicted soft completion is used as supervision.
    """
    if variant not in ('simple', 'events', 'moe'):
        raise ValueError(f'Unknown AS2 depth variant {variant}')
    tasks = task_id.reshape(-1).long()
    zero = s1.new_zeros(len(s1)); ctx = context.clone()
    completed0 = context[:, COMPLETED] >= .5
    recovering = (tasks == 4)&~completed0 if variant != 'simple' else torch.zeros_like(completed0)
    # Continuous formulas never switch on imagined completion. All hard recovery
    # mode changes are exact corrections in the real-event return below.
    on_flat = terrain_id.reshape(-1) == FLAT_TERRAIN_ID
    raw, weighted, crop = motion_terms(s0, s1, a, previous, command, c, pose=pose,
        telemetry=telemetry, previous2=previous2, on_flat=on_flat, iteration=iteration,
        field=field, height_above_crop=height_above_crop, soft_contacts=soft and variant=='simple',
        base_height_target=base_height_target)
    ctx[:, ELAPSED] = advance_episode_time(context[:, ELAPSED])
    timeout = episode_time_reached(ctx[:, ELAPSED], context[:, EPISODE_LIMIT])
    if variant == 'simple':
        continuous = sum(weighted.values())
        failure = fall_probability(s1) if soft else fell(s1)
        positive = weighted['track_xy']+weighted['track_yaw']
        terms = {**raw, 'continuous_reward':continuous, 'event_reward':zero,
            'positive':positive, 'negative':continuous-positive, 'event_success':zero,
            'event_failure':zero, 'completed':context[:, COMPLETED], 'success_probability':zero,
            'bootstrap_mask':finite_horizon_mask(tasks, failure, timeout, variant)}
        return continuous, terms, ctx, failure, zero

    if _BUDGET is None:
        raise ValueError('Event experiments require a measured AS2 reward calibration')
    if 'contact_prob' not in telemetry:
        raise ValueError('Task events require actual/predicted contact telemetry')
    contact = telemetry['contact_prob']; support = contact[:, :4]
    collision = contact[:, 4:6].amax(-1) >= .5
    normal_fail = fell(s1) | ((tasks == 3)&collision)
    deadline = episode_time_reached(ctx[:, ELAPSED], context[:, RECOVERY_DEADLINE])
    failure_hard = torch.where(recovering, deadline, normal_fail)
    success = torch.zeros_like(completed0)
    if not soft:
        delta = pose[:, :2]-context[:, GOAL][:, :2]
        direction = context[:, GOAL][:, :2]-context[:, ORIGIN]
        direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        along = (delta*direction).sum(-1)
        lateral = (delta[:, 0]*direction[:, 1]-delta[:, 1]*direction[:, 0]).abs()
        region = ((along >= -context[:, TARGET_RADIUS]) & (lateral <= context[:, TARGET_RADIUS])
                  & ((pose[:, 2]-context[:, GOAL][:, 2]-base_height_target).abs() < .28))
        eligible = ((-s1[:, 3] >= math.cos(math.radians(35))) & (support.sum(-1) >= 2.)
                    & (crop >= .24) & (s1[:, 7:10].norm(dim=-1) < 3.)
                    & ((tasks == 4)|region) & ((tasks == 3)|(tasks == 4)))
        count = torch.round(context[:, STABLE]/DT).long()
        count = torch.where(eligible, count+1, torch.zeros_like(count))
        # The identical threshold is visible to Ue through task context. Terrain
        # geometry can become indistinguishable on the landing platform; the
        # Critic must not have to guess whether this task requires 16 or 32 ticks.
        required = torch.round(context[:, REQUIRED_STABLE_SECONDS]/DT).long()
        success = ~completed0 & (count >= required) & ~failure_hard
        ctx[:, STABLE] = count.to(ctx.dtype)*DT
        ctx[:, COMPLETED] = (completed0|success).to(ctx.dtype)
    ctx[:, PRIOR_SUPPORT] = support
    completed1 = ctx[:, COMPLETED] >= .5
    # Exact real recovery gate = smooth tracking + a hard event correction.
    contact_cost = torch.where(recovering, zero, weighted['undesired_contacts'])
    continuous = sum(v for key, v in weighted.items() if key != 'undesired_contacts')
    tracking = weighted['track_xy']+weighted['track_yaw']
    velocity_correction = torch.where((tasks == 4)&~completed1, -tracking, zero)
    upright = ((1-s1[:, 3])/2).clamp(0, 1)
    recovery_shape = torch.where(tasks == 4,
        (upright+.5*(crop/base_height_target).clamp(0, 1))*DT, zero)
    continuous = continuous+recovery_shape
    shape_correction = torch.where((tasks == 4)&completed1, -recovery_shape, zero)
    # AS2 uses 10x posture penalties on flat zero/lateral-zero commands. Before
    # getting up, retain the ordinary 1x penalty; never remove the penalty itself.
    hip_boost = on_flat&(command[:,1:].norm(dim=-1)<1e-3)
    leg_boost = on_flat&(command.norm(dim=-1)<1e-3)
    posture_correction = torch.where(recovering,
        -.9*(weighted['hip_pos_penalty_l1']*hip_boost
             +weighted['joint_pos_penalty_l1']*leg_boost), zero)
    amounts = s1.new_tensor([row['success_bonus'] for row in _BUDGET['tasks']])[tasks]
    waiting_rate = s1.new_tensor([row.get('unfinished_rate', 0.) for row in _BUDGET['tasks']])[tasks]
    event_success = amounts*success.to(s1.dtype)
    waiting = -waiting_rate*(~completed1).to(s1.dtype)*DT
    event_failure = -float(_BUDGET['failure_penalty'])*failure_hard.to(s1.dtype)*(tasks >= 3).to(s1.dtype)
    event = (contact_cost+velocity_correction+shape_correction+posture_correction
             +waiting+event_success+event_failure)
    if soft:
        # Same AS2 motion-continuation surrogate, not a second event penalty.
        failure = torch.where(recovering, deadline.to(s1.dtype), fall_probability(s1))
        event = zero; event_success = zero; event_failure = zero
    else:
        failure = failure_hard
    total = continuous+event
    true_tracking = tracking+velocity_correction if not soft else tracking
    true_shape = recovery_shape+shape_correction if not soft else recovery_shape
    positive = true_tracking+true_shape+event_success
    terms = {**raw, 'recovery_shape':recovery_shape, 'continuous_reward':continuous,
        'event_reward':event, 'positive':positive, 'negative':total-positive,
        'event_success':event_success, 'event_failure':event_failure,
        'unfinished_time':waiting if not soft else zero,
        'recovery_velocity_correction':velocity_correction if not soft else zero,
        'recovery_shape_correction':shape_correction if not soft else zero,
        'recovery_posture_correction':posture_correction if not soft else zero,
        'completed':ctx[:, COMPLETED], 'success_probability':success.to(s1.dtype),
        'bootstrap_mask':finite_horizon_mask(tasks, failure, timeout, variant)}
    return total, terms, ctx, failure, success.to(s1.dtype)
