"""AS2 motion rewards plus once-only crossing completion; direct BPTT surrogate."""
import json
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from . import state as st
from .terrain import ground

DT = .02
TASK_CONTEXT_DIM = 62
STABLE_MASS = slice(21, 53)
PREVIOUS_COMMAND = slice(53, 56)
CHANGE_AGE, CHANGE_STRENGTH = 56, 57
POST_COMMAND = slice(58, 61)
COMMAND_INITIALIZED = 61
GOAL = slice(0, 3)
COMPLETED, STABLE, ELAPSED = 3, 4, 5
ROUTE = slice(6, 8)
ROUTE_YAW, PROGRESS = 8, 9
ORIGIN = slice(10, 12)
EPISODE_LIMIT, RESERVED, TARGET_RADIUS, ROUTE_LENGTH = 12, 13, 14, 15
PRIOR_SUPPORT = slice(16, 20)
REQUIRED_STABLE_SECONDS = 20
FLAT_TERRAIN_ID, GAP_TERRAIN_ID = 7, 10
from .robots import profile
BASE_HEIGHT_TARGET = profile('as2').reward_height
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


def initial_task_context(pose, goal_xyz=None, episode_limit=20.,
                         target_radius=.8, route_length=5., stable_seconds=.32):
    ctx = pose.new_zeros((len(pose), TASK_CONTEXT_DIM))
    ctx[:, GOAL] = pose[:, :3] if goal_xyz is None else goal_xyz
    ctx[:, ORIGIN] = pose[:, :2]
    ctx[:, ROUTE] = pose[:, :2]
    ctx[:, ROUTE_YAW] = pose_yaw(pose)
    for key, value in ((EPISODE_LIMIT, episode_limit),
                       (TARGET_RADIUS, target_radius), (ROUTE_LENGTH, route_length),
                       (REQUIRED_STABLE_SECONDS, stable_seconds)):
        ctx[:, key] = torch.as_tensor(value, device=pose.device, dtype=pose.dtype)
    ctx[:, STABLE_MASS.start] = 1.
    return ctx


def configure_reward_budget(path_or_record):
    """Reject the old Go2 ten-task calibration rather than silently reusing it."""
    global _BUDGET
    if path_or_record is None:
        _BUDGET = None
        return None
    value = (json.loads(Path(path_or_record).read_text(encoding='utf-8'))
             if isinstance(path_or_record, (str, Path)) else dict(path_or_record))
    if value.get('schema') != 'as2_depth_crossing_budget_v2' or value.get('robot') != 'as2':
        raise ValueError('A measured AS2 four-task crossing calibration is required')
    if len(value.get('tasks', [])) != 4 or value.get('reference_net_max', 0.) <= 0:
        raise ValueError('Invalid AS2 reward calibration')
    if not value.get('source', {}).get('policy_sha256'):
        raise ValueError('Calibration must identify its mature AS2 policy')
    for task in (3,):
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


def _smooth_tail(x, threshold):
    # Equal value and derivative at the join; no dead-gradient penalty cap.
    return torch.where(x <= threshold, x, 2*torch.sqrt(threshold*x.clamp_min(threshold))-threshold)


def smoothstep(x, low, high):
    t=((x-low)/(high-low)).clamp(0.,1.)
    return t.square()*(3-2*t)


def motion_terms(s0, s1, a, previous, command, c, *, pose, telemetry, previous2,
                 on_flat, iteration=0, field=None, height_above_crop=None, soft_contacts=False,
                 base_height_target=BASE_HEIGHT_TARGET, transient_alpha=None):
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
    transient = torch.zeros_like(alpha) if transient_alpha is None else transient_alpha*on_flat
    alpha = torch.maximum(alpha, transient)
    tracking = (1-alpha)*torch.exp(-error/.25)+alpha*torch.exp(-error/1.)
    yaw_error=(s1[:,9]-command[:,2]).square()
    terms = {'track_xy':tracking, 'track_yaw':(1-transient)*torch.exp(-yaw_error/.25)+transient*torch.exp(-yaw_error),
        'lin_vel_z_l2':vb[:, 2].square(), 'ang_vel_xy_l2':s1[:, 7:9].square().sum(-1),
        'joint_acc_l2':_capped(acc.square().sum(-1), 1e8), 'joint_power':(tau*s1[:, st.QD]).abs().sum(-1),
        'joint_torques_l2':tau.square().sum(-1), 'base_height_l2':(crop-base_height_target).square(),
        'action_rate_l2':_smooth_tail((a-previous).square().sum(-1), 20.),
        'action_smoothness_l2':_smooth_tail(smooth.sum(-1), 50.), 'undesired_contacts':contact.sum(-1),
        'joint_pos_limits':((lim[..., 0]-q).clamp_min(0)+(q-lim[..., 1]).clamp_min(0)).sum(-1),
        'feet_regulation':(feet_v.square().sum(-1)*torch.exp(-feet_h/(.025*base_height_target))).sum(-1),
        'hip_pos_penalty_l1':hip, 'joint_pos_penalty_l1':legs}
    weights = reward_weights(iteration)
    return terms, {key:weights[key]*value*DT for key, value in terms.items()}, crop


def finite_horizon_mask(task_ids, terminated, timeout, variant='simple'):
    finite = (task_ids.long() == 3) if variant != 'simple' else torch.zeros_like(timeout, dtype=torch.bool)
    return (1-terminated.to(torch.float32))*(~(timeout.bool()&finite)).to(torch.float32)


def completion_step(mass, completed, eligible, required_steps):
    """Finite-state run-length mass, not a calibrated success probability.

    Hard eligibility reproduces the integer consecutive-stability automaton.
    Absorbing completion guarantees that total awarded increment is <= 1.
    """
    required=required_steps.long().clamp(1,32)
    increment=eligible*mass.gather(1,(required-1)[:,None])[:,0]
    shifted=torch.cat(((1-eligible)[:,None]*(1-completed)[:,None],eligible[:,None]*mass[:,:-1]),-1)
    shifted=shifted*(torch.arange(32,device=mass.device)[None,:]<required[:,None])
    return shifted,completed+increment,increment


def task_reward(s0,s1,a,previous,command,c,*,task_id,terrain_id,context,pose0,
                pose,telemetry,previous2=None,history_valid=None,iteration=0,field=None,
                soft=False,height_above_crop=None,variant='simple',base_height_target=BASE_HEIGHT_TARGET):
    tasks=task_id.reshape(-1).long();zero=s1.new_zeros(len(s1));ctx=context.clone()
    delta=command-context[:,PREVIOUS_COMMAND]
    change=torch.maximum(delta[:,:2].norm(dim=-1)/.5,delta[:,2].abs())
    switched=(change>1e-6)&(context[:,COMMAND_INITIALIZED]>.5)
    age=torch.where(switched,zero,context[:,CHANGE_AGE]+DT)
    strength=torch.where(switched,.25*change.clamp(0,1),context[:,CHANGE_STRENGTH])
    transient=strength*(1-age/.3).clamp(0,1)
    ctx[:,PREVIOUS_COMMAND]=command;ctx[:,CHANGE_AGE]=age
    ctx[:,CHANGE_STRENGTH]=strength;ctx[:,COMMAND_INITIALIZED]=1.
    ctx[:,ELAPSED]=advance_episode_time(context[:,ELAPSED])
    timeout=episode_time_reached(ctx[:,ELAPSED],context[:,EPISODE_LIMIT])
    raw,weighted,crop=motion_terms(s0,s1,a,previous,command,c,pose=pose,telemetry=telemetry,
        previous2=previous2,on_flat=terrain_id.reshape(-1)==FLAT_TERRAIN_ID,iteration=iteration,
        field=field,height_above_crop=height_above_crop,soft_contacts=soft,
        base_height_target=base_height_target,transient_alpha=transient)
    special=(tasks==3) if variant!='simple' else torch.zeros_like(tasks,dtype=torch.bool)
    # Only the approved crossing height coefficient differs from base motion.
    weighted['base_height_l2']=torch.where(special,-raw['base_height_l2']*DT,weighted['base_height_l2'])
    motion=sum(weighted.values());success=zero;waiting=zero;failure_cost=zero;bonus=zero;eligible=zero
    failure=fall_probability(s1) if soft else fell(s1)
    if variant!='simple':
        if _BUDGET is None:raise ValueError('Crossing budget must be calibrated')
        contact=telemetry['contact_prob'];support=contact[:,:4]
        collision=contact[:,4:6].amax(-1)
        failure=torch.maximum(failure,collision*special) if soft else failure|((collision>=.5)&special)
        offset=pose[:,:2]-context[:,GOAL][:,:2]
        direction=context[:,GOAL][:,:2]-context[:,ORIGIN]
        direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        along=(offset*direction).sum(-1)+context[:,TARGET_RADIUS]
        lateral=context[:,TARGET_RADIUS]-(offset[:,0]*direction[:,1]-offset[:,1]*direction[:,0]).abs()
        height_error=(pose[:,2]-context[:,GOAL][:,2]-base_height_target).abs()
        upright=-s1[:,3];omega=s1[:,7:10].norm(dim=-1)
        if soft:
            feet=smoothstep(support,.2,.8).sum(-1)
            eligible=(smoothstep(along,0.,.05)*smoothstep(lateral,0.,.05)
                *smoothstep(upright,math.cos(math.radians(35)),math.cos(math.radians(30)))
                *smoothstep(crop,base_height_target-.06,base_height_target-.03)
                *smoothstep(-omega,-3.,-2.5)*smoothstep(-height_error,-.28,-.23)
                *smoothstep(feet,1.,2.))*special*(1-failure)
            required=torch.round(context[:,REQUIRED_STABLE_SECONDS]/DT).long()
            mass,completed,success=completion_step(context[:,STABLE_MASS],context[:,COMPLETED],eligible,required)
            ctx[:,STABLE_MASS]=mass;ctx[:,COMPLETED]=completed
            ctx[:,STABLE]=(mass*torch.arange(32,device=s1.device)[None,:]).sum(-1)*DT
        else:
            eligible=((along>=0)&(lateral>=0)&(upright>=math.cos(math.radians(35)))
                &(crop>=base_height_target-.06)&(omega<3.)&(height_error<.28)&(support.sum(-1)>=2)&special&~failure)
            count=torch.where(eligible,torch.round(context[:,STABLE]/DT).long()+1,torch.zeros_like(tasks))
            required=torch.round(context[:,REQUIRED_STABLE_SECONDS]/DT).long()
            success=((count>=required)&(context[:,COMPLETED]<.5)).to(s1.dtype)
            completed=torch.maximum(context[:,COMPLETED],success)
            ctx[:,COMPLETED]=completed;ctx[:,STABLE]=count.to(s1.dtype)*DT
            ctx[:,STABLE_MASS]=torch.nn.functional.one_hot(count.clamp_max(31),32).to(s1.dtype)*(1-completed[:,None])
        row=_BUDGET['tasks'][3]
        bonus=float(row['success_bonus'])*success
        waiting=-float(row['unfinished_rate'])*(1-ctx[:,COMPLETED])*DT*special
        failure_cost=-float(_BUDGET['failure_penalty'])*failure.to(s1.dtype)*special
        ctx[:,PRIOR_SUPPORT]=support
    total=motion+bonus+waiting+failure_cost
    positive=weighted['track_xy']+weighted['track_yaw']+bonus
    terms={**raw,'continuous_reward':total,'event_reward':bonus+waiting+failure_cost,
        'positive':positive,'negative':total-positive,'event_success':bonus,'event_failure':failure_cost,
        'unfinished_time':waiting,'completed':ctx[:,COMPLETED],'success_probability':eligible.to(s1.dtype),
        'bootstrap_mask':finite_horizon_mask(tasks,failure,timeout,variant)}
    return total,terms,ctx,failure,success
