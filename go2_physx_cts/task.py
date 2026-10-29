"""Validated13-term task plus two explicitly imported Go2RLLab reward courses.

This is not the complete official15-term reward. Control constants and the
existing13 terms retain the validated PhysX baseline definitions and weights.
"""
import math
from dataclasses import dataclass
import torch
from . import state as st
from .terrain import ground

@dataclass
class RobotConsts:
    default_q: torch.Tensor
    action_scale: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor
    effort: torch.Tensor
    soft_limits: torch.Tensor

WEIGHTS={'track_xy':2.,'track_yaw':1.,'upright':1.,'tilt_rate_l2':-.05,'hard_landing':-.5,
         'crouch':-30.,'joint_deviation_l2':-.1,'base_height_low':-10.,'terminated':-200.,
         'action_rate_l2':-.05,'dof_pos_limits':-1.,'joint_torques_l2':-2e-5,'joint_acc_l2':-2.5e-7}

OFFICIAL_REWARD_SOURCE='wty-yy/go2_rl_lab@6c95d9feef4814c25eb6536e61bb38e1d1d718d4'
BASE_HEIGHT_TARGET=.38
BASE_HEIGHT_CROP_X=(-.2,-.1,0.,.1,.2)
BASE_HEIGHT_CROP_Y=(-.1,0.,.1)

def reward_weights(iteration,stage='flat'):
    """Official linear courses on an explicit caller-owned curriculum clock.

    The original source uses common_step_counter//24. This approved port uses
    the SuperMimic logical learner iteration (one collection step per update),
    so it is not equivalent in physical sample count to a PPO24-step iteration.
    The caller passes current iteration, never a replay record's old timestamp.
    Scalar tensors keep this path differentiable/compilable without host reads.
    """
    if torch.is_tensor(iteration):
        vz_progress=(iteration/1500.).clamp(0.,1.)
        height_progress=(iteration/5000.).clamp(0.,1.)
    else:
        vz_progress=min(1.,max(0.,float(iteration)/1500.))
        height_progress=min(1.,max(0.,float(iteration)/5000.))
    return dict(WEIGHTS,terminated=-600. if stage=='terrain' else -200.,
                lin_vel_z_l2=-2.+2.*vz_progress,
                base_height_l2=-1.-9.*height_progress)

def crop_height_from_map(s,pose,field=None):
    """Known-terrain15-ray crop matching the official0.1m yaw-aligned grid.

    Real physics supplies measured mesh-ray crop height directly to reward().
    Imagination queries the mesh-derived height map at its predicted pose.
    These share the same height definition; map/raycast error remains measurable.
    """
    xs,ys=torch.meshgrid(pose.new_tensor(BASE_HEIGHT_CROP_X),pose.new_tensor(BASE_HEIGHT_CROP_Y),indexing='ij')
    x,y=xs.flatten(),ys.flatten();c,sn=pose[:,2:3].cos(),pose[:,2:3].sin()
    xy=torch.stack((pose[:,0:1]+c*x-sn*y,pose[:,1:2]+sn*x+c*y),-1)
    base_world_z=s[:,0]+ground(pose[:,:2],field)
    return base_world_z-ground(xy,field).mean(-1)

def fall_probability(s):
    tilt=torch.acos((-s[:,3]).clamp(-1+1e-4,1-1e-4))
    return torch.sigmoid((tilt-math.radians(70))/math.radians(5))

def fell(s):return -s[:,3]<math.cos(math.radians(70))

def reward(s0,s1,a,previous,command,c,stage='flat',soft=False,*,iteration=0,pose=None,field=None,
           height_above_crop=None,base_height_target=BASE_HEIGHT_TARGET):
    q=s1[:,st.Q]+c.default_q;lim=c.soft_limits
    tau=(c.kp*(a*c.action_scale+c.default_q-q)-c.kd*s1[:,st.QD]).clamp(-c.effort,c.effort)
    g=s1[:,st.GRAV]
    d=fall_probability(s1) if soft else fell(s1).float()
    velocity_body=st.velocity_body(s1)
    if height_above_crop is None:
        if stage=='terrain' and field is None:
            raise ValueError('Terrain reward requires measured crop height or a known terrain height map.')
        if pose is None:
            if field is not None or stage=='terrain':
                raise ValueError('Terrain reward requires measured crop height or predicted pose/height map.')
            crop_height=s1[:,0]
        else:
            crop_height=crop_height_from_map(s1,pose,field)
    else:
        crop_height=height_above_crop
    if crop_height.shape!=s1.shape[:1]:
        raise ValueError('Crop height must have one scalar per environment.')
    terms={'track_xy':torch.exp(-(velocity_body[:,:2]-command[:,:2]).square().sum(-1)/.25),
           'track_yaw':torch.exp(-(s1[:,9]-command[:,2]).square()/.25),
           'upright':torch.exp(-g[:,:2].square().sum(-1)/.16),
           'tilt_rate_l2':s1[:,7:9].square().sum(-1),
           'hard_landing':(-s1[:,6]).clamp(min=0).square()*(st.feet_world_z(s1).min(-1).values<.05).float(),
           'crouch':(.23-s1[:,0]).clamp(min=0),
           'joint_deviation_l2':s1[:,st.Q].square().sum(-1),
           'base_height_low':(.18-s1[:,0]).clamp(min=0),
           'terminated':d,'action_rate_l2':(a-previous).square().sum(-1),
           'dof_pos_limits':((lim[...,0]-q).clamp(min=0)+(q-lim[...,1]).clamp(min=0)).sum(-1),
           'joint_torques_l2':tau.square().sum(-1),
           'joint_acc_l2':((s1[:,st.QD]-s0[:,st.QD])/.02).square().sum(-1),
           'lin_vel_z_l2':velocity_body[:,2].square(),
           'base_height_l2':(crop_height-base_height_target).square()}
    weights=reward_weights(iteration,stage)
    return sum(weights[k]*v for k,v in terms.items())*.02,terms,d
