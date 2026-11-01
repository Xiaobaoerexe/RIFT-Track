"""Pinned PhysX15 reward port. Contact threshold is softened only in imagination."""
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

WEIGHTS={'track_xy':2.,'track_yaw':1.,'lin_vel_z_l2':-2.,'ang_vel_xy_l2':-.05,
 'joint_acc_l2':-1e-7,'joint_power':-2e-5,'joint_torques_l2':-1e-4,
 'base_height_l2':-1.,'action_rate_l2':-.01,'action_smoothness_l2':-.01,
 'undesired_contacts':-1.,'joint_pos_limits':-2.,'feet_regulation':-.05,
 'hip_pos_penalty_l1':-.05,'joint_pos_penalty_l1':-.01}

def reward_weights(iteration,stage='flat'):
    progress=(iteration/3000.).clamp(0.,1.) if torch.is_tensor(iteration) else min(1.,max(0.,float(iteration)/3000.))
    return dict(WEIGHTS,lin_vel_z_l2=-2.+2.*progress,base_height_l2=-1.-9.*progress)

OFFICIAL_REWARD_SOURCE='wty-yy/go2_rl_lab@6c95d9feef4814c25eb6536e61bb38e1d1d718d4'
BASE_HEIGHT_TARGET=.38
BASE_HEIGHT_CROP_X=(-.2,-.1,0.,.1,.2)
BASE_HEIGHT_CROP_Y=(-.1,0.,.1)

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
           height_above_crop=None,base_height_target=BASE_HEIGHT_TARGET,aux=None,previous_previous=None,on_flat=None):
    if aux is None:raise ValueError('Official rewards require measured/predicted reward telemetry')
    if previous_previous is None:raise ValueError('Second-order action reward requires two previous actions')
    if on_flat is None:
        if stage=='terrain':raise ValueError('Official posture reward requires actual terrain classification')
        on_flat=torch.ones(len(s1),dtype=torch.bool,device=s1.device)
    q=s1[:,st.Q]+c.default_q;lim=c.soft_limits
    tau=aux[:,st.AUX_TORQUE];acc=aux[:,st.AUX_ACC]
    vfeet=aux[:,st.AUX_FEET_VXY].reshape(-1,4,2)
    logforce=aux[:,st.AUX_LOG_FORCE]
    # Hard force threshold in real physics; differentiable surrogate ONLY in imagination.
    contact=torch.sigmoid((logforce-math.log1p(5.))/.15) if soft else (logforce>math.log1p(5.)).to(s1.dtype)
    crop=height_above_crop if height_above_crop is not None else crop_height_from_map(s1,pose,field)
    vb=st.velocity_body(s1)
    feet_h=(crop[:,None]+s1[:,st.FEET].reshape(-1,4,3)[:,:,2]).clamp_min(0.)
    smooth=(a-2*previous+previous_previous).square()*(previous!=0)*(previous_previous!=0)
    hip=s1[:,st.Q].reshape(-1,4,3)[:,:,0].abs().sum(-1)
    legs=s1[:,st.Q].reshape(-1,4,3)[:,:,1:].abs().sum((-1,-2))
    hip=hip*torch.where((command[:,1:].norm(dim=-1)<1e-3)&on_flat,10.,1.)
    legs=legs*torch.where((command.norm(dim=-1)<1e-3)&on_flat,10.,1.)
    def capped(x,maximum):return torch.where(torch.isfinite(x),x.clamp_max(maximum),x)
    terms={'track_xy':torch.exp(-(vb[:,:2]-command[:,:2]).square().sum(-1)/.25),
      'track_yaw':torch.exp(-(s1[:,9]-command[:,2]).square()/.25),
      'lin_vel_z_l2':vb[:,2].square(),'ang_vel_xy_l2':s1[:,7:9].square().sum(-1),
      'joint_acc_l2':capped(acc.square().sum(-1),1e8),'joint_power':(tau*s1[:,st.QD]).abs().sum(-1),
      'joint_torques_l2':tau.square().sum(-1),'base_height_l2':(crop-base_height_target).square(),
      'action_rate_l2':capped((a-previous).square().sum(-1),20.),
      'action_smoothness_l2':capped(smooth.sum(-1),50.),
      'undesired_contacts':contact.sum(-1),
      'joint_pos_limits':((lim[...,0]-q).clamp_min(0)+(q-lim[...,1]).clamp_min(0)).sum(-1),
      'feet_regulation':(vfeet.square().sum(-1)*torch.exp(-feet_h/(.025*base_height_target))).sum(-1),
      'hip_pos_penalty_l1':hip,'joint_pos_penalty_l1':legs}
    weights=reward_weights(iteration,stage)
    done=fall_probability(s1) if soft else fell(s1).float()
    return sum(weights[k]*v for k,v in terms.items())*.02,terms,done
