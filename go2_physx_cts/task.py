"""Shared real/imagination velocity rewards, independent of any simulator."""
import math
from dataclasses import dataclass
import torch
from . import state as st

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

def fall_probability(s):
    tilt=torch.acos((-s[:,3]).clamp(-1+1e-4,1-1e-4))
    return torch.sigmoid((tilt-math.radians(70))/math.radians(5))

def fell(s):return -s[:,3]<math.cos(math.radians(70))

def reward(s0,s1,a,previous,command,c,stage='flat',soft=False):
    q=s1[:,st.Q]+c.default_q;lim=c.soft_limits
    tau=(c.kp*(a*c.action_scale+c.default_q-q)-c.kd*s1[:,st.QD]).clamp(-c.effort,c.effort)
    g=s1[:,st.GRAV]
    d=fall_probability(s1) if soft else fell(s1).float()
    terms={'track_xy':torch.exp(-(st.velocity_body(s1)[:,:2]-command[:,:2]).square().sum(-1)/.25),
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
           'joint_acc_l2':((s1[:,st.QD]-s0[:,st.QD])/.02).square().sum(-1)}
    weights=dict(WEIGHTS,terminated=-600. if stage=='terrain' else -200.)
    return sum(weights[k]*v for k,v in terms.items())*.02,terms,d

def advance_commands(command,timer,maxima,dt=.02):
    timer=timer-dt;due=timer[:,0]<=0
    new=(2*torch.rand_like(command)-1)*command.new_tensor(maxima)
    new[torch.rand(command.shape[0],device=command.device)<.1]=0
    command=torch.where(due[:,None],new,command)
    timer=torch.where(due[:,None],2+3*torch.rand_like(timer),timer)
    return command,timer
