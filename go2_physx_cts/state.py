"""Source70-D state and45x10 blind CTS interface; no simulator imports."""
import torch
STATE_DIM, POSE_DELTA_DIM, CONTEXT_DIM, NUM_JOINTS = 70, 3, 5, 12
Z, GRAV, V_YAW, W_B = slice(0,1), slice(1,4), slice(4,7), slice(7,10)
Q, QD, HIPS, KNEES, FEET = slice(10,22),slice(22,34),slice(34,46),slice(46,58),slice(58,70)
LEGS = ("FL", "FR", "RL", "RR")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip","thigh","calf"))
TERM_DIMS = (3,3,3,12,12,12)
HISTORY, OBS_DIM, SINGLE_DIM = 10, 450, 45
PRIV_DIM = STATE_DIM + 3 + 12 + CONTEXT_DIM + 77

def feet_world_z(s):
    return s[:,Z]+s[:,FEET].reshape(-1,4,3)[...,2]

def yaw_inverse(v, yaw):
    c,s=yaw.cos(),yaw.sin()
    return torch.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1],v[...,2]),-1)

def velocity_body(s):
    """Undo pitch and roll after yaw removal; command axes remain body axes."""
    g=s[:,GRAV];roll=torch.atan2(-g[:,1],-g[:,2])
    pitch=torch.atan2(g[:,0],torch.sqrt(g[:,1].square()+g[:,2].square()).clamp_min(1e-6))
    vx,vy,vz=s[:,V_YAW].unbind(-1);cp,sp=pitch.cos(),pitch.sin();cr,sr=roll.cos(),roll.sin()
    xp=cp*vx-sp*vz;zp=sp*vx+cp*vz
    return torch.stack((xp,cr*vy+sr*zp,-sr*vy+cr*zp),-1)

def pose_delta(p0,p1):
    d=p1-p0;c,s=p0[:,2].cos(),p0[:,2].sin()
    return torch.stack((c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1],torch.atan2(d[:,2].sin(),d[:,2].cos())),-1)

def apply_pose_delta(p,d):
    c,s=p[:,2].cos(),p[:,2].sin()
    return torch.stack((p[:,0]+c*d[:,0]-s*d[:,1],p[:,1]+s*d[:,0]+c*d[:,1],p[:,2]+d[:,2]),-1)

def frame(s,a,command,noise=False):
    # Preserve deployed CTS term order and units/scales; running norms follow.
    terms=[s[:,W_B]*.25,s[:,GRAV],command*command.new_tensor([.5,1,1]),s[:,Q],s[:,QD]*.05,a]
    if noise:
        amplitudes=(.05,.05,0.,.01,.075,0.)
        terms=[x+(2*torch.rand_like(x)-1)*amp if amp else x for x,amp in zip(terms,amplitudes)]
    return torch.cat(terms,-1)

def push_history(hist,one):
    blocks=[];i=j=0
    for d in TERM_DIMS:
        blocks.append(torch.cat((hist[:,i+d:i+d*HISTORY],one[:,j:j+d]),-1))
        i+=d*HISTORY;j+=d
    return torch.cat(blocks,-1)

def single_from_history(hist):
    out=[];i=0
    for d in TERM_DIMS:
        out.append(hist[:,i+d*(HISTORY-1):i+d*HISTORY]);i+=d*HISTORY
    return torch.cat(out,-1)

def privileged(s,a,command,xi,relief):
    return torch.cat((s,command,a,xi,s[:,Z]+relief),-1)
