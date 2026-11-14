"""Simulator-independent state algebra for full-pose differentiable rollouts."""
import torch
STATE_DIM, POSE_DIM, POSE_DELTA_DIM, CONTEXT_DIM, NUM_JOINTS = 70, 7, 6, 5, 12
Z, GRAV, V_YAW, W_B = slice(0,1), slice(1,4), slice(4,7), slice(7,10)
Q,QD,HIPS,KNEES,FEET=slice(10,22),slice(22,34),slice(34,46),slice(46,58),slice(58,70)
LEGS=("FL","FR","RL","RR")
JOINT_NAMES=tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip","thigh","calf"))
TERM_DIMS=(3,3,3,12,12,12)
HISTORY,OBS_DIM,SINGLE_DIM=10,450,45
PRIV_DIM=STATE_DIM+3+12+CONTEXT_DIM+77
DEFAULT_Q=(-.1,.9,-1.8,.1,.9,-1.8,-.1,.9,-1.8,.1,.9,-1.8)
AUX_DIM=40
AUX_TORQUE,AUX_ACC,AUX_FEET_VXY,AUX_LOG_FORCE=slice(0,12),slice(12,24),slice(24,32),slice(32,40)
REWARD_STATE=("applied_torque","joint_acceleration","foot_velocity","log_contact_force")

def quat_normalize(q):return q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-9)
def quat_conjugate(q):return torch.cat((-q[...,:3],q[...,3:]),-1)
def quat_mul(a,b):
    return torch.cat((a[...,3:]*b[...,:3]+b[...,3:]*a[...,:3]+torch.cross(a[...,:3],b[...,:3],dim=-1),a[...,3:]*b[...,3:]-(a[...,:3]*b[...,:3]).sum(-1,keepdim=True)),-1)
def quat_apply(q,v):
    xyz,w=q[...,:3],q[...,3:]
    xyz,v=torch.broadcast_tensors(xyz,v)
    return v+2*w*torch.cross(xyz,v,dim=-1)+2*torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1)
def quat_inverse(q,v):return quat_apply(quat_conjugate(q),v)
def rotvec_quat(v):
    angle=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    return torch.cat((v*(.5*torch.sinc(angle/(2*torch.pi))),torch.cos(angle*.5)),-1)
def quat_rotvec(q):
    q=quat_normalize(q);q=torch.where(q[...,3:]<0,-q,q)
    xyz=q[...,:3];length=xyz.norm(dim=-1,keepdim=True)
    angle=2*torch.atan2(length,q[...,3:].clamp_min(0.))
    factor=torch.where(length>1e-6,angle/length.clamp_min(1e-9),2.+length.square()/3.)
    return xyz*factor

def yaw_from_pose(p):
    q=p[...,3:7]
    return torch.atan2(2*(q[...,3]*q[...,2]+q[...,0]*q[...,1]),1-2*(q[...,1].square()+q[...,2].square()))
def yaw_inverse(v,yaw):
    c,s=yaw.cos(),yaw.sin()
    return torch.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1],v[...,2]),-1)
def velocity_body(s,pose=None):
    if pose is not None:
        world=yaw_inverse(s[:,V_YAW],-yaw_from_pose(pose))
        return quat_inverse(pose[:,3:7],world)
    g=s[:,GRAV];roll=torch.atan2(-g[:,1],-g[:,2])
    pitch=torch.atan2(g[:,0],torch.sqrt(g[:,1].square()+g[:,2].square()).clamp_min(1e-6))
    vx,vy,vz=s[:,V_YAW].unbind(-1);cp,sp=pitch.cos(),pitch.sin();cr,sr=roll.cos(),roll.sin()
    xp=cp*vx-sp*vz;zp=sp*vx+cp*vz
    return torch.stack((xp,cr*vy+sr*zp,-sr*vy+cr*zp),-1)
def feet_world_z(s):return s[:,Z]+s[:,FEET].reshape(-1,4,3)[...,2]
def pose_delta(p0,p1):
    return torch.cat((p1[:,:3]-p0[:,:3],quat_rotvec(quat_mul(quat_conjugate(p0[:,3:]),p1[:,3:]))),-1)
def apply_pose_delta(p,d):
    return torch.cat((p[:,:3]+d[:,:3],quat_normalize(quat_mul(p[:,3:],rotvec_quat(d[:,3:])))),-1)

def forward_kinematics(q):
    """AS2 URDF leg chain, joints in FL/FR/RL/RR order; positions in base frame.

    Constants are pinned to the training asset and validated against loaded PhysX
    body transforms by the runtime audit before training.
    """
    q=q.reshape(-1,4,3);hip,thigh,calf=q.unbind(-1)
    sides=q.new_tensor([1.,-1.,1.,-1.]);front=q.new_tensor([1.,1.,-1.,-1.])
    root=torch.stack((front*.2119,sides*.06,torch.zeros_like(front)),-1).expand(len(q),-1,-1)
    off=torch.stack((torch.zeros_like(sides),sides*.1054,torch.zeros_like(sides)),-1).expand(len(q),-1,-1)
    def rx(v):
        x,y,z=v.unbind(-1);c,s=hip.cos(),hip.sin();return torch.stack((x,c*y-s*z,s*y+c*z),-1)
    h=root+rx(off)
    def seg(angle,length):return torch.stack((-length*angle.sin(),torch.zeros_like(angle),-length*angle.cos()),-1)
    knee=h+rx(seg(thigh,.212));foot=knee+rx(seg(thigh+calf,.21344))
    return h,knee,foot

def reconcile_state(s,pose,field=None,default_q=None):
    """Use authoritative pose and FK rather than independent geometry predictions."""
    from .terrain import ground
    q0=s.new_tensor(DEFAULT_Q) if default_q is None else default_q
    body_points=forward_kinematics(s[:,Q]+q0)
    yaw=yaw_from_pose(pose)
    points=[yaw_inverse(quat_apply(pose[:,None,3:],p),yaw[:,None]).flatten(1) for p in body_points]
    gravity=quat_inverse(pose[:,3:],s.new_tensor([0.,0.,-1.]).expand(len(s),3))
    height=(pose[:,2]-ground(pose[:,:2],field))[:,None]
    return torch.cat((height,gravity,s[:,V_YAW.start:QD.stop],*points),-1)
def foot_positions_world(s,pose):
    return pose[:,None,:3]+yaw_inverse(s[:,FEET].reshape(-1,4,3),-yaw_from_pose(pose)[:,None])

def frame(s,a,command,noise=False):
    terms=[s[:,W_B]*.25,s[:,GRAV],command*command.new_tensor([.5,1,1]),s[:,Q],s[:,QD]*.05,a]
    if noise:
        amplitudes=(.05,.05,0.,.01,.075,0.)
        terms=[x+(2*torch.rand_like(x)-1)*amp if amp else x for x,amp in zip(terms,amplitudes)]
    return torch.cat(terms,-1)
def push_history(hist,one):
    blocks=[];i=j=0
    for d in TERM_DIMS:
        blocks.append(torch.cat((hist[:,i+d:i+d*HISTORY],one[:,j:j+d]),-1));i+=d*HISTORY;j+=d
    return torch.cat(blocks,-1)
def single_from_history(hist):
    out=[];i=0
    for d in TERM_DIMS:out.append(hist[:,i+d*(HISTORY-1):i+d*HISTORY]);i+=d*HISTORY
    return torch.cat(out,-1)
def privileged(s,a,command,xi,relief):return torch.cat((s,command,a,xi,s[:,Z]+relief),-1)
