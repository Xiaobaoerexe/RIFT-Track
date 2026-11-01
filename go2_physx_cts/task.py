"""Ten agreed rewards. Continuous rates use dt; terminal/success events do not.

Real hard predicates and imagined soft expectations are deliberately distinct.
Budget numbers come only from complete official-policy PhysX episodes.
"""
from dataclasses import dataclass
import json, math
from pathlib import Path
import torch
from . import state as st
from .terrain import ground

DT=.02
TASK_CONTEXT_DIM=21
GOAL=slice(0,3)
COMPLETED,STABLE,ELAPSED=3,4,5
ROUTE=slice(6,8)
ROUTE_YAW,PROGRESS=8,9
ORIGIN=slice(10,12)
EPISODE_LIMIT,RECOVERY_DEADLINE,TARGET_RADIUS,ROUTE_LENGTH=12,13,14,15
PRIOR_SUPPORT=slice(16,20)
HIGH_WATERMARK=20
FLAT_TERRAIN_ID=7
OFFICIAL_REWARD_SOURCE='wty-yy/go2_rl_lab@6c95d9feef4814c25eb6536e61bb38e1d1d718d4'
BASE_HEIGHT_TARGET=.38
BASE_HEIGHT_CROP_X=(-.2,-.1,0.,.1,.2)
BASE_HEIGHT_CROP_Y=(-.1,0.,.1)
_BUDGET=None

@dataclass
class RobotConsts:
    default_q:torch.Tensor
    action_scale:torch.Tensor
    kp:torch.Tensor
    kd:torch.Tensor
    effort:torch.Tensor
    soft_limits:torch.Tensor

def configure_reward_budget(path_or_record):
    global _BUDGET
    value=json.loads(Path(path_or_record).read_text()) if isinstance(path_or_record,(str,Path)) else dict(path_or_record)
    if value.get('schema')!='depth_reward_budget_v1' or value['reference_net_max']<=0:
        raise ValueError('Measured complete-episode official-policy calibration required')
    if len(value['tasks'])!=10:raise ValueError('Exactly ten task budgets required')
    for row in value['tasks'][1:]:
        bound=row['base_positive_rate_bound']*row['episode_seconds']
        if bound+row['success_bonus']+row['progress_budget']>value['positive_cap']+1e-5:
            raise ValueError('Gross positive budget exceeds measured cap')
    _BUDGET=value
    return value

def reward_budget():return _BUDGET

def pose_yaw(pose):
    x,y,z,w=pose[:,3:7].unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def initial_task_context(pose,goal_xyz=None,episode_limit=20.,recovery_deadline=8.,target_radius=.6,route_length=5.):
    ctx=pose.new_zeros((pose.shape[0],TASK_CONTEXT_DIM))
    ctx[:,GOAL]=pose[:,:3] if goal_xyz is None else goal_xyz
    ctx[:,ORIGIN]=pose[:,:2];ctx[:,ROUTE]=pose[:,:2];ctx[:,ROUTE_YAW]=pose_yaw(pose)
    for k,v in ((EPISODE_LIMIT,episode_limit),(RECOVERY_DEADLINE,recovery_deadline),(TARGET_RADIUS,target_radius),(ROUTE_LENGTH,route_length)):
        ctx[:,k]=torch.as_tensor(v,device=pose.device,dtype=pose.dtype)
    return ctx

def reward_weights(iteration,stage=None):
    p=torch.as_tensor(iteration).sub(1500).div(1500).clamp(0,1)
    return {'lin_vel_z_l2':-2+2*p,'base_height_l2':-1-9*p}

def crop_height_from_map(s,pose,field=None):
    xs,ys=torch.meshgrid(pose.new_tensor(BASE_HEIGHT_CROP_X),pose.new_tensor(BASE_HEIGHT_CROP_Y),indexing='ij')
    x,y=xs.flatten(),ys.flatten();yaw=pose_yaw(pose)[:,None]
    xy=torch.stack((pose[:,0:1]+yaw.cos()*x-yaw.sin()*y,pose[:,1:2]+yaw.sin()*x+yaw.cos()*y),-1)
    return pose[:,2]-ground(xy,field).mean(-1)

def fall_probability(s):return torch.sigmoid((math.cos(math.radians(70))+s[:,3])/.035)
def fell(s):return -s[:,3]<math.cos(math.radians(70))
def _clip(x,v):return torch.where(torch.isfinite(x),x.clamp_max(v),x)
def _table(s,ids,values):return s.new_tensor(values)[ids]
def _contacts(telemetry,soft):
    if 'contact_prob' in telemetry:return telemetry['contact_prob'].clamp(0,1)
    f=telemetry['contact_force']
    return torch.sigmoid((f-5.)/1.) if soft else (f>5.).to(f.dtype)

def task_reward(s0,s1,a,previous,command,c,*,task_id,terrain_id,context,pose0,pose,telemetry,
                previous2=None,history_valid=None,iteration=0,field=None,soft=False,height_above_crop=None):
    """Return reward, terms, next_context, failure probability/bool, new success.

    telemetry: contact_prob14 (feet4,illegal,head,limb8), applied_torque12,
    foot_velocity4x3; real transitions also provide foot_positions/joint_acc.
    None of the imagined input derivatives are detached.
    """
    ids=task_id.long();ctx=context.clone();b=s1.shape[0];zero=s1.new_zeros(b)
    contacts=_contacts(telemetry,soft)
    support,illegal,head,limb=contacts[:,:4],contacts[:,4],contacts[:,5],contacts[:,6:14]
    tau=telemetry['applied_torque'];foot_vel=telemetry['foot_velocity'].reshape(b,4,3)
    crop_h=crop_height_from_map(s1,pose,field) if height_above_crop is None else height_above_crop
    flat=(terrain_id==FLAT_TERRAIN_ID).to(s1.dtype);velocity=st.velocity_body(s1)
    qoffset,qd=s1[:,st.Q],s1[:,st.QD];q=qoffset+c.default_q;lim=c.soft_limits
    e2=(velocity[:,:2]-command[:,:2]).square().sum(-1)
    tv=torch.exp(-e2/.25);tw=torch.exp(-(s1[:,9]-command[:,2]).square()/.25)
    width=.5+.125*command[:,:2].norm(dim=-1)
    fast=1.5/(1+e2/width.square())+.5*tv
    done0=context[:,COMPLETED].clamp(0,1);recovery=ids==6
    vweight=_table(s1,ids,[2,2,2,1.5,1.5,2,2,1,1,1])
    wweight=_table(s1,ids,[1,1,1,.75,.75,1,1,.5,.5,.5])
    gate=torch.where(recovery,done0,torch.ones_like(done0))
    positive={'track_xy':torch.where(ids==1,fast,vweight*tv)*gate,'track_yaw':wweight*tw*gate}
    upright=((1-s1[:,3])/2).clamp(0,1)
    positive['recovery_lift']=torch.where(recovery,.2*upright.square()*(crop_h/.38).clamp(0,1),zero)
    valid1=(context[:,ELAPSED]>=DT).to(s1.dtype);valid2=(context[:,ELAPSED]>=2*DT).to(s1.dtype)
    if history_valid is not None:valid1,valid2=history_valid[:,0],history_valid[:,1]
    if previous2 is None:raise ValueError('Actual second previous action required')
    a1=_clip((a-previous).square().sum(-1),20.)*valid1
    a2=_clip(((a-2*previous+previous2).square()*(previous!=0)*(previous2!=0)).sum(-1),50.)*valid2
    acc=telemetry.get('joint_acc',(qd-s0[:,st.QD])/DT)
    qacc=_clip(acc.square().sum(-1),1e8)*valid1
    qlim=((lim[...,0]-q).clamp_min(0)+(q-lim[...,1]).clamp_min(0)).sum(-1)
    costs={
        'joint_limit':-2*qlim,
        'joint_acc':-_table(s1,ids,[2.5e-7,5e-8,2.5e-7,2.5e-7,2.5e-7,2.5e-7,2.5e-7,2.5e-7,1e-7,1e-7])*qacc,
        'power':-_table(s1,ids,[2e-5,5e-6,2e-5,2e-5,2e-5,1e-5,1e-5,1e-5,1e-5,1e-5])*(tau*qd).abs().sum(-1),
        'torque':-_table(s1,ids,[1e-4,2e-5,1e-4,1e-4,1e-4,2e-5,2e-5,2e-5,2e-5,2e-5])*tau.square().sum(-1),
        'action_rate':-_table(s1,ids,[.01,.01,.01,.01,.01,.01,.005,.01,.01,.01])*a1,
        'action_smooth':-_table(s1,ids,[.01,.005,.01,.01,.01,.005,.002,.005,.005,.005])*a2,
        'tilt_rate':-_table(s1,ids,[.05,.02,.05,.05,.05,.02,0,.05,.02,.05])*s1[:,7:9].square().sum(-1),
        'hip':-.05*(1+9*flat*(ids!=1))*qoffset[:,0::3].abs().sum(-1)}
    if 'foot_positions' in telemetry:
        foot_h=(crop_h[:,None]+telemetry['foot_positions'][...,2]-pose[:,2:3]).clamp_min(0)
    else:foot_h=(crop_h[:,None]+s1[:,st.FEET].reshape(b,4,3)[...,2]).clamp_min(0)
    freg=(foot_vel[...,:2].square().sum(-1)*torch.exp(-foot_h/(.025*.38))).sum(-1)
    normal=telemetry.get('ground_normal')
    tangent=foot_vel if normal is None else foot_vel-(foot_vel*normal).sum(-1,keepdim=True)*normal
    slip=(support*(tangent[...,:2].square().sum(-1) if normal is None else tangent.square().sum(-1))).sum(-1)
    costs['feet']=torch.where(ids==0,-.05*freg,-_table(s1,ids,[.05,.05,.10,.20,.10,.05,0,.05,.05,.10])*slip)
    costs['undesired_contact']=torch.where(recovery|(ids>=8),zero,-limb.sum(-1))
    p=torch.as_tensor(iteration,device=s1.device,dtype=s1.dtype).sub(1500).div(1500).clamp(0,1)
    costs['height']=flat*torch.where(ids==1,-10*(.36-crop_h).clamp_min(0).square(),(-1-9*p)*(crop_h-.38).square())
    costs['vertical_speed_course']=torch.where(ids==0,(-2+2*p)*velocity[:,2].square(),zero)
    leg=qoffset.reshape(b,4,3)[...,1:].abs().sum((1,2))
    costs['leg_deviation']=torch.where(ids==0,-.01*(1+9*flat*(command.norm(dim=-1)<1e-3))*leg,zero)
    az=(s1[:,6]-s0[:,6])/DT
    costs['vertical_acceleration']=torch.where((ids==3)|(ids==4),-.0015*az.square()*valid1,zero)
    landing=(support-context[:,PRIOR_SUPPORT]).clamp_min(0).amax(-1)
    costs['landing']=torch.where((ids==4)|(ids==5),-.5*landing*(-s1[:,6]-1).clamp_min(0).square(),zero)
    costs['fast_direction']=torch.where(ids==1,-.05*(velocity[:,1]-command[:,1]).square()-.02*(s1[:,9]-command[:,2]).square(),zero)
    costs['recovery_posture']=torch.where(recovery,-(1-upright)-.3*(1-(support.sum(-1)/3).clamp_max(1)),zero)
    # Integrate commanded route; rotating a direction cannot earn reward from
    # old displacement. Returning over already credited distance earns zero.
    yaw=context[:,ROUTE_YAW]
    direction=torch.stack((yaw.cos()*command[:,0]-yaw.sin()*command[:,1],yaw.sin()*command[:,0]+yaw.cos()*command[:,1]),-1)
    delta=pose[:,:2]-pose0[:,:2]
    net=context[:,PROGRESS]+(delta*direction).sum(-1)/direction.norm(dim=-1).clamp_min(1e-6)
    watermark=torch.maximum(context[:,HIGH_WATERMARK],net)
    ctx[:,ROUTE]=context[:,ROUTE]+direction*DT;ctx[:,ROUTE_YAW]=yaw+command[:,2]*DT
    ctx[:,PROGRESS]=net;ctx[:,HIGH_WATERMARK]=watermark
    ctx[:,ELAPSED]=context[:,ELAPSED]+DT;ctx[:,PRIOR_SUPPORT]=support
    goal_delta=context[:,GOAL][:,:2]-context[:,ORIGIN]
    goal_dir=goal_delta/goal_delta.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    target_delta=pose[:,:2]-context[:,GOAL][:,:2]
    target_along=(target_delta*goal_dir).sum(-1)
    target_lateral=(target_delta[:,0]*goal_dir[:,1]-target_delta[:,1]*goal_dir[:,0]).abs()
    vertical_error=(pose[:,2]-context[:,GOAL][:,2]-.38).abs()
    if soft:
        upright_ok=torch.sigmoid((-s1[:,3]-math.cos(math.radians(35)))/.05)
        support_ok=torch.sigmoid((support.sum(-1)-1.5)/.25)
        height_ok=torch.sigmoid((crop_h-.24)/.035)
        rate_ok=torch.sigmoid((3.-s1[:,7:10].norm(dim=-1))/.3)
        region=torch.sigmoid((target_along+context[:,TARGET_RADIUS])/.12)*torch.sigmoid((context[:,TARGET_RADIUS]-target_lateral)/.12)*torch.sigmoid((.28-vertical_error)/.05)
        eligible=upright_ok*support_ok*height_ok*rate_ok*torch.where(recovery,torch.ones_like(region),region)
        # A soft occupancy clock avoids pretending Boolean persistence has an
        # exact gradient. Leakage is bounded and measured in validation.
        stable=(context[:,STABLE]+DT*eligible)*torch.sigmoid((eligible-.4)/.03)
        required=torch.where(ids==5,s1.new_full((b,),.64),s1.new_full((b,),.32))
        success_p=torch.sigmoid((stable-required)/.02)*eligible
    else:
        eligible=((-s1[:,3]>=math.cos(math.radians(35)))&(support.sum(-1)>=2)&(crop_h>=.24)&(s1[:,7:10].norm(dim=-1)<3.)
                  &(recovery|((target_along>=-context[:,TARGET_RADIUS])&(target_lateral<=context[:,TARGET_RADIUS])&(vertical_error<.28))))
        stable=torch.where(eligible,context[:,STABLE]+DT,zero)
        required=torch.where(ids==5,s1.new_full((b,),.64),s1.new_full((b,),.32))
        success_p=(stable>=required).to(s1.dtype)
    success_p*=((ids==3)|(ids==4)|(ids==5)|(ids==6)|(ids==8)|(ids==9))
    fall=fall_probability(s1) if soft else fell(s1).to(s1.dtype)
    normal_failure=1-(1-fall)*(1-illegal)*(1-head)
    deadline=(ctx[:,ELAPSED]>=context[:,RECOVERY_DEADLINE]).to(s1.dtype)*(1-done0)
    failure=torch.where(recovery,done0*normal_failure+(1-done0)*deadline,normal_failure).clamp(0,1)
    success_p=success_p*(1-failure)
    complete=done0+(1-done0)*success_p;success=complete-done0
    ctx[:,COMPLETED]=complete;ctx[:,STABLE]=stable
    costs['unfinished_time']=-_table(s1,ids,[0,0,0,0,0,.1,.2,0,.1,.1])*(1-complete)
    if _BUDGET is None:success_budget,route_budget,failure_amount=zero,zero,0.
    else:
        success_budget=_table(s1,ids,[r['success_bonus'] for r in _BUDGET['tasks']])
        route_budget=_table(s1,ids,[r['progress_budget'] for r in _BUDGET['tasks']])
        failure_amount=_BUDGET['failure_penalty']
    event_success=success_budget*success
    old=(context[:,HIGH_WATERMARK]/context[:,ROUTE_LENGTH].clamp_min(.1)).clamp(0,1)
    new=(watermark/context[:,ROUTE_LENGTH].clamp_min(.1)).clamp(0,1)
    event_progress=route_budget*(new-old).clamp_min(0)
    # A head hit receives one -18 event in avoidance, never stacked generic
    # failure plus head penalties. Soft branch smoothly mixes the event cost.
    collision=1-(1-head)*(1-illegal)
    penalty=torch.where(ids>=7,failure_amount+(18.-failure_amount)*collision,s1.new_full((b,),failure_amount))
    event_failure=-failure*penalty
    positive_sum=sum(positive.values())*DT+event_success+event_progress
    negative_sum=sum(costs.values())*DT+event_failure
    terms={**positive,**costs,'event_success':event_success,'event_progress':event_progress,'event_failure':event_failure,
           'positive':positive_sum,'negative':negative_sum,'survival':1-failure,'success_probability':success,'completed':complete}
    return positive_sum+negative_sum,terms,ctx,(failure if soft else failure.bool()),success
