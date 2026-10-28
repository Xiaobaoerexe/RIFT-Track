"""Independent MuJoCo velocity checks for both branches; no training updates."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import mujoco
from . import state as st
from .networks import Policy
from .physx_env import quat_inverse,motor_values

@torch.no_grad()
def evaluate(checkpoint,xml,seconds=10.,commands=None):
    torch.set_num_threads(2)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=ck['cfg']
    p=Policy(tuple(cfg['actor_hidden']));p.load_state_dict(ck['policy']);p.eval()
    commands=commands or [(0,0,0),(.5,0,0),(-.3,0,0),(0,.3,0),(0,0,.8)]
    model=mujoco.MjModel.from_xml_path(str(xml));model.opt.timestep=.005
    joint=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in st.JOINT_NAMES]
    if min(joint)<0:raise ValueError('MuJoCo joint names differ')
    qa=model.jnt_qposadr[joint];va=model.jnt_dofadr[joint]
    base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base')
    if base<0:base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'trunk')
    if base<0:raise ValueError('No base body')
    ids={part:[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_{part}') for leg in st.LEGS] for part in ('thigh','calf','foot')}
    inertia,kp,kd,effort,scale=motor_values('cpu');q0=ck['consts']['default_q'];speed=torch.tensor([30.1,30.1,20.07]*4)
    model.dof_armature[va]=inertia.numpy();model.dof_damping[va]=0;model.dof_frictionloss[va]=0
    rows=[];nomass=float(model.body_mass[base]);rng=np.random.default_rng(17)
    for branch in (0,1):
      for cmd in commands:
       for payload in (0.,1.):
        model.body_mass[base]=nomass+payload
        d=mujoco.MjData(model);d.qpos[:7]=[0,0,.29,1,0,0,0];d.qpos[qa]=q0.numpy();mujoco.mj_forward(model,d)
        hist=torch.zeros(1,450);last=torch.zeros(1,12);command=torch.tensor([cmd],dtype=torch.float32)
        xi=torch.tensor([[1,0,0,0,payload]],dtype=torch.float32);ring=q0.repeat(3,1);rp=0;delay=1
        errors=[];heights=[];fell=False
        for i in range(round(seconds/.02)):
            q=torch.tensor(d.xquat[base][[1,2,3,0]],dtype=torch.float32)[None];position=torch.tensor(d.xpos[base],dtype=torch.float32)[None]
            mat=d.xmat[base].reshape(3,3);yaw=float(math.atan2(mat[1,0],mat[0,0]))
            vel=np.zeros(6);mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_BODY,base,vel,0)
            vb=quat_inverse(q,torch.tensor(vel[3:],dtype=torch.float32)[None]);wb=quat_inverse(q,torch.tensor(vel[:3],dtype=torch.float32)[None])
            g=quat_inverse(q,torch.tensor([[0.,0.,-1.]]));vy=st.yaw_inverse(torch.tensor(vel[3:],dtype=torch.float32)[None],torch.tensor([yaw]))
            pts=[]
            for part in ('thigh','calf','foot'):
                if min(ids[part])>=0:points=d.xpos[ids[part]]
                else:points=np.stack([d.xpos[j]+d.xmat[j].reshape(3,3)@np.array([0,0,-.213]) for j in ids['calf']])
                pts.append(st.yaw_inverse(torch.tensor(points,dtype=torch.float32)[None]-position[:,None,:],torch.tensor([[yaw]])).flatten(1))
            s=torch.cat((position[:,2:3],g,vy,wb,torch.tensor(d.qpos[qa],dtype=torch.float32)[None]-q0,torch.tensor(d.qvel[va],dtype=torch.float32)[None],*pts),-1)
            hist=st.push_history(hist,st.frame(s,last,command));priv=st.privileged(s,last,command,xi,torch.zeros(1,77))
            a=p(hist,priv,branch);target=q0+a[0]*scale
            for _ in range(4):
                ring[rp]=target;delayed=ring[(rp-delay)%3];rp=(rp+1)%3
                qv=torch.tensor(d.qvel[va],dtype=torch.float32);qp=torch.tensor(d.qpos[qa],dtype=torch.float32)
                tau=kp*(delayed-qp)-kd*qv
                upper=(2*effort*(1-qv/speed)).clamp(0);upper=torch.minimum(upper,effort)
                lower=(2*effort*(-1-qv/speed)).clamp(max=0);lower=torch.maximum(lower,-effort)
                tau=torch.maximum(torch.minimum(tau,upper),lower)
                d.ctrl[:]=0;d.qfrc_applied[:]=0;d.qfrc_applied[va]=tau.numpy();mujoco.mj_step(model,d)
            last=a
            if i>=50:errors.append([float((vb[0,:2]-command[0,:2]).square().mean()),float((wb[0,2]-command[0,2]).square())]);heights.append(float(position[0,2]))
            if float(-g[0,2])<math.cos(math.radians(70)) or not np.isfinite(d.qpos).all():fell=True;break
        er=np.asarray(errors) if errors else np.array([[100.,100.]])
        rows.append({'branch':'teacher' if branch==0 else 'student','command':cmd,'payload_kg':payload,'survived':not fell,'seconds':(i+1)*.02,'xy_rmse':float(np.sqrt(er[:,0].mean())),'yaw_rmse':float(np.sqrt(er[:,1].mean())),'height':float(np.mean(heights)) if heights else 0.})
    summary={}
    for branch in ('teacher','student'):
        part=[r for r in rows if r['branch']==branch]
        summary[branch]={'survival_fraction':float(np.mean([r['survived'] for r in part])),'xy_rmse':float(np.mean([r['xy_rmse'] for r in part])),'yaw_rmse':float(np.mean([r['yaw_rmse'] for r in part])),'height':float(np.mean([r['height'] for r in part]))}
    student=summary['student'];student_rows=[r for r in rows if r['branch']=='student']
    passed=all(r['survived'] and r['xy_rmse']<.2 and r['yaw_rmse']<.3 and r['height']>.22 for r in student_rows)
    return {'backend':'mujoco_native','checkpoint':str(checkpoint),'iteration':ck['iteration'],'stage':cfg['stage'],'summary':summary,'rows':rows,'flat_gate_pass':passed,'B0_claim':False,'training_updates':0}

def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--xml',required=True);a.add_argument('--output',required=True);a.add_argument('--seconds',type=float,default=10.)
    args=a.parse_args();result=evaluate(args.checkpoint,args.xml,args.seconds)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result['summary']),flush=True)
if __name__=='__main__':main()
