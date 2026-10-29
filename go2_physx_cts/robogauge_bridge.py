"""CTS task adapters for pinned RoboGauge. Simulation and control loop are upstream.

Only robot observation ABI, fixed command protocol and additional raw metrics are
extensions. BasePipeline.run, MujocoSimulator.step/update_torque/reset/load remain
the official implementations. Checkpoint control constants are mandatory.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,math,random,hashlib,importlib.metadata as metadata
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from robogauge.tasks.pipeline import base_pipeline as upstream
from robogauge.tasks.robots.go2.go2 import Go2
from robogauge.tasks.robots.go2.go2_lab_config import Go2Lab131Config
from robogauge.tasks.simulator.mujoco_config import MujocoConfig
from robogauge.tasks.gauge.gauge_configs.flat_gauge_config import FlatGaugeConfig
from robogauge.tasks.gauge.goal_data import GoalData,VelocityGoal
from robogauge.utils.math_utils import get_projected_gravity
from robogauge.utils.logger import logger

UPSTREAM_COMMIT='add2c7ee0185ea8fd93d242f86e1c8cac6e9913b'
LEGS=('FL','FR','RL','RR')

def verify_runtime(exports,iterations):
 manifest=json.loads((Path(__file__).parent/'robogauge_source.json').read_text())
 import robogauge
 root=Path(robogauge.__file__).resolve().parent.parent
 for name,digest in manifest['files'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Official source changed: '+name)
 if mujoco.__version__!='3.2.3' or metadata.version('dm-control')!='1.0.23' or torch.__version__!='2.4.1+cu124':raise RuntimeError('Pinned RoboGauge runtime required')
 errors={}
 for it in iterations:
  meta=json.loads((Path(exports)/f'{it}_metadata.json').read_text())
  for branch in ('teacher','student'):
   path=Path(meta['models'][branch]['path'])
   if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['models'][branch]['sha256']:raise RuntimeError('Export hash mismatch')
   model=torch.jit.load(str(path)).eval();data=np.load(Path(exports)/f'{it}_{branch}_golden.npz');outputs=[]
   with torch.no_grad():
    for i,x in enumerate(data['inputs']):
     if i in data['reset_at']:model.reset()
     outputs.append(model(torch.from_numpy(x).unsqueeze(0))[0].numpy())
   error=float(np.max(np.abs(np.array(outputs)-data['expected'])));assert error<1e-5,(it,branch,error)
   errors[f'{it}_{branch}']=error
 return {'upstream_commit':UPSTREAM_COMMIT,'verified_source_files':len(manifest['files']),'export_errors':errors,'mujoco':mujoco.__version__,'torch':torch.__version__}

def yaw_inverse(v,yaw):
 c,s=math.cos(yaw),math.sin(yaw)
 return np.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1],v[...,2]),-1)

class CTSTeacher(Go2):
 """Simulation-only teacher;167 privileged values append to official45 frame."""
 def build_observation(self,sim_data,goal_data):
  obs=super().build_observation(sim_data,goal_data);sim=self.sim;m,d=sim.mj_model,sim.mj_data
  base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'go2/base_link')
  pos=d.xpos[base];R=d.xmat[base].reshape(3,3);yaw=math.atan2(R[1,0],R[0,0])
  v=sim_data.proprio.base.lin_vel
  # Official base linear velocity is body-frame; state70 stores yaw-frame velocity.
  vy=yaw_inverse(R@v,yaw);wb=sim_data.proprio.imu.ang_vel
  points=[]
  for part in ('thigh','calf','foot'):
   ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'go2/{leg}_{part}') for leg in LEGS]
   if min(ids)<0:raise ValueError(f'Missing anatomical {part} frames')
   points.append(yaw_inverse(d.xpos[ids]-pos,yaw).reshape(-1))
  state=np.concatenate(([pos[2]],get_projected_gravity(sim_data.proprio.imu.quat),vy,wb,sim_data.proprio.joint.pos-self.training_default,sim_data.proprio.joint.vel,*points))
  cmd=obs[6:9]/np.asarray(self.cfg.control.scales.cmd)
  context=np.array([1.,0,0,0,self.payload])
  obs[45:]=np.concatenate((state,cmd,self.last_action,context,np.full(77,pos[2])))
  return obs

def loaded_model(sim,robot):
 m=sim.mj_model
 def rows(kind,n,keys):
  return [{ 'id':i,'name':mujoco.mj_id2name(m,kind,i),**{k:np.asarray(getattr(m,k)[i]).tolist() for k in keys}} for i in range(n)]
 return {'mujoco':mujoco.__version__,'total_mass_kg':float(m.body_mass.sum()),
  'dof_names':sim.dof_names,'damping':m.dof_damping.tolist(),'armature':m.dof_armature.tolist(),'frictionloss':m.dof_frictionloss.tolist(),
  'joints':rows(mujoco.mjtObj.mjOBJ_JOINT,m.njnt,('jnt_range','jnt_limited','jnt_pos','jnt_axis','jnt_qposadr','jnt_dofadr')),
  'bodies':rows(mujoco.mjtObj.mjOBJ_BODY,m.nbody,('body_parentid','body_pos','body_quat','body_mass','body_ipos','body_iquat','body_inertia')),
  'geoms':rows(mujoco.mjtObj.mjOBJ_GEOM,m.ngeom,('geom_bodyid','geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_priority','geom_condim','geom_margin','geom_gap','geom_solref','geom_solimp','geom_contype','geom_conaffinity')),
  'actuators':rows(mujoco.mjtObj.mjOBJ_ACTUATOR,m.nu,('actuator_trnid','actuator_ctrlrange','actuator_ctrllimited','actuator_forcerange','actuator_forcelimited','actuator_gear')),
  'sensors':rows(mujoco.mjtObj.mjOBJ_SENSOR,m.nsensor,('sensor_type','sensor_objtype','sensor_objid','sensor_adr','sensor_dim')),
  'sites':rows(mujoco.mjtObj.mjOBJ_SITE,m.nsite,('site_bodyid','site_pos','site_quat')),
  'options':{k:np.asarray(getattr(m.opt,k)).tolist() for k in ('timestep','solver','integrator','iterations','tolerance','cone','impratio','gravity','disableflags','enableflags')},
  'contact_pair_count':m.npair,'control':{'p_gains':robot.p_gains.tolist(),'d_gains':robot.d_gains.tolist(),'default_dof_pos':robot.default_dof_pos.tolist(),'action_scale':np.asarray(robot.action_scale).tolist(),'mj2model':robot.mj2model_idx},
  'joint_velocity_limit':'No native MuJoCo hard velocity limit; upstream uses static motor ctrlrange, not training torque-speed curve.'}

class CommandGauge:
 """One fixed velocity case. Uses official pipeline lifecycle and reset handling."""
 def __init__(self,cfg,robot_cfg):
  self.cfg=cfg;self.goal_str='fixed_velocity';self.results={};self.done=False;self.start=None
  self.samples=[];self.trace=[];self.first_failure=None;self.resets=0
  self.contact_parameters={}
 def get_goal(self,data):
  if self.start is None:self.start=data.sim_time
  x,y,w=self.cfg.command
  return GoalData('velocity',VelocityGoal(lin_vel_x=x,lin_vel_y=y,ang_vel_yaw=w))
 def is_done(self):return self.done
 def is_reset(self,data):return False
 def reset_metrics(self):pass
 def reset_current_goal(self):self.resets+=1;self.done=True
 def switch_to_next_goal(self):self.done=True;self.first_failure=self.first_failure or 0.
 def update_metrics(self,data,goal):
  sim=self.sim;m,d=sim.mj_model,sim.mj_data
  for contact in d.contact[:d.ncon]:
   names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in contact.geom]
   key=str(names)
   if key not in self.contact_parameters:
    self.contact_parameters[key]={'geoms':names,'dim':int(contact.dim),'friction':contact.friction.tolist(),'solref':contact.solref.tolist(),'solimp':contact.solimp.tolist(),'first_distance':float(contact.dist)}
  gravity=get_projected_gravity(data.proprio.imu.quat)
  tilt=math.acos(float(np.clip(-gravity[2],-1,1)))
  t=0. if self.start is None else data.sim_time-self.start
  if tilt>math.radians(70) or not np.isfinite(d.qpos).all():
   if self.first_failure is None:self.first_failure=t
   self.done=True
  if self.start is None:return False
  if t>=self.cfg.duration:self.done=True
  foot_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'go2/'+leg) for leg in LEGS]
  speeds=[];geom_mask=[];force_mask=[]
  contacts=data.dynamics.contacts
  for gid,leg in zip(foot_ids,LEGS):
   vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,gid,vel,0)
   radius=m.geom_size[gid,0];bottom=np.array([0.,0.,-radius]);v=vel[3:]+np.cross(vel[:3],bottom)
   speeds.append(float(np.linalg.norm(v[:2])))
   geom_mask.append(bool(d.geom_xpos[gid,2]<=radius+.002))
   force_mask.append(any(n.split('/')[-1]==leg and f>1. for n,f in zip(contacts.robot_geom_names,contacts.normal_forces)))
  if t>=1. and data.n_step%max(1,round(.02/data.sim_dt))==0:
   v=data.proprio.base.lin_vel;w=data.proprio.imu.ang_vel;cmd=np.asarray(self.cfg.command)
   self.samples.append({'time':t,'xy_sq':float(np.mean((v[:2]-cmd[:2])**2)),'yaw_sq':float((w[2]-cmd[2])**2),'height':float(d.qpos[2]),'vx':float(v[0]),'vy':float(v[1]),'wz':float(w[2]),'roll_pitch_tilt':tilt,'foot_speed':speeds,'geometric_contact':geom_mask,'force_contact':force_mask})
  if data.n_step%max(1,round(.02/data.sim_dt))==0:self.trace.append({'time':t,'qpos':d.qpos.tolist(),'foot_speed':speeds,'geometric_contact':geom_mask,'force_contact':force_mask})
  return False
 def summary(self):
  a=self.samples;result={'survived':self.first_failure is None and self.resets==0 and bool(a),'survival_seconds':self.cfg.duration if self.first_failure is None else self.first_failure,'resets':self.resets,'samples':len(a)}
  for key in ('height','vx','vy','wz','roll_pitch_tilt'):result[key]=float(np.mean([x[key] for x in a])) if a else None
  result['xy_rmse']=float(np.sqrt(np.mean([x['xy_sq'] for x in a]))) if a else None
  result['yaw_rmse']=float(np.sqrt(np.mean([x['yaw_sq'] for x in a]))) if a else None
  for label,mask in (('geometric','geometric_contact'),('force','force_contact')):
   values=[v for x in a for v,c in zip(x['foot_speed'],x[mask]) if c]
   result[f'slip_{label}_mean_mps']=float(np.mean(values)) if values else None
   result[f'slip_{label}_p95_mps']=float(np.quantile(values,.95)) if values else None
   result[f'contact_{label}_samples']=len(values)
  tolerance=max(.2,.2*float(np.linalg.norm(self.cfg.command[:2])))
  result['pass']=bool(result['survived'] and result['xy_rmse']<tolerance and result['yaw_rmse']<.3 and result['height']>.22)
  result['tracking_xy_tolerance']=tolerance
  return result

class AttachedPipeline(upstream.BasePipeline):
 def load(self):
  super().load();self.gauge.sim=self.sim
  if isinstance(self.robot,CTSTeacher):
   self.robot.sim=self.sim;self.robot.training_default=np.asarray(self.robot_cfg.training_default);self.robot.payload=self.sim_cfg.domain_rand.base_mass
  self.loaded=loaded_model(self.sim,self.robot)
  self.robot.reset()

def evaluate_case(exports,iteration,branch,command,payload,output,seed=17,video=False,official_noise_delay=False):
 if mujoco.__version__!='3.2.3':raise RuntimeError('Use the pinned official MuJoCo3.2.3 runtime; no version bypass.')
 torch.set_num_threads(1);torch.set_grad_enabled(False);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 meta=json.loads((Path(exports)/f'{iteration}_metadata.json').read_text());const=meta['consts']
 robot=Go2Lab131Config();sim=MujocoConfig();gauge=FlatGaugeConfig()
 robot.control.model_path=meta['models'][branch]['path'];robot.robot_class='CTSTeacher' if branch=='teacher' else 'Go2'
 robot.control.num_observations=212 if branch=='teacher' else 45
 robot.control.mj2model_dof_indices=list(range(12));robot.training_default=const['default_q']
 for key in ('kp','kd','default_q','action_scale'):
  value=np.asarray(const[key])
  if value.shape!=(12,) or not np.isfinite(value).all():raise ValueError('Invalid checkpoint control constant: '+key)
 robot.control.p_gains=const['kp'];robot.control.d_gains=const['kd'];robot.control.default_dof_pos=const['default_q'];robot.control.scales.action=np.asarray(const['action_scale'])
 robot.commands.lin_vel_x=[-max(2.,abs(command[0])),max(2.,abs(command[0]))]
 robot.commands.lin_vel_y=[-max(.3,abs(command[1])),max(.3,abs(command[1]))]
 robot.commands.ang_vel_yaw=[-max(.8,abs(command[2])),max(.8,abs(command[2]))]
 sim.viewer.headless=True;sim.render.save_video=video;sim.domain_rand.base_mass=payload
 sim.domain_rand.action_delay=official_noise_delay;sim.noise.enabled=official_noise_delay
 gauge.gauge_class='CommandGauge';gauge.command=list(command);gauge.duration=10.
 upstream.CTSTeacher=CTSTeacher;upstream.CommandGauge=CommandGauge
 tag=f'{iteration}_{branch}_checkpoint_control_{command[0]}_{command[1]}_{command[2]}_load{payload}_s{seed}'
 logger.create('official_cases',tag,console_output=False,parent_log_dir=output)
 pipeline=AttachedPipeline(tag,sim,robot,gauge,SimpleNamespace(seed=seed))
 _,warning,error=pipeline.run();result=pipeline.gauge.summary()
 result.update(iteration=iteration,branch=branch,command=list(command),payload_kg=payload,control_contract='checkpoint',seed=seed,checkpoint_sha256=meta['checkpoint_sha256'],upstream_commit=UPSTREAM_COMMIT,warning=str(warning) if warning else None,error=str(error) if error else None,training_updates=0,protocol={'physics_dt':sim.physics.simulation_dt,'control_dt':.02,'metric_dt':.02,'duration':10,'ignore_seconds':1,'initial_settle':'official BasePipeline first_reset','noise_enabled':official_noise_delay,'random_action_delay':official_noise_delay},case_dir=str(logger.log_dir))
 pipeline.loaded['observed_contact_parameters']=list(pipeline.gauge.contact_parameters.values())
 (logger.log_dir/'loaded_parameters.json').write_text(json.dumps(pipeline.loaded,indent=2))
 (logger.log_dir/'metrics.json').write_text(json.dumps(result,indent=2))
 (logger.log_dir/'trajectory.json').write_text(json.dumps(pipeline.gauge.trace))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--exports',required=True);p.add_argument('--output',required=True);p.add_argument('--iterations',default='5000,10000,40000');p.add_argument('--forward-only',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--official-noise-delay',action='store_true')
 p.add_argument('--payloads',default='0,1,2');p.add_argument('--forward-speeds',default='');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 (out/'runtime_verification.json').write_text(json.dumps(verify_runtime(a.exports,list(map(int,a.iterations.split(',')))),indent=2))
 commands=[(.5,0,0)] if a.forward_only else [(0,0,0),(.5,0,0),(-.3,0,0),(0,.3,0),(0,0,.8)]
 for speed in filter(None,a.forward_speeds.split(',')):
  cmd=(float(speed),0,0)
  if cmd not in commands:commands.append(cmd)
 rows=[]
 for it in map(int,a.iterations.split(',')):
  for branch in ('teacher','student'):
   for cmd in commands:
    for payload in ((0.,) if a.forward_only else map(float,a.payloads.split(','))):
     row=evaluate_case(a.exports,it,branch,cmd,payload,out,video=a.video,official_noise_delay=a.official_noise_delay);rows.append(row)
     (out/'results.json').write_text(json.dumps({'rows':rows,'training_updates':0,'upstream_commit':UPSTREAM_COMMIT},indent=2));print(json.dumps(row),flush=True)
if __name__=='__main__':main()
