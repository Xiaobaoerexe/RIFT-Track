"""CTS task adapters for pinned RoboGauge. Simulation and control loop are upstream.

Only robot observation ABI, fixed command protocol and additional raw metrics are
extensions. BasePipeline.run, MujocoSimulator.step/update_torque/reset/load remain
the official implementations. Checkpoint control constants are mandatory.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,math,random,hashlib,sys,importlib.metadata as metadata
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
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

def verify_source_runtime():
 manifest=json.loads((Path(__file__).parent/'robogauge_source.json').read_text())
 import robogauge
 root=Path(robogauge.__file__).resolve().parent.parent
 for name,digest in manifest['files'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Official source changed: '+name)
 if mujoco.__version__!='3.2.3' or metadata.version('dm-control')!='1.0.23' or torch.__version__!='2.4.1+cu124':raise RuntimeError('Pinned RoboGauge runtime required')
 return {'upstream_commit':UPSTREAM_COMMIT,'verified_source_files':len(manifest['files']),'mujoco':mujoco.__version__,'torch':torch.__version__}

def verify_runtime(exports,iterations):
 verified=verify_source_runtime()
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
 return dict(verified,export_errors=errors)

def yaw_inverse(v,yaw):
 c,s=math.cos(yaw),math.sin(yaw)
 return np.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1],v[...,2]),-1)

class CTSPerceptive(Go2):
 """Observation-only extension; upstream PD, integration and resets are retained."""
 def setup_perception(self):
  root=Path(self.cfg.depth_package);sys.path.insert(0,str(root))
  from go2_depth_input.vendor.camera import rays,MOUNT,optical_rotation
  from go2_depth_input.vendor.noise_reference import DepthNoise
  self.rays=rays().numpy().reshape(-1,3).astype(np.float64)
  self.mount=np.asarray(MOUNT);self.optical=np.asarray([[0,0,1],[-1,0,0],[0,-1,0.]])
  self.depth_noise=DepthNoise().eval()
  state=torch.load(root/'go2_depth_input/assets/weights.pt',map_location='cpu',weights_only=True)
  self.depth_noise.load_state_dict(state['noise']);self.depth_noise.freeze()
  self.height_cache={};self.last_depth_frame=-1;self.depth_image=np.zeros((48,64),dtype=np.float32)
  self.depth_time_origin=None;self.depth_episode=-1
  m=self.sim.mj_model
  # Geometry groups affect ray inclusion only; collision bits/materials remain
  # untouched. Excluding all robot links implements the agreed no-self-view.
  self.robot_ray_geoms=[]
  for g in range(m.ngeom):
   name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
   if name.startswith('go2/'):self.robot_ray_geoms.append(g)
  self.original_geom_groups=m.geom_group.copy()
  self.ray_groups=np.array([1,1,1,1,1,0],dtype=np.uint8)
  self.source_surface_ids=[g for g in range(m.ngeom) if
      (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').split('/')[-1]=='source_surface']
 def surface_ray(self,origin,direction):
  # Exact original terrain mesh, using MuJoCo's own mesh BVH. Avoid scanning
  # tens of thousands of ray-excluded convex contact cells for every pixel.
  distances=[mujoco.mj_rayMesh(self.sim.mj_model,self.sim.mj_data,g,origin,direction) for g in self.source_surface_ids]
  return min((v for v in distances if v>=0),default=-1.)
 @contextmanager
 def ray_filter(self):
  m=self.sim.mj_model;m.geom_group[self.robot_ray_geoms]=5
  try:yield
  finally:m.geom_group[:]=self.original_geom_groups
 def reset(self):
  super().reset();self.last_depth_frame=-1;self.depth_time_origin=None
  self.depth_episode=getattr(self,'depth_episode',-1)+1
 def depth_capture_step(self,sim_time):
  if self.depth_time_origin is None:self.depth_time_origin=float(sim_time)
  control_step=int((float(sim_time)-self.depth_time_origin)/self.cfg.perception_control_dt+1e-6)
  return (control_step//self.cfg.depth_period)*self.cfg.depth_period
 def height(self,xy):
  shape=xy.shape[:-1];points=np.asarray(xy).reshape(-1,2);grid=points/.05
  floor=np.floor(grid).astype(np.int64);frac=grid-floor;values=[]
  m,d=self.sim.mj_model,self.sim.mj_data
  with self.ray_filter():
   for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
    corners=floor+np.array([dx,dy]);heights=[]
    for ix,iy in corners:
     key=(int(ix),int(iy))
     if key not in self.height_cache:
      gid=np.zeros(1,dtype=np.int32)
      origin=np.array([ix*.05,iy*.05,100.]);direction=np.array([0.,0.,-1.])
      distance=self.surface_ray(origin,direction) if self.source_surface_ids else mujoco.mj_ray(m,d,origin,direction,self.ray_groups,1,-1,gid)
      self.height_cache[key]=(100-distance) if distance>=0 else 0.
     heights.append(self.height_cache[key])
    values.append(np.asarray(heights))
  fx,fy=frac[:,0],frac[:,1]
  return ((values[0]*(1-fx)+values[1]*fx)*(1-fy)+(values[2]*(1-fx)+values[3]*fx)*fy).reshape(shape)
 def depth(self,pos,R,frame):
  if frame==self.last_depth_frame:return self.depth_image
  from go2_depth_input.core import noise_keys,fields_from_keys
  origin=pos+R@self.mount;vectors=self.rays@(R@self.optical).T
  norm=np.linalg.norm(vectors,axis=1);unit=np.ascontiguousarray(vectors/norm[:,None])
  distances=np.zeros(len(unit),dtype=np.float64);geoms=np.zeros(len(unit),dtype=np.int32)
  with self.ray_filter():
   if self.source_surface_ids:
    distances[:]=[self.surface_ray(origin,vector) for vector in unit]
   else:mujoco.mj_multiRay(self.sim.mj_model,self.sim.mj_data,origin,unit.ravel(),self.ray_groups,1,-1,geoms,distances,len(unit),12.)
  z=distances/norm;valid=(distances>=0)&(z>=.1)&(z<=6.)
  image=torch.from_numpy(np.clip(z,.1,6.).reshape(1,48,64)).float();mask=torch.from_numpy(valid.reshape(1,48,64)).float()
  keys=noise_keys(torch.tensor([self.cfg.eval_seed]),frame,episode_ids=self.depth_episode)
  with torch.no_grad():
   noisy=self.depth_noise(image,torch.ones(1,dtype=torch.long),geometry_valid=mask,random=fields_from_keys(keys),hole_strength=2.)['depth']
  self.depth_image=noisy[0].numpy();self.last_depth_frame=frame;return self.depth_image
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
  if self.cfg.branch=='student':
   # Use the checkpoint's control-frame clock; resetting the robot forces a
   # fresh frame and starts a new camera phase just as a real training episode.
   obs[45:]=self.depth(pos,R,self.depth_capture_step(sim_data.sim_time)).reshape(-1)
   return obs
  ground_z=float(self.height(pos[None,:2])[0])
  state=np.concatenate(([pos[2]-ground_z],get_projected_gravity(sim_data.proprio.imu.quat),vy,wb,sim_data.proprio.joint.pos-self.training_default,sim_data.proprio.joint.vel,*points))
  cmd=obs[6:9]/np.asarray(self.cfg.control.scales.cmd)
  context=np.array([1.,0,0,0,self.payload])
  x,y=np.meshgrid(np.linspace(-.42,.78,11),np.linspace(-.36,.36,7),indexing='ij')
  xy=np.stack((pos[0]+math.cos(yaw)*x-math.sin(yaw)*y,pos[1]+math.sin(yaw)*x+math.cos(yaw)*y),-1)
  scan=pos[2]-self.height(xy).reshape(-1)
  obs[45:212]=np.concatenate((state,cmd,self.last_action,context,scan))
  y,x=np.meshgrid(np.linspace(-6,6,64),np.linspace(0,6,128),indexing='ij')
  xy=np.stack((pos[0]+math.cos(yaw)*x-math.sin(yaw)*y,pos[1]+math.sin(yaw)*x+math.cos(yaw)*y),-1)
  obs[212:-1]=np.stack((self.height(xy)-pos[2],np.ones_like(x)),0).reshape(-1)
  obs[-1]=self.cfg.task_id
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
  self.completed=False;self.stable_time=0.;self.completion_time=None
  self.origin=None
  self.task_id=getattr(cfg,'task_id',0);self.target=np.asarray(getattr(cfg,'target',[5.,0.,0.]))
 def get_goal(self,data):
  if self.start is None:self.start=data.sim_time;self.origin=np.asarray(self.sim.mj_data.qpos[:2]).copy()
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
  recovering=self.task_id==6 and not self.completed
  if ((tilt>math.radians(70) and not recovering) or not np.isfinite(d.qpos).all()):
   if self.first_failure is None:self.first_failure=t
   self.done=True
  if self.start is None:return False
  if self.task_id>=7:
   base_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'go2/base_link')
   for i,contact in enumerate(d.contact[:d.ncon]):
    bodies=[int(m.geom_bodyid[g]) for g in contact.geom]
    if base_id not in bodies:continue
    other=bodies[1-bodies.index(base_id)]
    name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,other) or ''
    if name.startswith('go2/'):continue
    force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
    if np.linalg.norm(force[:3])>5:
     self.first_failure=t;self.done=True;break
  if recovering and t>=min(8.,self.cfg.duration):
   self.first_failure=t;self.done=True
  if t>=self.cfg.duration:self.done=True
  foot_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'go2/'+leg) for leg in LEGS]
  speeds=[];geom_mask=[];force_mask=[]
  contacts=data.dynamics.contacts
  for gid,leg in zip(foot_ids,LEGS):
   vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,gid,vel,0)
   radius=m.geom_size[gid,0];bottom=np.array([0.,0.,-radius]);v=vel[3:]+np.cross(vel[:3],bottom)
   speeds.append(float(np.linalg.norm(v[:2])))
   ground_z=float(self.robot.height(d.geom_xpos[gid,None,:2])[0])
   geom_mask.append(bool(d.geom_xpos[gid,2]-ground_z<=radius+.002))
   force_mask.append(any(n.split('/')[-1]==leg and f>1. for n,f in zip(contacts.robot_geom_names,contacts.normal_forces)))
  if t>=1. and data.n_step%max(1,round(.02/data.sim_dt))==0:
   v=data.proprio.base.lin_vel;w=data.proprio.imu.ang_vel;cmd=np.asarray(self.cfg.command)
   relative_height=float(d.qpos[2]-self.robot.height(d.qpos[None,:2])[0])
   self.samples.append({'time':t,'xy_sq':float(np.mean((v[:2]-cmd[:2])**2)),'yaw_sq':float((w[2]-cmd[2])**2),'height':relative_height,'vx':float(v[0]),'vy':float(v[1]),'wz':float(w[2]),'roll_pitch_tilt':tilt,'foot_speed':speeds,'geometric_contact':geom_mask,'force_contact':force_mask})
  if data.n_step%max(1,round(.02/data.sim_dt))==0:self.trace.append({'time':t,'qpos':d.qpos.tolist(),'foot_speed':speeds,'geometric_contact':geom_mask,'force_contact':force_mask})
  if data.n_step%max(1,round(.02/data.sim_dt))==0 and self.task_id in (3,4,5,6,8,9):
   pos=d.qpos[:3];ground_z=float(self.robot.height(pos[None,:2])[0]);height=pos[2]-ground_z
   route=self.target[:2]-self.origin;direction=route/max(np.linalg.norm(route),1e-6);offset=pos[:2]-self.target[:2]
   radius=.8  # PhysX reset's target-radius contract, unchanged by the adapter.
   region=float(offset@direction)>=-radius and abs(float(offset[0]*direction[1]-offset[1]*direction[0]))<=radius and abs(pos[2]-self.target[2]-.38)<.28
   eligible=tilt<math.radians(35) and sum(force_mask)>=2 and height>=.24 and np.linalg.norm(data.proprio.imu.ang_vel)<3 and (self.task_id==6 or region)
   self.stable_time=self.stable_time+.02 if eligible else 0.
   if not self.completed and self.stable_time>=(.64 if self.task_id==5 else .32):
    self.completed=True;self.completion_time=t
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
  result['task_id']=self.task_id;result['task_completed']=self.completed;result['completion_seconds']=self.completion_time
  if self.task_id in (5,6,8,9):result['pass']=bool(result['survived'] and self.completed)
  return result

class AttachedPipeline(upstream.BasePipeline):
 def load(self):
  super().load();self.gauge.sim=self.sim;self.gauge.robot=self.robot
  if isinstance(self.robot,CTSPerceptive):
   self.robot.sim=self.sim;self.robot.training_default=np.asarray(self.robot_cfg.training_default);self.robot.payload=self.sim_cfg.domain_rand.base_mass
   self.robot.setup_perception()
  initial=getattr(self.robot_cfg,'recovery_qpos',None)
  if initial is not None:
   value=np.asarray(initial,dtype=np.float64)
   if value.shape!=self.sim.mj_data.qpos.shape:raise ValueError('Recovery recorded pose dimension mismatch')
   self.sim.mj_data.qpos[:]=value;self.sim.mj_data.qvel[:]=0
   mujoco.mj_forward(self.sim.mj_model,self.sim.mj_data)
   self.first_reset=False
  self.loaded=loaded_model(self.sim,self.robot)
  self.robot.reset()

def evaluate_case(exports,iteration,branch,command,payload,output,seed=17,video=False,official_noise_delay=False,task_id=0,task_map=None):
 if mujoco.__version__!='3.2.3':raise RuntimeError('Use the pinned official MuJoCo3.2.3 runtime; no version bypass.')
 torch.set_num_threads(1);torch.set_grad_enabled(False);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 meta=json.loads((Path(exports)/f'{iteration}_metadata.json').read_text());const=meta['consts']
 if meta.get('schema')!='depth_cts_export_v2' or meta.get('architecture_version')!='physx-depth-multitask-v2':
  raise ValueError('Architecture mismatch: official bridge requires the versioned V2 export')
 robot=Go2Lab131Config();sim=MujocoConfig();gauge=FlatGaugeConfig()
 robot.control.model_path=meta['models'][branch]['path'];robot.robot_class='CTSPerceptive'
 robot.control.num_observations=meta['models'][branch]['input_dim']
 robot.depth_package=meta['depth_package'];robot.branch=branch;robot.task_id=task_id;robot.eval_seed=seed
 camera=meta['camera'];robot.depth_period=int(camera['depth_period']);robot.perception_control_dt=float(camera['control_dt'])
 if robot.depth_period<=0 or abs(robot.perception_control_dt-.02)>1e-9:raise ValueError('Unsupported checkpoint perception/control cadence')
 if abs(float(camera['rate_hz'])-1/(robot.depth_period*robot.perception_control_dt))>1e-6:raise ValueError('Inconsistent exported camera rate')
 if task_map is not None:
  gauge.assets.terrain_xmls=[task_map['xml']];gauge.assets.terrain_spawn_pos=task_map.get('spawn',[0,0,0])
  gauge.assets.terrain_name=f"depth_task{task_id}_level{task_map.get('level',0)}"
  gauge.assets.terrain_level=task_map.get('level',0)
  gauge.backward=task_map.get('backward',False)
  if task_id==6:
   if 'recovery_qpos' not in task_map:raise ValueError('Recovery evaluation requires a recorded verified PhysX initialization')
   # Upstream compares -gravity.z to cos(threshold). A value above pi lowers
   # the allowed angle again, so exactly pi is required for inverted starts.
   # Other official penetration checks stay enabled; task recovery/deadline
   # and post-recovery fall rules are handled by CommandGauge.
   robot.recovery_qpos=task_map['recovery_qpos'];sim.truncation.projected_gravity_rad=math.pi
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
 gauge.gauge_class='CommandGauge';gauge.command=list(command);gauge.duration=10.;gauge.task_id=task_id
 gauge.target=task_map.get('goal',[5.,0.,0.]) if task_map else [5.,0.,0.]
 upstream.CTSPerceptive=CTSPerceptive;upstream.CommandGauge=CommandGauge
 fixture=task_map.get('label','default') if task_map else 'default'
 tag=f'{iteration}_{branch}_task{task_id}_{fixture}_checkpoint_control_{command[0]}_{command[1]}_{command[2]}_load{payload}_s{seed}'
 logger.create('official_cases',tag,console_output=False,parent_log_dir=output)
 pipeline=AttachedPipeline(tag,sim,robot,gauge,SimpleNamespace(seed=seed))
 _,warning,error=pipeline.run();result=pipeline.gauge.summary()
 result.update(iteration=iteration,branch=branch,fixture_label=fixture,command=list(command),payload_kg=payload,control_contract='checkpoint',seed=seed,checkpoint_sha256=meta['checkpoint_sha256'],upstream_commit=UPSTREAM_COMMIT,warning=str(warning) if warning else None,error=str(error) if error else None,training_updates=0,protocol={'physics_dt':sim.physics.simulation_dt,'control_dt':robot.perception_control_dt,'depth_dt':robot.depth_period*robot.perception_control_dt,'metric_dt':.02,'duration':10,'ignore_seconds':1,'initial_settle':'recorded PhysX recovery pose' if task_id==6 else 'official BasePipeline first_reset','depth':'actual MuJoCo ray geometry + frozen supplied noise model','noise_enabled':official_noise_delay,'random_action_delay':official_noise_delay},case_dir=str(logger.log_dir))
 pipeline.loaded['observed_contact_parameters']=list(pipeline.gauge.contact_parameters.values())
 pipeline.loaded['perception']={'mount':pipeline.robot.mount.tolist(),'optical_axes':'right/down/forward',
     'depth_hz':float(camera['rate_hz']),'depth_period':robot.depth_period,'noise_key_frame':'episode_control_step',
     'height_query_resolution_m':.05,'history':'term-major45x10','gru_reset':'official robot.reset',
     'geometry':'actual loaded MuJoCo ray intersections','self_geometry_excluded':True,
     'noise_weights_sha256':hashlib.sha256((Path(robot.depth_package)/'go2_depth_input/assets/weights.pt').read_bytes()).hexdigest()}
 (logger.log_dir/'loaded_parameters.json').write_text(json.dumps(pipeline.loaded,indent=2))
 (logger.log_dir/'metrics.json').write_text(json.dumps(result,indent=2))
 (logger.log_dir/'trajectory.json').write_text(json.dumps(pipeline.gauge.trace))
 return result

def task_map_for(task_id,level,manifest):
 import robogauge
 root=Path(robogauge.__file__).resolve().parent.parent/'resources/terrains'
 if task_id in (0,1):return {'xml':str(root/'flat.xml'),'spawn':[0,0,0],'goal':[5,0,0],'level':level}
 if task_id==2:return {'xml':str(root/f'wave/wave_{level+1}.xml'),'spawn':[1.5,1.25,0.], 'goal':[6.5,1.25,0.],'level':level}
 if task_id in (3,4):
  import xml.etree.ElementTree as ET
  path=root/f'stairs/stairs_{level+1}.xml'
  first=next(g for g in ET.parse(path).iter('geom') if g.get('type')=='box')
  step=2*float(first.get('size').split()[2])
  return {'xml':str(root/f'stairs/stairs_{level+1}.xml'),'spawn':[-.8,0,0] if task_id==3 else [2.015,0,7*step],
          'goal':[1.4,0,5*step] if task_id==3 else [-.5,0,0],'level':level,'backward':task_id==4}
 rows=[m for m in manifest.get('maps',[]) if m['task_id']==task_id and m.get('level',0)==level]
 if not rows:raise ValueError(f'Missing exact task{task_id} level{level} official-chain asset')
 return rows[0]

def task_maps_for(task_id,level,manifest):
 if task_id!=6:return [task_map_for(task_id,level,manifest)]
 import robogauge
 root=Path(robogauge.__file__).resolve().parent.parent/'resources/terrains'
 fixtures=json.loads((Path(__file__).parent/'recovery_fixtures.json').read_text())
 return [dict(row,task_id=6,level=0,xml=str(root/'flat.xml'),spawn=[0,0,0],goal=[5,0,0],
              scope=fixtures['scope']) for row in fixtures['records']]

def main():
 p=argparse.ArgumentParser();p.add_argument('--exports',required=True);p.add_argument('--output',required=True);p.add_argument('--iterations',default='5000,10000,40000');p.add_argument('--forward-only',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--official-noise-delay',action='store_true')
 p.add_argument('--payloads',default='0,1.2');p.add_argument('--forward-speeds',default='')
 p.add_argument('--high-speed-probes',default='',help='Additional task2 speeds; excluded from the fixed stability cohort')
 p.add_argument('--allocation-cohort',action='store_true',help='Matched five-seed/two-payload task allocation panel')
 p.add_argument('--tasks',default='0');p.add_argument('--level',type=int,default=0);p.add_argument('--task-maps');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 (out/'runtime_verification.json').write_text(json.dumps(verify_runtime(a.exports,list(map(int,a.iterations.split(',')))),indent=2))
 commands=[(.5,0,0)] if a.forward_only else [(0,0,0),(.5,0,0),(-.3,0,0),(0,.3,0),(0,0,.8)]
 for speed in filter(None,a.forward_speeds.split(',')):
  cmd=(float(speed),0,0)
  if cmd not in commands:commands.append(cmd)
 rows=[];skipped=[];manifest=json.loads(Path(a.task_maps).read_text()) if a.task_maps else {'maps':[]}
 for it in map(int,a.iterations.split(',')):
  for branch in ('teacher','student'):
   for task_id in map(int,a.tasks.split(',')):
    try:task_maps=task_maps_for(task_id,a.level,manifest)
    except ValueError as error:
     skipped.append({'task_id':task_id,'branch':branch,'reason':str(error)});continue
    cases=commands if task_id==0 else [(1.5 if task_id==1 else 1. if task_id in (5,8,9) else .5,0,0)]
    if task_id==1:
     for speed in filter(None,a.high_speed_probes.split(',')):
      cmd=(float(speed),0,0)
      if cmd not in cases:cases.append(cmd)
    for task_map in task_maps:
     for cmd in cases:
      for payload in ((0.,) if a.forward_only else map(float,a.payloads.split(','))):
       row=evaluate_case(a.exports,it,branch,cmd,payload,out,video=a.video,official_noise_delay=a.official_noise_delay,task_id=task_id,task_map=task_map)
       row['cohort_role']='high_speed_probe' if task_id==1 and cmd[0]!=1.5 else 'fixed_protocol'
       row['map_sha256']=hashlib.sha256(Path(task_map['xml']).read_bytes()).hexdigest();rows.append(row)
       (out/'results.json').write_text(json.dumps({'rows':rows,'skipped':skipped,'training_updates':0,'upstream_commit':UPSTREAM_COMMIT},indent=2));print(json.dumps(row),flush=True)
  if a.allocation_cohort:
   from .allocation_evaluation import plan_cases
   from .task_allocation import summarize_cohort
   meta=json.loads((Path(a.exports)/f'{it}_metadata.json').read_text())
   protocol={'version':1,'pass_rule':'existing_CommandGauge','duration':10.,'ignore_seconds':1.,
             'camera':meta['camera'],'consts':meta['consts'],'noise_delay':a.official_noise_delay,'mujoco':mujoco.__version__,
             'upstream_commit':UPSTREAM_COMMIT,'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'noise_sha256':hashlib.sha256((Path(meta['depth_package'])/'go2_depth_input/assets/weights.pt').read_bytes()).hexdigest()}
   plans,missing=plan_cases(it,manifest,protocol)
   (out/'allocation_expected.json').write_text(json.dumps({'cases':[r for r,m in plans],'missing':missing},indent=2))
   for case,task_map in plans:
    mapped=dict(task_map,label='allocation_'+task_map.get('label',f"terrain{case['terrain_id']}_layout{task_map.get('layout',0)}"))
    try:
     row=evaluate_case(a.exports,it,case['branch'],case['command'],case['payload_kg'],out,seed=case['seed'],
         video=a.video and case['seed']==17 and case['payload_kg']==0.,official_noise_delay=a.official_noise_delay,task_id=case['task_id'],task_map=mapped)
    except Exception as error:row={'task_id':case['task_id'],'branch':case['branch'],'error':str(error),'pass':None}
    row.update(iteration=it,cohort_role='task_allocation',allocation_case_id=case['case_id'],terrain_id=case['terrain_id'])
    rows.append(row)
    summary=summarize_cohort(it,rows,[r for r,m in plans])
    (out/'results.json').write_text(json.dumps({'rows':rows,'skipped':skipped,'allocation_missing':missing,
        'task_allocation':summary,'training_updates':0,'upstream_commit':UPSTREAM_COMMIT},indent=2))
if __name__=='__main__':main()
