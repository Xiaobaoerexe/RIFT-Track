"""Frozen As2 evaluation using pinned RoboGauge simulation/control lifecycle."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,math,argparse,random,hashlib,time
from types import SimpleNamespace
import numpy as np
import torch,mujoco
ROOT=Path(__file__).resolve().parents[2]
OUT=None
ITERATION=120000
sys.path.insert(0,str(ROOT))
from go2_physx_cts.robogauge_bridge import verify_runtime,yaw_inverse
from robogauge.tasks.pipeline import base_pipeline as upstream
from robogauge.tasks.robots.go2.go2 import Go2
from robogauge.tasks.robots.go2.go2_lab_config import Go2Lab131Config
from robogauge.tasks.simulator.mujoco_config import MujocoConfig
from robogauge.tasks.gauge.gauge_configs.flat_gauge_config import FlatGaugeConfig
from robogauge.tasks.gauge.goal_data import GoalData,VelocityGoal
from robogauge.utils.math_utils import get_projected_gravity
from robogauge.utils.logger import logger
LEGS=('FL','FR','RL','RR')
GROUP=np.array([1,0,0,0,0,0],np.uint8)
def ground(sim,xy):
 m,d=sim.mj_model,sim.mj_data
 distance=mujoco.mj_ray(m,d,np.array([xy[0],xy[1],20.]),np.array([0.,0.,-1.]),GROUP,1,-1,np.zeros(1,np.int32))
 return 20.-distance if distance>=0 else float('nan')
class As2Teacher(Go2):
 def build_observation(self,sd,goal):
  obs=super().build_observation(sd,goal);m,d=self.sim.mj_model,self.sim.mj_data
  base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'as2/base_link')
  pos=d.xpos[base];R=d.xmat[base].reshape(3,3);yaw=math.atan2(R[1,0],R[0,0])
  center=ground(self.sim,pos[:2])
  points=[]
  for part in ('thigh','calf'):
   ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'as2/{leg}_{part}') for leg in LEGS]
   points.append(yaw_inverse(d.xpos[ids]-pos,yaw).reshape(-1))
  ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'as2/{leg}_foot_frame') for leg in LEGS]
  points.append(yaw_inverse(d.site_xpos[ids]-pos,yaw).reshape(-1))
  state=np.concatenate(([pos[2]-center],get_projected_gravity(sd.proprio.imu.quat),yaw_inverse(R@sd.proprio.base.lin_vel,yaw),
         sd.proprio.imu.ang_vel,sd.proprio.joint.pos-self.default_dof_pos,sd.proprio.joint.vel,*points))
  offsets=np.array([(x,y) for x in np.arange(11)*.12-.42 for y in np.arange(7)*.12-.36])
  c,s=math.cos(yaw),math.sin(yaw);coords=pos[:2]+offsets@np.array([[c,s],[-s,c]])
  heights=np.array([ground(self.sim,q) for q in coords])
  scan=pos[2]-center+np.clip(center-heights,-1.2,1.2)
  cmd=obs[6:9]/np.array(self.cfg.control.scales.cmd)
  obs[45:]=np.concatenate((state,cmd,self.last_action,[1.,0,0,0,0],scan))
  assert np.isfinite(obs).all()
  return obs
class As2Gauge:
 def __init__(self,cfg,robot_cfg):
  self.cfg=cfg;self.goal_str='as2_fixed_velocity';self.results={};self.done=False;self.start=None
  self.failure=None;self.reason=None;self.samples=[];self.trace=[];self.course=False;self.finish_time=None
 def get_goal(self,sd):
  if self.start is None:self.start=sd.sim_time
  return GoalData('velocity',VelocityGoal(lin_vel_x=self.cfg.speed,lin_vel_y=0.,ang_vel_yaw=0.))
 def is_done(self):return self.done
 def is_reset(self,sd):return False
 def reset_metrics(self):pass
 def reset_current_goal(self):self.failure='pipeline_penetration_reset';self.done=True
 def switch_to_next_goal(self):self.failure='pipeline_exception';self.done=True
 def update_metrics(self,sd,goal):
  m,d=self.sim.mj_model,self.sim.mj_data
  tilt=math.acos(float(np.clip(-get_projected_gravity(sd.proprio.imu.quat)[2],-1,1)))
  t=0. if self.start is None else sd.sim_time-self.start
  if not np.isfinite(d.qpos).all() or tilt>math.radians(70):
   self.failure='nonfinite_or_tilt';self.done=True;self.finish_time=t
  if self.start is None:return False
  if sd.n_step%10:return False
  height=float(d.qpos[2]-ground(self.sim,d.qpos[:2]))
  if not np.isfinite(height) or height<.12:
   self.failure='low_body_or_no_ground';self.done=True
  if not self.cfg.flat:
   if abs(d.qpos[1])>3.5 or d.qpos[0]<-3.5:self.failure='left_course';self.done=True
   if d.qpos[0]>=3.5:self.course=True;self.done=True
  if t>=self.cfg.duration:self.done=True
  if self.done:self.finish_time=t
  slip=[]
  for idx,contact in enumerate(d.contact[:d.ncon]):
   g1,g2=map(int,contact.geom);gid=g1 if g1 in self.foot_ids else g2 if g2 in self.foot_ids else None
   if gid is None:continue
   other=g2 if gid==g1 else g1
   if m.geom_bodyid[other]!=0:continue
   force=np.zeros(6);mujoco.mj_contactForce(m,d,idx,force)
   if force[0]<=1.:continue
   vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,gid,vel,0)
   v=vel[3:]+np.cross(vel[:3],contact.pos-d.geom_xpos[gid]);normal=contact.frame[:3]
   slip.append(float(np.linalg.norm(v-normal*np.dot(v,normal))))
  v=sd.proprio.base.lin_vel;w=sd.proprio.imu.ang_vel
  self.trace.append([t,*d.qpos.tolist(),float(v[0]),float(v[1]),float(w[2])])
  if t>=1.:
   self.samples.append(dict(t=t,vx=float(v[0]),vy=float(v[1]),wz=float(w[2]),height=height,tilt=tilt,
    speed_sq=float((v[0]-self.cfg.speed)**2),xy_sq=float(np.mean((v[:2]-[self.cfg.speed,0])**2)),
    slip=slip,velocity_exceeded=bool(np.any(np.abs(sd.proprio.joint.vel)>np.array([24,24,16]*4)*1.01))))
  return False
 def summary(self):
  rows=self.samples;survived=self.failure is None and bool(rows) and self.done
  result={'survived':survived,'course_completed':self.course,'failure':self.failure,'elapsed_seconds':self.finish_time,'samples':len(rows)}
  for k in ('vx','vy','wz','height','tilt'):
   result[k]=float(np.mean([r[k] for r in rows])) if rows else None
  for name,key in [('vx_rmse','speed_sq'),('xy_rmse','xy_sq')]:
   result[name]=float(np.sqrt(np.mean([r[key] for r in rows]))) if rows else None
  result['yaw_rmse']=float(np.sqrt(np.mean([r['wz']**2 for r in rows]))) if rows else None
  vals=[v for r in rows for v in r['slip']]
  result['slip_mps']=float(np.mean(vals)) if vals else None
  result['joint_velocity_exceed_fraction']=float(np.mean([r['velocity_exceeded'] for r in rows])) if rows else None
  result['tracking_pass']=bool(survived and result['vx_rmse']<max(.2,.2*self.cfg.speed) and result['yaw_rmse']<.3)
  return result
class As2Pipeline(upstream.BasePipeline):
 def load(self):
  super().load();m,d=self.sim.mj_model,self.sim.mj_data
  self.gauge.sim=self.sim;self.gauge.foot_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'as2/'+leg) for leg in LEGS]
  assert min(self.gauge.foot_ids)>=0
  if isinstance(self.robot,As2Teacher):self.robot.sim=self.sim
  rng=np.random.default_rng(self.args.seed);d.qpos[0:2]+=rng.uniform(-.04,.04,2)
  mujoco.mj_forward(m,d);self.robot.reset()
  names=[f'{leg}_{part}_joint' for leg in LEGS for part in ('hip','thigh','calf')]
  assert [n.split('/')[-1] for n in self.sim.dof_names]==names
  act=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(i)).split('/')[-1] for i in m.actuator_trnid[:,0]]
  assert act==names
  assert np.allclose(m.dof_damping[6:],0) and np.allclose(m.dof_frictionloss[6:],0)
  assert np.allclose(m.dof_armature[6:],[.026,.026,.038]*4)
  assert np.allclose(m.actuator_ctrlrange[:,1],[50,50,75]*4)
  assert abs(m.body_mass.sum()-17.64)<1e-5
  assert np.allclose(self.sim.get_sensor_data('joint_pos'),d.qpos[7:],atol=1e-8)
  audit={'mass':float(m.body_mass.sum()),'dof_order':names,'actuator_order':act,'armature':m.dof_armature[6:].tolist(),
   'damping':m.dof_damping[6:].tolist(),'frictionloss':m.dof_frictionloss[6:].tolist(),'joint_ranges':m.jnt_range[1:].tolist(),
   'torque_limits':m.actuator_ctrlrange.tolist(),'kp':self.robot.p_gains.tolist(),'kd':self.robot.d_gains.tolist(),
   'default_q':self.robot.default_dof_pos.tolist(),'action_scale':np.asarray(self.robot.action_scale).tolist(),
   'geom_count':m.ngeom,'contact_friction_unique':np.unique(m.geom_friction,axis=0).tolist(),
   'physics_dt':m.opt.timestep,'solver':int(m.opt.solver),'integrator':int(m.opt.integrator),
   'contact_solref_unique':np.unique(m.geom_solref,axis=0).tolist(),'contact_solimp_unique':np.unique(m.geom_solimp,axis=0).tolist()}
  (logger.log_dir/'loaded_contract.json').write_text(json.dumps(audit,indent=2))
def run_case(info,branch,speed):
 seed=info['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 meta=json.loads((OUT/'exports'/f'{ITERATION}_metadata.json').read_text());c=meta['consts']
 rc=Go2Lab131Config();sc=MujocoConfig();gc=FlatGaugeConfig()
 rc.robot_name='as2';rc.robot_class='As2Teacher' if branch=='teacher' else 'Go2'
 rc.assets.robot_xml=str(OUT/'as2_checkpoint.xml');rc.assets.foot_geom_names=list(LEGS)
 rc.control.model_path=meta['models'][branch]['path'];rc.control.num_observations=212 if branch=='teacher' else 45
 rc.control.default_dof_pos=c['default_q'];rc.control.p_gains=c['kp'];rc.control.d_gains=c['kd'];rc.control.scales.action=np.array(c['action_scale'])
 rc.control.mj2model_dof_indices=list(range(12))
 rc.commands.lin_vel_x=[-6,6];rc.commands.lin_vel_y=[-1,1];rc.commands.ang_vel_yaw=[-2,2]
 sc.viewer.headless=True;sc.noise.enabled=False;sc.domain_rand.action_delay=False;sc.render.save_video=False
 gc.gauge_class='As2Gauge';gc.speed=speed;gc.duration=12.;gc.flat=info['name']=='flat'
 gc.assets.terrain_xmls=[info['path']];gc.assets.terrain_spawn_pos=info['spawn'];gc.backward=False
 upstream.As2Teacher=As2Teacher;upstream.As2Gauge=As2Gauge
 tag=f"{info['name']}_l{info['level']}_s{seed}_{branch}_v{speed:g}"
 logger.create('cases',tag,console_output=False,parent_log_dir=OUT)
 p=As2Pipeline(tag,sc,rc,gc,SimpleNamespace(seed=seed))
 begin=time.monotonic();_,warning,error=p.run()
 row=p.gauge.summary();row.update(terrain=info['name'],level=info['level'],difficulty=info['difficulty'],seed=seed,branch=branch,command=speed,
  error=str(error) if error else None,warning=str(warning) if warning else None,wall_seconds=time.monotonic()-begin,case_dir=str(logger.log_dir),
  terrain_representation=info['method'],payload_kg=0)
 if error or warning:row['survived']=False;row['tracking_pass']=False
 np.savez_compressed(logger.log_dir/'trajectory.npz',trace=np.asarray(p.gauge.trace))
 (logger.log_dir/'metrics.json').write_text(json.dumps(row,indent=2))
 return row
def main():
 global OUT,ITERATION
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--worker',type=int,default=0)
 p.add_argument('--output',required=True);p.add_argument('--iteration',type=int,default=120000);p.add_argument('--speeds',default='');a=p.parse_args()
 OUT=Path(a.output).expanduser().resolve();ITERATION=a.iteration
 torch.set_num_threads(1);torch.set_grad_enabled(False)
 verified=verify_runtime(OUT/'exports',[ITERATION]);(OUT/f'runtime_verification_{a.worker}.json').write_text(json.dumps(verified,indent=2))
 items=json.loads((OUT/'terrain_manifest.json').read_text());seed=(17,29,43)[a.worker]
 rows=[];suffix='smoke' if a.smoke else str(a.worker)
 for info in items:
  if info['seed']!=seed or (a.smoke and info['name']!='flat'):continue
  speeds=list(map(float,a.speeds.split(','))) if a.speeds else ([.5] if a.smoke else ([.5,1.,1.5,4.2,5.] if info['name']=='flat' else [.5,1.,1.5]))
  for speed in speeds:
   for branch in ('teacher','student'):
    row=run_case(info,branch,speed);rows.append(row)
    (OUT/f'results_{suffix}.json').write_text(json.dumps({'complete':False,'rows':rows},indent=2))
    print(json.dumps(row),flush=True)
 (OUT/f'results_{suffix}.json').write_text(json.dumps({'complete':True,'rows':rows},indent=2))
 print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
