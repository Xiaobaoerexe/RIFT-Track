import os
os.environ['MUJOCO_GL']='egl'
import json,sys,numpy as np,mujoco,torch
from pathlib import Path
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/abc/Documents/go2_supermimic_lab/physx_rebuild')
from go2_physx_cts.robogauge_bridge import *
from go2_physx_cts import state as st
r=Path('/home/abc/data2/go2_supermimic_lab/reports/ROBOGAUGE_MIGRATION_20261029');meta=json.loads((r/'exports/40000_metadata.json').read_text());c=meta['consts']
rc=Go2Lab131Config();sc=MujocoConfig();gc=FlatGaugeConfig();sc.viewer.headless=True;sc.noise.enabled=False;sc.domain_rand.action_delay=False
rc.control.model_path=meta['models']['teacher']['path'];rc.control.num_observations=212;rc.robot_class='CTSTeacher';rc.training_default=c['default_q'];rc.control.default_dof_pos=c['default_q'];rc.control.p_gains=c['kp'];rc.control.d_gains=c['kd'];rc.control.scales.action=np.array(c['action_scale'])
gc.gauge_class='CommandGauge';gc.command=[.5,.2,.7];gc.duration=10;upstream.CTSTeacher=CTSTeacher;upstream.CommandGauge=CommandGauge
logger.create('interface_audit','actual_loaded',console_output=False,parent_log_dir=r)
p=AttachedPipeline('audit',sc,rc,gc,SimpleNamespace(seed=17));p.load();m,d=p.sim.mj_model,p.sim.mj_data
native=json.loads((r/'native_evidence/trace_00040000.json').read_text())['first_inputs']['state'][0]
d.qpos[:3]=[0,0,.29];d.qpos[3:7]=[1,0,0,0];d.qpos[7:]=c['default_q'];d.qvel[:]=0;mujoco.mj_forward(m,d)
points=np.concatenate([d.xpos[[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'go2/{leg}_{part}') for leg in LEGS]].reshape(-1)-np.tile(d.qpos[:3],4) for part in ('thigh','calf','foot')])
kinerr=float(np.max(np.abs(points-np.array(native)[34:70])))
tests=[]
for euler in ((0,0,0),(.2,-.15,.7)):
 d.qpos[:3]=[0,0,1.];q=Rotation.from_euler('xyz',euler).as_quat();d.qpos[3:7]=q[[3,0,1,2]];d.qpos[7:]=np.array(c['default_q'])+np.linspace(-.02,.02,12);d.qvel[:]=np.linspace(-.3,.4,m.nv);mujoco.mj_forward(m,d)
 sd=p.sim.step();goal=GoalData('velocity',VelocityGoal(lin_vel_x=.5,lin_vel_y=.2,ang_vel_yaw=.7));p.robot.last_action=np.linspace(-.2,.3,12)
 obs=p.robot.build_observation(sd,goal);q=sd.proprio.imu.quat;R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();gravity=R.T@np.array([0,0,-1.])
 expected=np.concatenate((sd.proprio.imu.ang_vel*.25,gravity,[.25,.2,.7],sd.proprio.joint.pos-np.array(c['default_q']),sd.proprio.joint.vel*.05,p.robot.last_action))
 err=float(np.max(np.abs(obs[:45]-expected)));assert err<1e-6
 s=torch.tensor(obs[45:115]).reshape(1,-1);a=torch.tensor(p.robot.last_action).reshape(1,-1);cmd=torch.tensor([[.5,.2,.7]])
 frame=st.frame(s,a,cmd).numpy()[0];assert np.max(np.abs(frame-obs[:45]))<1e-6
 tests.append({'euler':euler,'frame_error':err,'joint_sensor_qpos_error':float(np.max(np.abs(sd.proprio.joint.pos-d.qpos[7:]))),'gyro_vs_freejoint_local_angular_velocity_error':float(np.max(np.abs(sd.proprio.imu.ang_vel-d.qvel[3:6]))),'unused_official_base_ang_vel_vs_gyro_error':float(np.max(np.abs(sd.proprio.base.ang_vel-sd.proprio.imu.ang_vel)))})
 p.robot.reset();assert np.max(np.abs(p.robot.last_action))==0
out={'kinematic_anatomical_frames_vs_native_initial_state_max_error_m':kinerr,'sensor_frame_checks':tests,'dof_order':p.sim.dof_names,'normalization':'frozen checkpoint running mean/variance inside TorchScript; no additional normalization in official Go2','history':'term-major 10 frames, zero reset, current frame appended before action; golden50 frames reset0/25 verified','action':'linear unbounded network output to default + scale*action; no tanh or additional clipping; official motor ctrlrange remains','reset':'official 0.445m spawn then zero-command settling; native0.29m immediate command, intentionally distinct'}
(r/'interface_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
