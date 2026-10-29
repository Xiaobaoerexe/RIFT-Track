"""Visualize measured PhysX poses; never integrate MuJoCo dynamics."""
import os
os.environ['MUJOCO_GL']='egl'
import json,numpy as np,mujoco
from pathlib import Path
from robogauge.tasks.simulator import MujocoSimulator,MujocoConfig
from robogauge.tasks.gauge.gauge_configs.flat_gauge_config import FlatGaugeConfig
from robogauge.tasks.robots.go2.go2_lab_config import Go2Lab131Config
from robogauge.utils.logger import logger
r=Path('/home/abc/data2/go2_supermimic_lab/reports/ROBOGAUGE_MIGRATION_20261029')
a=np.load(r/'native_evidence/poses_00040000.npz');rows=json.loads((r/'native_evidence/physx_00040000.json').read_text())['rows'];meta=json.loads((r/'exports/40000_metadata.json').read_text())
records=[]
for branch in ('teacher','student'):
 idx=next(i for i,x in enumerate(rows) if x['branch']==branch and x['command']==[.5,0,0] and x['payload_kg']==0)
 cfg=MujocoConfig();cfg.viewer.headless=True;cfg.render.save_video=True;cfg.render.video_fps=50
 logger.create('physx_measured_pose_visualization',branch,console_output=False,parent_log_dir=r/'native_videos')
 sim=MujocoSimulator(cfg);terrain=FlatGaugeConfig();robot=Go2Lab131Config()
 sim.load(terrain.assets.terrain_xmls,robot.assets.robot_xml,terrain.assets.terrain_spawn_pos,meta['consts']['default_q'],False)
 for pose,joints in zip(a['root_pose'][:,idx],a['joint_pos'][:,idx]):
  sim.mj_data.qpos[:3]=pose[:3];sim.mj_data.qpos[3:7]=pose[[6,3,4,5]];sim.mj_data.qpos[7:]=joints
  mujoco.mj_forward(sim.mj_model,sim.mj_data)
  sim.renderer.update_scene(sim.mj_data,camera=sim.offscreen_cam);sim.vid_writer.append_data(sim.renderer.render())
 sim.close_video_writer();sim.close_viewer()
 records.append({'branch':branch,'source':'native_evidence/poses_00040000.npz','case_index':idx,'frames':len(a['root_pose']),'fps':50,'description':'Measured PhysX pose replay on RoboGauge mesh; no MuJoCo dynamics and no native PhysX camera rendering','directory':str(logger.log_dir)})
(r/'native_videos/manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
