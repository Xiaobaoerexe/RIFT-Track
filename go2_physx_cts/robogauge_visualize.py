"""Render recorded PhysX poses with official RoboGauge assets; no dynamics run."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,struct
from pathlib import Path
import numpy as np
import mujoco
import imageio.v3 as imageio
from robogauge.tasks.simulator import MujocoSimulator,MujocoConfig
from robogauge.tasks.robots.go2.go2_lab_config import Go2Lab131Config
from robogauge.tasks.gauge.gauge_configs.flat_gauge_config import FlatGaugeConfig
from robogauge.utils.logger import logger
from robogauge.utils.helpers import parse_path

def height_asset(record,out):
    path=Path(record['terrain_height']);h=np.load(path).astype(np.float32)
    if h.ndim!=2 or not np.isfinite(h).all():raise ValueError('Finite measured height matrix required')
    origin=np.asarray(record['height_origin']);offset=np.asarray(record.get('world_offset',[0,0,0]))
    res=float(record['height_resolution']);span=max(.01,float(h.max()-h.min()))
    binary=out/(record['label']+'_height.bin')
    binary.write_bytes(struct.pack('ii',*h.shape)+((h-h.min())/span).astype(np.float32).tobytes())
    xy=origin[:2]+np.array([h.shape[1]-1,h.shape[0]-1])*res/2-offset[:2]
    z=float(h.min()-offset[2]);half=np.array([h.shape[1]-1,h.shape[0]-1])*res/2
    xml=out/(record['label']+'_terrain.xml')
    xml.write_text(f'<mujoco model="recorded_height"><visual><global offwidth="1920" offheight="1080"/></visual><asset><hfield name="measured" file="{binary.name}" size="{half[0]} {half[1]} {span} .05"/></asset><worldbody><light pos="0 0 5" dir="0 0 -1" directional="true"/><geom type="hfield" hfield="measured" pos="{xy[0]} {xy[1]} {z}" rgba=".5 .55 .6 1"/></worldbody></mujoco>')
    return str(xml)

def render_records(source,output):
    if mujoco.__version__!='3.2.3':raise RuntimeError('Pinned official MuJoCo3.2.3 required')
    records=json.loads(Path(source).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    robot=Go2Lab131Config();result=[];maps=[]
    for row in records['records']:
        cfg=MujocoConfig();cfg.viewer.headless=True;cfg.render.save_video=False
        logger.create('recorded_physx_poses',row['label'],console_output=False,parent_log_dir=out)
        sim=MujocoSimulator(cfg)
        terrain=height_asset(row,out) if row.get('terrain_height') else parse_path(row.get('terrain_xml',FlatGaugeConfig.assets.terrain_xmls[0]))
        sim.load([terrain],robot.assets.robot_xml,[0,0,0],records['consts']['default_q'],False)
        pose=np.asarray(row['pose_xyzw']);offset=np.asarray(row.get('world_offset',[0,0,0]))
        sim.mj_data.qpos[:3]=pose[:3]-offset;sim.mj_data.qpos[3:7]=pose[[6,3,4,5]]
        sim.mj_data.qpos[7:]=np.asarray(row['joint_pos']);mujoco.mj_forward(sim.mj_model,sim.mj_data)
        maps.append({'task_id':6,'level':0,'source_terrain_level':row.get('terrain_level'),
                     'xml':str(Path(terrain).resolve()),'spawn':[0,0,0],'goal':[5,0,0],
                     'recovery_qpos':sim.mj_data.qpos.copy().tolist(),'label':row['label'],
                     'initialization_source':'recorded and visually inspected PhysX pose'})
        renderer=mujoco.Renderer(sim.mj_model,height=480,width=640)
        sim.offscreen_cam.lookat[:]=sim.mj_data.qpos[:3]
        sim.offscreen_cam.distance=2.0;sim.offscreen_cam.elevation=-22;sim.offscreen_cam.azimuth=45
        renderer.update_scene(sim.mj_data,camera=sim.offscreen_cam)
        path=out/(row['label']+'.png');imageio.imwrite(path,renderer.render())
        renderer.close();sim.close_viewer()
        result.append({'label':row['label'],'image':str(path),'pose_source':str(source),'dynamics_steps':0,
                       'terrain_source':'measured PhysX height crop' if row.get('terrain_height') else terrain,
                       'interpretation':'Recorded PhysX pose on official RoboGauge mesh; visualization, not a MuJoCo rollout.'})
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    (out/'recovery_maps.json').write_text(json.dumps({'maps':maps},indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(render_records(a.records,a.output)))
if __name__=='__main__':main()
