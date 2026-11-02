"""Exact box-terrain asset adapter for official RoboGauge's MJCF loader."""
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

def export_task_maps(terrain_metadata,output):
    """Use the very same boxes used to construct PhysX collision meshes.

    This is asset conversion, not a second simulator. Original terrain tests
    continue to use the pinned RoboGauge terrain recipes.
    """
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True);records=[]
    task_for_terrain={7:1,10:5,11:7,12:8,13:9}
    all_boxes=[b for a in terrain_metadata['arenas'] for b in a['boxes']]
    all_boxes+=terrain_metadata.get('background_boxes',[])
    for arena in terrain_metadata['arenas']:
        tid=arena['terrain_id']
        if tid not in task_for_terrain:continue
        origin=np.asarray(arena['origin']);name=f"terrain{tid}_level{arena['level']}_layout{arena['layout']}"
        model=ET.Element('mujoco',model=name)
        visual=ET.SubElement(model,'visual');ET.SubElement(visual,'global',offwidth='1920',offheight='1080')
        world=ET.SubElement(model,'worldbody');ET.SubElement(world,'light',pos='0 0 5',dir='0 0 -1',directional='true')
        xmin,xmax,ymin,ymax=arena['bounds']
        boxes=[b for b in all_boxes if b[1][0]>=xmin-6 and b[0][0]<=xmax+6 and b[1][1]>=ymin-6 and b[0][1]<=ymax+6]
        for i,bounds in enumerate(boxes):
            bounds=np.asarray(bounds);center=bounds.mean(0)-origin;half=(bounds[1]-bounds[0])/2
            ET.SubElement(world,'geom',name=f'floor_box_{i}',type='box',pos=' '.join(map(str,center)),size=' '.join(map(str,half)),
                          friction='1 .005 .0001',rgba='.45 .5 .55 1')
        path=out/(name+'.xml');ET.indent(model);ET.ElementTree(model).write(path,encoding='unicode')
        records.append({'task_id':task_for_terrain[tid],'terrain_id':tid,'level':arena['level'],'layout':arena['layout'],
                        'xml':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'spawn':[0.,0.,0.],'goal':(np.asarray(arena['goal'])-origin).tolist(),
                        'source_origin':origin.tolist(),'length':arena['length'],'width':arena['width'],
                        'boxes':len(boxes),'conversion':'exact same axis-aligned collision boxes including6m neighboring/background context; coordinates translated by spawn origin'})
    result={'schema':'robogauge_depth_maps_v1','maps':records,
            'source':'loaded PhysX terrain metadata','original_terrain_source':'pinned official RoboGauge recipes'}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));return result

def validate_recovery_maps(manifest_path):
    """Select actual PhysX reset poses that also clear the official loaded XML.

    Never translate/rotate a pose to hide a mismatch. Keep every candidate and
    measured penetration in the manifest, failing if a pose category is absent.
    """
    import mujoco
    from .robogauge_bridge import verify_source_runtime
    from robogauge.tasks.simulator.mujoco_simulator import MujocoSimulator
    from robogauge.tasks.simulator.mujoco_config import MujocoConfig
    from robogauge.tasks.robots.go2.go2_lab_config import Go2Lab131Config
    from robogauge.utils.logger import logger
    path=Path(manifest_path);manifest=json.loads(path.read_text());runtime=verify_source_runtime()
    logger.create('asset_validation','recovery_poses',console_output=False,parent_log_dir=path.parent)
    config=MujocoConfig();config.viewer.headless=True;config.render.save_video=False
    robot=Go2Lab131Config();sim=MujocoSimulator(config);results=[]
    for terrain in sorted({m['terrain_id'] for m in manifest['maps'] if m['task_id']==6}):
        rows=[m for m in manifest['maps'] if m['task_id']==6 and m['terrain_id']==terrain]
        sim.load([rows[0]['xml']],robot.assets.robot_xml,[0,0,0],manifest['default_q'],False)
        model,data=sim.mj_model,sim.mj_data
        is_robot=np.array([(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(b)) or '').startswith('go2/') for b in model.geom_bodyid])
        for row in rows:
            measurements=[];selected=None
            for i,pose in enumerate(row['recovery_candidates']):
                data.qpos[:]=pose;data.qvel[:]=0;mujoco.mj_forward(model,data)
                depths=[max(0.,-float(c.dist)) for c in data.contact[:data.ncon] if is_robot[c.geom[0]]!=is_robot[c.geom[1]]]
                penetration=max(depths,default=0.);measurements.append(penetration)
                if selected is None and np.isfinite(data.qpos).all() and penetration<=.005:selected=i
            if selected is None:raise RuntimeError(f"No nonpenetrating official reset for {row['label']}: {measurements}")
            row['recovery_qpos']=row['recovery_candidates'][selected]
            row['recovery_validation']={'valid':True,'selected_candidate':selected,'terrain_penetration_m':measurements[selected],
                'candidate_penetrations_m':measurements,'maximum_allowed_m':.005,'pose_modified':False}
            results.append(dict(label=row['label'],terrain_id=terrain,**row['recovery_validation']))
    sim.close_viewer();sim.close_video_writer()
    manifest['recovery_runtime']=runtime;path.write_text(json.dumps(manifest,indent=2))
    (path.parent/'recovery_validation.json').write_text(json.dumps({'runtime':runtime,'rows':results},indent=2))
    return results

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--validate-recovery',required=True)
    a=p.parse_args();print(json.dumps(validate_recovery_maps(a.validate_recovery)))
