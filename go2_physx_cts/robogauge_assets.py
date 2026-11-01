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
