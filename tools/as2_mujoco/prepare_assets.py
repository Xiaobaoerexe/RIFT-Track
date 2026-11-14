"""Evaluation-only As2 assets and source-recipe terrain conversion."""
from pathlib import Path
import sys,json,hashlib,copy,struct,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=None
ITERATION=120000
sys.path.insert(0,str(ROOT))
def numbers(a):return ' '.join(format(float(x),'.10g') for x in np.asarray(a).reshape(-1))
def write(tree,path):ET.indent(tree);ET.ElementTree(tree).write(path,encoding='utf-8',xml_declaration=True)
def asset():
 meta=json.loads((OUT/'exports'/f'{ITERATION}_metadata.json').read_text());c=meta['consts']
 raw=Path(__file__).with_name('official_as2.xml');r=ET.parse(raw).getroot()
 r.find('compiler').set('meshdir',str(ROOT/'assets/as2/official/as2_description/meshes'))
 for m in r.findall('./asset/mesh'):m.attrib.pop('content_type',None)
 changes=[]
 for j in r.findall('./default/default/joint'):
  changes.append(dict(original=dict(j.attrib),new_damping=0,new_frictionloss=0))
  j.set('damping','0');j.set('frictionloss','0')
 names=[f'{leg}_{part}_joint' for leg in ('FL','FR','RL','RR') for part in ('hip','thigh','calf')]
 for body in r.findall('.//worldbody//body'):
  for j in body.findall('joint'):
   if j.get('type')=='free':j.tag='freejoint';j.attrib.pop('type')
   elif j.get('name') in names:
    i=names.index(j.get('name'));lim=c['effort'][i]
    j.set('actuatorfrcrange',numbers([-lim,lim]))
 base=r.find(".//body[@name='base_link']");base.set('pos','0 0 0.4')
 for leg in ('FL','FR','RL','RR'):
  calf=r.find(f".//body[@name='{leg}_calf']")
  foot=next(g for g in calf.findall('geom') if g.get('class')=='foot')
  foot.set('name',leg);foot.set('friction','1 .005 .0001');foot.set('condim','3')
  ET.SubElement(calf,'site',name=leg+'_foot_frame',pos=foot.get('pos'),size='.002')
 a=r.find('actuator');motors={x.get('joint'):x for x in list(a)}
 a[:]=[motors[n] for n in names]
 for i,x in enumerate(a):x.set('ctrlrange',numbers([-c['effort'][i],c['effort'][i]]))
 sensor=r.find('sensor');ordered=[]
 for tag in ('jointpos','jointvel','jointactuatorfrc'):
  items={x.get('joint'):x for x in sensor if x.tag==tag};ordered.extend(items[n] for n in names)
 ordered.extend(x for x in sensor if x.get('joint') is None);sensor[:]=ordered
 for geom in r.findall('.//worldbody//geom'):
  if geom.get('contype')!='0':geom.set('friction','1 .005 .0001')
 write(r,OUT/'as2_checkpoint.xml')
 mass=sum(float(x.get('mass')) for x in r.findall('.//inertial'))
 assert abs(mass-17.64)<1e-6,mass
 (OUT/'asset_adaptation.json').write_text(json.dumps({'source_snapshot_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
 'source':'175:/home/abc/Documents/VLN/unitree_ros-master/robots/as2_description/as2.xml',
 'source_note':'Existing official-derived As2 snapshot includes fixed-foot inertia merged into each 0.265kg calf.',
 'nominal_mass':mass,'joint_order':names,'joint_loss_changes':changes,
 'control':c,'other_changes':['sensor/motor reorder to match FL FR RL RR joint order','free joint tag for official attachment loader','named foot geoms and marker sites','mesh content_type omitted for MuJoCo 3.2.3','collision friction 1 matches training material','spawn body offset .4m','motor and joint force limits 50/50/75 from checkpoint'],
 'unmatched_physics':['MuJoCo has no equivalent hard joint velocity cap; measured exceedance is reported','contact solver differs from PhysX','physics dt .002s official RoboGauge versus .005s training']},indent=2))
def terrain(name,level,seed):
 from go2_physx_cts.official_terrain_mesh import MIXED_MESH_TERRAIN_CFG as C,_height_field_with_border,_stairs_mesh,_height_field_origin
 cfg=copy.deepcopy(C.sub_terrains[name]);cfg.size=C.size
 for k,v in [('horizontal_scale',C.horizontal_scale),('vertical_scale',C.vertical_scale),('slope_threshold',C.slope_threshold),('border_width',.5)]:
  if hasattr(cfg,k):setattr(cfg,k,v)
 d=(level+.5)/10;np.random.seed(seed)
 folder=OUT/'terrains'/f'{name}_l{level}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
 r=ET.Element('mujoco',model='terrain');vis=ET.SubElement(r,'visual');ET.SubElement(vis,'global',offwidth='960',offheight='540')
 ET.SubElement(vis,'headlight',ambient='.5 .5 .5',diffuse='.6 .6 .6')
 a=ET.SubElement(r,'asset');w=ET.SubElement(r,'worldbody');ET.SubElement(w,'light',pos='0 0 6',dir='-.2 -.3 -1',directional='true')
 n=0
 def box(lo,hi):
  nonlocal n
  lo,hi=np.array(lo),np.array(hi);assert np.all(hi>lo)
  ET.SubElement(w,'geom',name=f'terrain_{n}',type='box',pos=numbers((lo+hi)/2),size=numbers((hi-lo)/2),group='0',friction='1 .005 .0001',rgba='.5 .56 .6 1')
  n+=1
 def convex(vertices):
  nonlocal n
  key=f'part_{n}';ET.SubElement(a,'mesh',name=key,vertex=numbers(vertices))
  ET.SubElement(w,'geom',name=f'terrain_{n}',type='mesh',mesh=key,group='0',friction='1 .005 .0001',rgba='.5 .56 .6 1');n+=1
 origin=0.;method=''
 if name=='flat':
  ET.SubElement(w,'geom',name='floor',type='plane',size='0 0 .1',group='0',friction='1 .005 .0001',rgba='.5 .56 .6 1');method='infinite plane'
 elif name in ('wave','slope_up','slope_down','rough_slope'):
  h,size,hs,vs=_height_field_with_border(d,cfg,.2);z=h.astype(float)*vs
  origin=float(_height_field_origin(h,size,hs,vs)[2])
  low,span=float(z.min()),max(.001,float(z.max()-z.min()))
  image=np.asarray(((z-low)/span).T,dtype='<f4')
  binary=folder/'height.bin'
  binary.write_bytes(struct.pack('<ii',*image.shape)+image.tobytes())
  ET.SubElement(a,'hfield',name='height',file=str(binary),size=numbers([4.5,4.5,span,.1]))
  ET.SubElement(w,'geom',name='terrain_hfield',type='hfield',hfield='height',pos=numbers([0,0,low]),group='0',friction='1 .005 .0001')
  np.savez_compressed(folder/'height_source.npz',heights=z)
  method='source .2m smooth mesh height samples as MuJoCo hfield; triangulation checked separately'
 elif name=='stairs_up' or name=='stairs_down':
  mesh,o=_stairs_mesh(d,cfg);origin=float(o[2])
  for piece in mesh.split(only_watertight=False):
   assert piece.is_convex
   v=np.asarray(piece.vertices)-[4.5,4.5,0]
   if len(v)==8 and len(np.unique(v[:,0]))==2 and len(np.unique(v[:,1]))==2 and len(np.unique(v[:,2]))==2:box(v.min(0),v.max(0))
   else:convex(v)
  method='exact source disconnected convex stair solids'
 elif name=='obstacles':
  h,size,hs,vs=_height_field_with_border(d,cfg);origin=float(_height_field_origin(h,size,hs,vs)[2])
  from go2_physx_cts.official_terrain_mesh import _compression_indices
  rows,cols=_compression_indices(h,0),_compression_indices(h,1);z=h[np.ix_(rows,cols)]*vs;bottom=float(h.min()*vs-.1)
  for i in range(len(rows)-1):
   start=0
   for j in range(1,len(cols)):
    if j==len(cols)-1 or z[i,j]!=z[i,start]:
     box([rows[i]*hs-4.5,cols[start]*hs-4.5,bottom],[rows[i+1]*hs-4.5,cols[j]*hs-4.5,z[i,start]])
     start=j
  method='exact source compressed cell heights as box union'
 elif name=='small_squares':
  meshes,o=cfg.function(d,cfg);origin=float(o[2])
  for v in np.asarray(meshes[0].vertices).reshape(-1,8,3):
   v=v-[4.5,4.5,0];box(v.min(0),v.max(0))
  method='exact source square-column boxes'
 elif name=='wheels':
  meshes,o=cfg.function(d,cfg);origin=float(o[2])
  base=meshes[0].bounds-[4.5,4.5,0];box(*base)
  for mesh in meshes[1:]:
   v=np.asarray(mesh.vertices)-[4.5,4.5,0];center=(v.max(0)+v.min(0))/2
   rad=np.linalg.norm(v[:,:2]-center[:2],axis=1);r0,r1=rad.min(),rad.max()
   angles=np.unique(np.round(np.mod(np.arctan2(v[:,1]-center[1],v[:,0]-center[0]),2*np.pi),10))
   for i,p in enumerate(angles):
    q=angles[(i+1)%len(angles)];verts=[]
    for z in (v[:,2].min(),v[:,2].max()):
     for r_ in (r0,r1):
      for ang in (p,q):verts.append([center[0]+r_*np.cos(ang),center[1]+r_*np.sin(ang),z])
    convex(verts)
  method='exact annulus polygon sectors as convex prisms; holes preserved'
 else:raise ValueError(name)
 write(r,folder/'terrain.xml')
 info={'name':name,'level':level,'difficulty':d,'seed':seed,'spawn':[0,0,origin],'path':str(folder/'terrain.xml'),'method':method,'tile_size':[9,9],'course_end_x':3.5,'geom_count':n}
 (folder/'terrain.json').write_text(json.dumps(info,indent=2));return info
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--iteration',type=int,default=120000);a=p.parse_args()
 OUT=Path(a.output).expanduser().resolve();OUT.mkdir(parents=True,exist_ok=True);ITERATION=a.iteration
 asset();items=[]
 for name in ('flat','wave','slope_up','slope_down','rough_slope','stairs_up','stairs_down','obstacles','wheels','small_squares'):
  for level in ((0,) if name=='flat' else (2,5)):
   for seed in (17,29,43):
    info=terrain(name,level,seed);items.append(info);print(json.dumps(info),flush=True)
 (OUT/'terrain_manifest.json').write_text(json.dumps(items,indent=2))
