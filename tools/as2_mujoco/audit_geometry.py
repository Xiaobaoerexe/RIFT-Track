"""Independent loaded-MJCF / training-URDF kinematics and inertia check."""
from pathlib import Path
import json,xml.etree.ElementTree as E,math
import numpy as np,mujoco
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
P=Path(args.output).expanduser().resolve()
R=Path(__file__).resolve().parents[2]
m=mujoco.MjModel.from_xml_path(str(P/'as2_checkpoint.xml'));d=mujoco.MjData(m)
u=E.parse(R/'assets/as2/official/as2_description/urdf/as2.urdf').getroot()
def vec(s,default='0 0 0'):return np.fromstring(s or default,sep=' ')
def rot(axis,a):
 axis=np.array(axis);axis=axis/np.linalg.norm(axis);x,y,z=axis
 K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
 return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)
def origin(x):
 T=np.eye(4)
 if x is not None:
  T[:3,3]=vec(x.get('xyz'));r,p,y=vec(x.get('rpy'))
  T[:3,:3]=rot([0,0,1],y)@rot([0,1,0],p)@rot([1,0,0],r)
 return T
links={x.get('name'):x for x in u.findall('link')};joints=list(u.findall('joint'))
children={j.find('child').get('link') for j in joints};roots=set(links)-children
rng=np.random.default_rng(17);errors=[]
for trial in range(12):
 d.qpos[:3]=[0,0,.4];d.qpos[3:7]=[1,0,0,0]
 q=np.array([-.1,.9,-1.8,.1,.9,-1.8,-.1,.9,-1.8,.1,.9,-1.8])+rng.uniform(-.1,.1,12)
 d.qpos[7:]=q;mujoco.mj_forward(m,d)
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,m.njnt)];angles=dict(zip(names,q))
 Ts={k:np.eye(4) for k in roots}
 for t in Ts.values():t[:3,3]=d.qpos[:3]
 todo=joints[:]
 while todo:
  nexts=[]
  for j in todo:
   a,b=j.find('parent').get('link'),j.find('child').get('link')
   if a not in Ts:nexts.append(j);continue
   T=origin(j.find('origin'))
   if j.get('type') in ('revolute','continuous'):
    Q=np.eye(4);Q[:3,:3]=rot(vec(j.find('axis').get('xyz')),angles[j.get('name')]);T=T@Q
   Ts[b]=Ts[a]@T
  if len(nexts)==len(todo):raise ValueError('Unresolved URDF chain')
  todo=nexts
 for leg in ('FL','FR','RL','RR'):
  for part in ('thigh','calf','foot'):
   name=leg+'_'+part
   actual=d.site_xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,leg+'_foot_frame')] if part=='foot' else d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)]
   errors.append(float(np.max(np.abs(actual-Ts[name][:3,3]))))
def inertial(link):
 x=links[link].find('inertial');mass=float(x.find('mass').get('value'));T=origin(x.find('origin'));a=x.find('inertia').attrib
 I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],[float(a['ixy']),float(a['iyy']),float(a['iyz'])],[float(a['ixz']),float(a['iyz']),float(a['izz'])]])
 return mass,T[:3,3],T[:3,:3]@I@T[:3,:3].T
inertia_err=[];com_err=[];mass_err=[]
for i in range(1,m.nbody):
 name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i);link='base' if name=='base_link' and 'base' in links else name
 mass,c,I=inertial(link)
 if name.endswith('_calf'):
  f=name.replace('_calf','_foot');fm,fc,FI=inertial(f)
  j=next(j for j in joints if j.find('child').get('link')==f);T=origin(j.find('origin'));fc=T[:3,3]+T[:3,:3]@fc;FI=T[:3,:3]@FI@T[:3,:3].T
  cnew=(mass*c+fm*fc)/(mass+fm)
  shift=lambda r:(np.dot(r,r)*np.eye(3)-np.outer(r,r))
  I=I+mass*shift(c-cnew)+FI+fm*shift(fc-cnew);mass+=fm;c=cnew
 Q=np.zeros(9);mujoco.mju_quat2Mat(Q,m.body_iquat[i]);Q=Q.reshape(3,3);actual=Q@np.diag(m.body_inertia[i])@Q.T
 inertia_err.append(float(np.max(np.abs(actual-I))));com_err.append(float(np.max(np.abs(c-m.body_ipos[i]))));mass_err.append(abs(mass-m.body_mass[i]))
result={'fk_random_poses':12,'max_landmark_error_m':max(errors),'max_mass_error_kg':max(mass_err),'max_com_error_m':max(com_err),'max_inertia_error_kg_m2':max(inertia_err),'nominal_mass':float(m.body_mass.sum())}
assert result['max_landmark_error_m']<1e-5,result
assert result['max_mass_error_kg']<1e-6 and result['max_com_error_m']<1e-6 and result['max_inertia_error_kg_m2']<2e-6,result
(P/'robot_geometry_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
# Check source diagonal convention against the loaded smooth heightfields.
results=[]
for row in json.loads((P/'terrain_manifest.json').read_text()):
 if row['name'] not in ('wave','slope_up','slope_down','rough_slope'):continue
 p=Path(row['path']);hm=mujoco.MjModel.from_xml_path(str(p));hd=mujoco.MjData(hm);mujoco.mj_forward(hm,hd)
 z=np.load(p.parent/'height_source.npz')['heights'];coords=rng.uniform(-4.3,4.3,(200,2));errs=[]
 for x,y in coords:
  gx=(x+4.5)/.2;gy=(y+4.5)/.2;i,j=int(gx),int(gy);fx,fy=gx-i,gy-j
  expected=z[i,j]+(z[i+1,j]-z[i,j])*fx+(z[i+1,j+1]-z[i+1,j])*fy if fx>=fy else z[i,j]+(z[i+1,j+1]-z[i,j+1])*fx+(z[i,j+1]-z[i,j])*fy
  dist=mujoco.mj_ray(hm,hd,np.array([x,y,20.]),np.array([0.,0.,-1.]),None,1,-1,np.zeros(1,np.int32))
  errs.append(abs((20-dist)-expected))
 results.append({'terrain':row['name'],'level':row['level'],'seed':row['seed'],'max_ray_error_m':max(errs),'mean_ray_error_m':float(np.mean(errs))})
(P/'heightfield_verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
