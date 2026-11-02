"""Loaded PhysX terrain -> convex MJCF cells for official RoboGauge.

The existing source terrains are vertical solids (including annular tyres).
Extruding their original upward-facing triangles preserves the upper collision
surface and holes, unlike loading an entire concave terrain as one convex mesh.
No height resampling, mesh decimation or whole-tile convex hull is used.
"""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .task_allocation import TERRAIN_WEIGHTS

def export_surface(mesh,origin,path,forward_extent=14.):
    origin=np.asarray(origin);tri=np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    lo=tri.min(1);hi=tri.max(1)
    # Ten seconds at task1's final3m/s plus the6m teacher view. Keep
    # enough backward/lateral context without exporting the whole atlas.
    lower=np.array([-8.,-9.]);upper=np.array([forward_extent,9.])
    select=(normal[:,2]>1e-9)&(hi[:,:2]>=origin[:2]+lower).all(1)&(lo[:,:2]<=origin[:2]+upper).all(1)
    # Original connected mesh is used for rendering/rays only. Independent
    # prism eigenframes can cause floating-point ray misses on shared edges.
    # Collision uses the convex cells below; the concave visual mesh never
    # participates in contacts (so its convex hull cannot fill tyre holes).
    visible=(hi[:,:2]>=origin[:2]+lower).all(1)&(lo[:,:2]<=origin[:2]+upper).all(1)
    visual_tri=tri[visible]-origin
    tri=tri[select]-origin;area=normal[select,2]/2
    if not len(tri):raise ValueError('No source surface in transfer region')
    base=float(tri[:,:,2].min())-.2
    root=ET.Element('mujoco',model=path.stem);asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    vis=ET.SubElement(root,'visual');ET.SubElement(vis,'global',offwidth='1920',offheight='1080')
    ET.SubElement(world,'light',pos='0 0 5',dir='0 0 -1',directional='true')
    # Collapse connected coplanar convex patches (e.g. a long slope strip)
    # before creating contact cells. Area equality rejects holes/concavities.
    import trimesh
    from scipy.spatial import ConvexHull,QhullError
    surface=trimesh.Trimesh(vertices=tri.reshape(-1,3),faces=np.arange(len(tri)*3).reshape(-1,3),process=False)
    surface.merge_vertices()
    polygons=[];consumed=np.zeros(len(tri),dtype=bool)
    for group in surface.facets:
        if len(group)<4:continue
        points=np.unique(tri[group].reshape(-1,3),axis=0)
        try:hull=ConvexHull(points[:,:2])
        except QhullError:continue
        if abs(hull.volume-area[group].sum())>1e-7*max(1.,hull.volume):continue
        plane=np.linalg.lstsq(np.c_[points[:,:2],np.ones(len(points))],points[:,2],rcond=None)[0]
        if np.max(abs(np.c_[points[:,:2],np.ones(len(points))]@plane-points[:,2]))>1e-6:continue
        polygons.append(points[hull.vertices]);consumed[group]=True
    flat={};other=[]
    for i,t in enumerate(tri):
        if consumed[i]:continue
        if np.ptp(t[:,2])<1e-8:
            key=tuple(np.r_[t[:,:2].min(0),t[:,:2].max(0),t[0,2]])
            flat.setdefault(key,[]).append(i)
        else:other.append(i)
    geoms=0
    def fmt(x):return ' '.join(f'{v:.9g}' for v in np.asarray(x).flatten())
    vertices,inverse=np.unique(visual_tri.reshape(-1,3),axis=0,return_inverse=True)
    ET.SubElement(asset,'mesh',name='source_surface',vertex=fmt(vertices),face=' '.join(map(str,inverse)))
    ET.SubElement(world,'geom',name='source_surface',type='mesh',mesh='source_surface',contype='0',conaffinity='0',group='0')
    rectangles=[]
    for key,ids in flat.items():
        x0,y0,x1,y1,z=key;rectangle=(x1-x0)*(y1-y0)
        xy=tri[ids,:,:2]
        corners=(((xy[:,:,0]==x0)|(xy[:,:,0]==x1))&((xy[:,:,1]==y0)|(xy[:,:,1]==y1))).all()
        if corners and rectangle>0 and abs(float(area[ids].sum())-rectangle)<1e-7*max(1.,rectangle):
            rectangles.append(key)
        else:other.extend(ids)
    # Exact adjacent rectangle union only; never bridge a tyre hole or change
    # the height surface to reduce collision cost.
    for axis in (0,1,0,1):
        grouped={}
        for r in rectangles:
            key=(r[1-axis],r[3-axis],r[4]);grouped.setdefault(key,[]).append(r)
        merged=[]
        for values in grouped.values():
            values.sort(key=lambda r:r[axis]);current=list(values[0])
            for r in values[1:]:
                if current[axis+2]==r[axis]:current[axis+2]=r[axis+2]
                else:merged.append(tuple(current));current=list(r)
            merged.append(tuple(current))
        rectangles=merged
    for i,(x0,y0,x1,y1,z) in enumerate(rectangles):
        ET.SubElement(world,'geom',name=f'contact_rectangle_{i}',type='box',pos=fmt([(x0+x1)/2,(y0+y1)/2,(z+base)/2]),
                      size=fmt([(x1-x0)/2,(y1-y0)/2,(z-base)/2]),friction='1 .005 .0001',group='5');geoms+=1
    faces='0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0'
    for i in other:
        t=tri[i];bottom=t.copy();bottom[:,2]=base;name=f'cell_{i}'
        ET.SubElement(asset,'mesh',name=name,vertex=fmt(np.vstack((t,bottom))),face=faces)
        ET.SubElement(world,'geom',name='contact_'+name,type='mesh',mesh=name,friction='1 .005 .0001',group='5');geoms+=1
    for i,top in enumerate(polygons):
        n=len(top);bottom=top.copy();bottom[:,2]=base;faces=[]
        for j in range(1,n-1):faces.extend(((0,j,j+1),(n,n+j+1,n+j)))
        for j in range(n):
            k=(j+1)%n;faces.extend(((j,n+j,n+k),(j,n+k,k)))
        name=f'patch_{i}'
        ET.SubElement(asset,'mesh',name=name,vertex=fmt(np.vstack((top,bottom))),face=' '.join(map(str,np.asarray(faces).ravel())))
        ET.SubElement(world,'geom',name='contact_'+name,type='mesh',mesh=name,friction='1 .005 .0001',group='5');geoms+=1
    ET.indent(root);ET.ElementTree(root).write(path,encoding='unicode')
    # Retain source triangle probes for reproducible loaded-model validation.
    rng=np.random.default_rng(17);ids=rng.choice(len(tri),min(256,len(tri)),replace=False)
    x,y=np.meshgrid(np.linspace(-7.9,forward_extent-.1,48),np.linspace(-8.9,8.9,37));grid=np.stack((x.ravel(),y.ravel(),np.zeros(x.size)),-1)
    vertices=tri[ids,0];edge=np.concatenate([vertices+np.array([dx,dy,0.]) for dx,dy in ((.003,.0017),(-.003,-.0017))])
    probes=np.concatenate((tri[ids].mean(1),grid,edge))+np.array([.000137,.000071,0.])
    probes=probes[((probes[:,:2]>lower+.01)&(probes[:,:2]<upper-.01)).all(1)]
    np.savez_compressed(path.with_suffix('.npz'),probes=probes,source_triangles=tri[ids])
    return {'xml':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'geometry_probes':str(path.with_suffix('.npz')),'surface_triangles':len(tri),'collision_cells':geoms,
            'source_origin':origin.tolist(),'roi_xy':[lower.tolist(),upper.tolist()],
            'conversion':'original connected source mesh for visual/rays(group0, no contacts); upward triangles extruded into convex contact cells(group5); planar rectangles merged exactly'}

def export_allocation_maps(env,output):
    """Capture fixed maps and actual reset poses; restore environment and RNG."""
    from .physx_env import tensor
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    saved=copy.deepcopy(env.curriculum_state_dict());was_eval=env.evaluation_mode
    records=[];cache={};device=env.device
    devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
    try:
        env.evaluation_mode=True
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(76119)
            for task,weights in enumerate(TERRAIN_WEIGHTS):
                if task in (1,5,7,8,9):continue  # Existing exact box-map exporter.
                for terrain in weights:
                    level=5 if task==6 else 0
                    ids=env.ids[:min(128,env.num_envs)] if task==6 else env.ids[:1]
                    env.task_ids[ids]=task;env.terrain_ids[ids]=terrain;env.terrain_levels[ids]=level
                    env.reset(ids,advance_curriculum=False)
                    # A fixed column/layout is necessary for matched task cohorts.
                    index=int(ids[0]);origin=env.origins[index].cpu().numpy()
                    key=(terrain,level)
                    if key not in cache:
                        cache[key]=export_surface(env.scene.terrain.collision_mesh,origin,out/f'terrain{terrain}_level{level}.xml',forward_extent=38. if task==0 else 14.)
                        p=Path(cache[key]['geometry_probes']);probes=np.load(p)['probes']
                        starts=torch.as_tensor(probes+origin,device=device,dtype=torch.float32);starts[:,2]=20.
                        heights,valid=env.field.raycaster.cast(starts)
                        np.savez_compressed(p,probes=probes,expected_height=heights.cpu().numpy()-origin[2],valid=valid.cpu().numpy())
                    base=dict(cache[key],task_id=task,terrain_id=terrain,level=level,spawn=[0,0,0],
                              goal=(env.goals[index].cpu().numpy()-origin).tolist())
                    if task!=6:
                        records.append(dict(base,label=f'terrain{terrain}_level{level}'));continue
                    # Only poses on this map's actual source column are retained.
                    same=(env.origins[ids]-env.origins[index]).abs().amax(-1)<1e-5
                    for kind in range(6):
                        choices=ids[same&(env.recovery_kind[ids]==kind)]
                        if not len(choices):raise RuntimeError(f'Missing recovery kind{kind} terrain{terrain} in fixed source tile')
                        j=int(choices[0]);pose=tensor(env.robot.data.root_link_pose_w)[j].cpu().numpy()
                        q=tensor(env.robot.data.joint_pos)[j].cpu().numpy()
                        qpos=np.r_[pose[:3]-origin,pose[6],pose[3:6],q]
                        poses=tensor(env.robot.data.root_link_pose_w)[choices].cpu().numpy()
                        joints=tensor(env.robot.data.joint_pos)[choices].cpu().numpy()
                        candidates=np.c_[poses[:,:3]-origin,poses[:,6],poses[:,3:6],joints]
                        records.append(dict(base,label=f'kind{kind}_terrain{terrain}_level5',recovery_qpos=qpos.tolist(),
                                            recovery_candidates=candidates.tolist(),
                                            scope='actual PhysX reset state on matching loaded terrain; not a recovery capability claim'))
    finally:
        env.load_curriculum_state_dict(saved);env.evaluation_mode=True
        env.reset(env.ids,advance_curriculum=False);env._snapshot=env._observe();env.evaluation_mode=was_eval
    result={'schema':'allocation_maps_v1','maps':records,'source_mesh':env.field.metadata,
            'levels':'ordinary level0; recovery fixedlevel5','evaluation_only':True,'default_q':env.consts.default_q.cpu().tolist()}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));return result
