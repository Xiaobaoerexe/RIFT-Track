"""Pinned source ten terrains plus the three approved event geometries.

The original mesh recipes remain unchanged. Additional collision meshes and
metadata are built together so physics, depth and dynamics observe one geometry.
"""
import hashlib
import numpy as np
import torch
import warp as wp
import trimesh
from isaaclab.terrains import TerrainImporter,TerrainImporterCfg
from isaaclab.sim import RigidBodyMaterialCfg
from .terrain import HeightField
from .official_terrain_mesh import MIXED_MESH_TERRAIN_CFG
from .commands import TERRAIN_NAMES

def runway_length(level,episode_seconds=20.,speed=8.,acceleration=2.):
    """60% reachability after acceleration; harder levels require faster progress."""
    available=max(.5,episode_seconds-1.);ramp=min(available,speed/acceleration)
    travel=.5*acceleration*ramp*ramp+speed*max(0.,available-ramp)
    return max(12.,(.30+.65*level/9)*travel/.6)

def _box(x0,x1,y0,y1,z0,z1):
    return trimesh.creation.box(extents=(x1-x0,y1-y0,z1-z0),transform=trimesh.transformations.translation_matrix(((x0+x1)/2,(y0+y1)/2,(z0+z1)/2)))

def additional_arenas(bounds,seed,episode_seconds=20.,task_episode_seconds=None):
    meshes=[];arenas=[];base_x=float(bounds[1,0])+4.
    durations=[20.,20.,20.,8.] if task_episode_seconds is None else task_episode_seconds
    column_span=12.+2.*durations[3]+7.
    def add(kind,level,layout,x,y,length,width,parts,spawn,goal):
        offset=np.array([x,y,0.]);world=[]
        for m in parts:m.apply_translation(offset);world.append(m)
        meshes.extend(world)
        arenas.append(dict(terrain_id=TERRAIN_NAMES.index(kind),level=level,layout=layout,
            bounds=[x,max(m.bounds[1,0] for m in world),y,y+width],origin=(offset+np.array(spawn)).tolist(),
            goal=(offset+np.array(goal)).tolist(),length=length,width=width,boxes=[m.bounds.tolist() for m in world]))
    for level in range(10):
        L,W=12.,12.;tail=2.*durations[3]+3.;width=.20+.07*level;far=L*.5;near=far-width
        parts=[_box(0,L,0,W,-1.,-.8),_box(0,near,0,W,-.8,0),_box(far,L+tail,0,W,-.8,0)]
        add('gap',level,0,base_x,90.+14*level,L,W,parts,(near-1.4,W/2,0.),(L*.7,W/2,0.))
        h=.25+.03*level;edge=3.;W=24.;tail=2.*durations[3]+3.
        parts=[_box(0,L+tail,0,W,-.2,0),_box(edge,L+tail,0,W,0,h)]
        add('obstacle_up',level,0,base_x+column_span,90.+28*level,L,W,parts,(edge-1.5,W/2,0.),(edge+1.1,W/2,h))
        tail=2.*durations[3]+3.
        parts=[_box(0,L+tail,0,W,-h-.2,-h),_box(0,edge,0,W,-h,0.)]
        add('obstacle_down',level,0,base_x+2*column_span,90.+28*level,L,W,parts,(edge-1.3,W/2,0.),(edge+1.1,W/2,-h))
        L=runway_length(level,episode_seconds);W=16.
        add('flat',level,100,base_x,-100.-18*level,L,W,[_box(0,L,0,W,-.2,0)],(1.,W/2,0.),(.8*L,W/2,0.))
    return meshes,arenas

def surrounding_floor(source_bounds,arenas,margin=25.):
    """Extend the source ground border around arenas, with exact rectangular cuts.

    Uncovered atlas pixels are zero-height ground in the frozen depth contract;
    matching physical slabs prevent those pixels representing nonexistent floor.
    Cuts preserve every gap/drop, including negative-height floors.
    """
    boxes=[(float(source_bounds[0,0]),float(source_bounds[1,0]),float(source_bounds[0,1]),float(source_bounds[1,1]))]+[tuple(a['bounds']) for a in arenas]
    xs=sorted({v for b in boxes for v in b[:2]}|{min(b[0] for b in boxes)-margin,max(b[1] for b in boxes)+margin})
    ys=sorted({v for b in boxes for v in b[2:]}|{min(b[2] for b in boxes)-margin,max(b[3] for b in boxes)+margin})
    pending={};rects=[]
    for iy in range(len(ys)-1):
        cy=(ys[iy]+ys[iy+1])/2;active=[]
        for ix in range(len(xs)-1):
            cx=(xs[ix]+xs[ix+1])/2;active.append(not any(b[0]<=cx<=b[1] and b[2]<=cy<=b[3] for b in boxes))
        runs=[];start=None
        for ix,on in enumerate(active+[False]):
            if on and start is None:start=ix
            if not on and start is not None:runs.append((start,ix));start=None
        new={}
        for run in runs:
            if run in pending:new[run]=pending.pop(run)
            else:new[run]=ys[iy]
        for (left,right),y0 in pending.items():rects.append((xs[left],xs[right],y0,ys[iy]))
        pending=new
    for (left,right),y0 in pending.items():rects.append((xs[left],xs[right],y0,ys[-1]))
    return [_box(x0,x1,y0,y1,-.2,0.) for x0,x1,y0,y1 in rects]

class CapturedTerrain(TerrainImporter):
    def __init__(self,cfg,*args,**kwargs):
        previous=np.random.get_state()
        try:np.random.seed(int(cfg.terrain_generator.seed));super().__init__(cfg,*args,**kwargs)
        finally:np.random.set_state(previous)
    def import_mesh(self,name,mesh):
        extra,self.arenas=additional_arenas(mesh.bounds,int(self.cfg.terrain_generator.seed),getattr(self.cfg.terrain_generator,'runway_episode_seconds',20.),getattr(self.cfg.terrain_generator,'task_episode_seconds',None))
        background=surrounding_floor(mesh.bounds,self.arenas);self.background_boxes=[m.bounds.tolist() for m in background]
        mesh=trimesh.util.concatenate([mesh,*extra,*background]);self.collision_mesh=mesh.copy()
        return super().import_mesh(name,mesh)

def terrain_config(seed=17,runway_episode_seconds=20.,task_episode_seconds=None):
    gen=MIXED_MESH_TERRAIN_CFG.copy();gen.seed=int(seed);gen.runway_episode_seconds=float(runway_episode_seconds);gen.task_episode_seconds=task_episode_seconds
    return TerrainImporterCfg(class_type=CapturedTerrain,prim_path='/World/Ground',terrain_type='generator',terrain_generator=gen,max_init_terrain_level=5,collision_group=-1,debug_vis=False,
        physics_material=RigidBodyMaterialCfg(friction_combine_mode='average',restitution_combine_mode='average',static_friction=1.,dynamic_friction=1.,restitution=0.))

class TerrainPositionLookup:
    def __init__(self,cfg,device,arenas=()):
        self.names=TERRAIN_NAMES;raw=np.asarray([v.proportion for v in cfg.sub_terrains.values()]);normalized=raw/raw.sum()
        ids=np.searchsorted(np.cumsum(normalized),np.arange(cfg.num_cols)/cfg.num_cols+.001,side='right')
        self.column_types=torch.as_tensor(ids,dtype=torch.long,device=device);self.size=torch.tensor(cfg.size,device=device,dtype=torch.float32);self.shape=torch.tensor([cfg.num_rows,cfg.num_cols],device=device)
        self.minimum_xy=-.5*self.size*self.shape;self.arenas=list(arenas)
        self.extra_bounds=torch.tensor([a['bounds'] for a in arenas],device=device,dtype=torch.float32) if arenas else torch.empty(0,4,device=device)
        self.extra_types=torch.tensor([a['terrain_id'] for a in arenas],device=device,dtype=torch.long)
        self.metadata={'names':list(self.names),'source_names':list(cfg.sub_terrains),'rows':cfg.num_rows,'columns':cfg.num_cols,'tile_size_m':list(cfg.size),'seed':cfg.seed,'max_init_terrain_level':5,'arenas':self.arenas}
    def query(self,p):
        shape=p.shape[:-1];xy=p[...,:2].reshape(-1,2);cells=torch.floor((xy-self.minimum_xy)/self.size).long()
        inside=((cells>=0)&(cells<self.shape)).all(-1);cols=cells[:,1].clamp(0,len(self.column_types)-1)
        result=torch.where(inside,self.column_types[cols],7)
        # Forty indexed rectangles, with no obstacle-avoidance arenas.
        if len(self.arenas):
            b=self.extra_bounds;hit=(xy[:,None,0]>=b[None,:,0])&(xy[:,None,0]<b[None,:,1])&(xy[:,None,1]>=b[None,:,2])&(xy[:,None,1]<b[None,:,3])
            idx=hit.long().argmax(-1);result=torch.where(hit.any(-1),self.extra_types[idx],result)
        return result.reshape(shape)
    __call__=query

def terrain_level_decisions(maximum_distance,accumulated_commands,resampling_time,
                            zero_command_probability,tile_length=9.,sub_terrain_border_width=.5,
                            *,success=None,task_ids=None):
    """Pinned AS2 progress rules; crossing tasks use hard success."""
    length=torch.as_tensor(tile_length,device=maximum_distance.device,dtype=maximum_distance.dtype)-2.*sub_terrain_border_width
    up=maximum_distance>length.clamp_min(0.)/2.
    target=accumulated_commands.norm(dim=1)*resampling_time*(1.-zero_command_probability)
    down=(maximum_distance<target*.5)&~up
    if task_ids is not None:
        if success is not None:
            up=torch.where(task_ids==3,success.bool(),up);down=torch.where(task_ids==3,~success.bool(),down)
    return up,down

@wp.kernel
def cast_height(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),height:wp.array(dtype=float),valid:wp.array(dtype=wp.int32)):
    i=wp.tid();query=wp.mesh_query_ray(mesh,points[i],wp.vec3(0.,0.,-1.),100.)
    if query.result:height[i]=points[i][2]-query.t;valid[i]=1

class CollisionHeightRaycaster:
    def __init__(self,mesh,device):
        self.device=device;vertices=np.ascontiguousarray(mesh.vertices,dtype=np.float32);faces=np.ascontiguousarray(mesh.faces,dtype=np.int32)
        self.mesh=wp.Mesh(points=wp.array(vertices,dtype=wp.vec3,device=device),indices=wp.array(faces.flatten(),dtype=int,device=device))
        self.identity={'vertices_sha256':hashlib.sha256(vertices.tobytes()).hexdigest(),'faces_sha256':hashlib.sha256(faces.tobytes()).hexdigest(),'vertex_count':len(vertices),'triangle_count':len(faces)}
    def cast(self,starts):
        shape=starts.shape[:-1];flat=starts.reshape(-1,3).contiguous().float();height=torch.zeros(len(flat),device=flat.device);valid=torch.zeros(len(flat),dtype=torch.int32,device=flat.device)
        wp.launch(cast_height,dim=len(flat),inputs=[self.mesh.id,wp.from_torch(flat,dtype=wp.vec3),wp.from_torch(height),wp.from_torch(valid)],device=self.device)
        return height.reshape(shape),valid.reshape(shape).bool()
    def height_crop(self,positions,yaw):
        x,y=torch.meshgrid(torch.arange(-8,9,device=positions.device)*.1,torch.arange(-5,6,device=positions.device)*.1,indexing='ij')
        x,y=x.flatten(),y.flatten();c,s=yaw[:,None].cos(),yaw[:,None].sin()
        starts=torch.stack((positions[:,0:1]+c*x-s*y,positions[:,1:2]+s*x+c*y,(positions[:,2:3]+20.).expand(-1,len(x))),-1)
        heights,valid=self.cast(starts)
        fallback=torch.where(valid,heights,torch.full_like(heights,torch.inf)).amin(-1)
        fallback=torch.where(torch.isfinite(fallback),fallback,positions[:,2]+19.)
        heights=torch.where(valid,heights,fallback[:,None]);crop=(x.abs()<=.200001)&(y.abs()<=.150001)
        return positions[:,2]-heights[:,crop].mean(-1),valid[:,crop].all(-1)

def height_grid_axis(lower,upper,res):return (float(lower)+np.arange(int(np.ceil((upper-lower)/res))+1,dtype=np.float64)*res).astype(np.float32)
def field_from_mesh(terrain,device,res=.05):
    mesh=terrain.collision_mesh;lo,hi=mesh.bounds;raycaster=CollisionHeightRaycaster(mesh,device)
    xs=height_grid_axis(lo[0],hi[0],res);ys=height_grid_axis(lo[1],hi[1],res);xs_t=torch.as_tensor(xs,device=device);ys_t=torch.as_tensor(ys,device=device)
    heights=torch.empty(len(ys)*len(xs),device=device);validity=torch.empty_like(heights,dtype=torch.bool)
    for first in range(0,heights.numel(),262144):
        index=torch.arange(first,min(first+262144,heights.numel()),device=device);x=xs_t[index.remainder(len(xs))];y=ys_t[torch.div(index,len(xs),rounding_mode='floor')]
        h,valid=raycaster.cast(torch.stack((x,y,torch.full_like(x,float(hi[2]+2.))),-1));heights[first:first+len(index)]=h;validity[first:first+len(index)]=valid
    field=HeightField(heights.reshape(len(ys),len(xs)),(float(lo[0]),float(lo[1])),res,validity.reshape(len(ys),len(xs)));field.raycaster=raycaster
    field.metadata=dict(raycaster.identity,resolution_m=res,shape=list(field.h.shape),ray_misses=int((~validity).sum()),collision_bounds=mesh.bounds.tolist(),invalid_height_policy='validity_channel; no edge-clamping')
    return field
