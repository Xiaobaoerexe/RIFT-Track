"""Pinned Go2RLLab direct mesh, source distance progression, authoritative raycasts.
Source wty-yy/go2_rl_lab@6c95d9feef4814c25eb6536e61bb38e1d1d718d4.
"""
import hashlib
import numpy as np
import torch
import warp as wp
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.sim import RigidBodyMaterialCfg
from .terrain import HeightField
from .official_terrain_mesh import MIXED_MESH_TERRAIN_CFG

class CapturedTerrain(TerrainImporter):
    def __init__(self,cfg,*args,**kwargs):
        # Pinned custom wheel/square/height functions use global NumPy RNG.
        # Seed only the generation scope and preserve unrelated process RNG.
        previous=np.random.get_state()
        try:
            np.random.seed(int(cfg.terrain_generator.seed))
            super().__init__(cfg,*args,**kwargs)
        finally:
            np.random.set_state(previous)
    def import_mesh(self,name,mesh):
        self.collision_mesh=mesh.copy()
        return super().import_mesh(name,mesh)

def terrain_config(seed=17):
    gen=MIXED_MESH_TERRAIN_CFG.copy();gen.seed=int(seed)
    return TerrainImporterCfg(class_type=CapturedTerrain,prim_path='/World/Ground',
        terrain_type='generator',terrain_generator=gen,max_init_terrain_level=5,
        collision_group=-1,debug_vis=False,
        physics_material=RigidBodyMaterialCfg(friction_combine_mode='average',
            restitution_combine_mode='average',static_friction=1.,dynamic_friction=1.,restitution=0.))

class TerrainPositionLookup:
    """Source-normalized columns and half-open world-space cell lookup."""
    def __init__(self,cfg,device):
        self.names=tuple(cfg.sub_terrains)
        raw=np.asarray([value.proportion for value in cfg.sub_terrains.values()],dtype=np.float64)
        if not np.isfinite(raw).all() or (raw<0).any() or raw.sum()<=0:
            raise ValueError('Invalid source terrain proportions')
        normalized=raw/raw.sum();columns=np.arange(cfg.num_cols)/cfg.num_cols+.001
        ids=np.searchsorted(np.cumsum(normalized),columns,side='right')
        self.column_types=torch.as_tensor(ids,dtype=torch.long,device=device)
        self.size=torch.tensor(cfg.size,dtype=torch.float32,device=device)
        self.shape=torch.tensor([cfg.num_rows,cfg.num_cols],device=device)
        self.minimum_xy=-.5*self.size*self.shape
        self.metadata={'names':list(self.names),'raw_proportions':raw.tolist(),
            'raw_proportion_sum':float(raw.sum()),'normalized_proportions':normalized.tolist(),
            'column_type_ids':ids.tolist(),'column_counts':np.bincount(ids,minlength=len(raw)).tolist(),
            'rows':cfg.num_rows,'columns':cfg.num_cols,'tile_size_m':list(cfg.size),
            'border_width_m':cfg.border_width,'sub_terrain_border_width_m':cfg.sub_terrain_border_width,
            'max_init_terrain_level':5,'default_geometry_backend':'source_direct_mesh','seed':cfg.seed,
            'numpy_geometry_seed_scope':'saved_seeded_generated_restored'}
    def query(self,positions_w):
        cells=torch.floor((positions_w[...,:2]-self.minimum_xy)/self.size).long()
        inside=((cells>=0)&(cells<self.shape)).all(-1)&torch.isfinite(positions_w[...,:2]).all(-1)
        columns=cells[...,1].clamp(0,len(self.column_types)-1)
        return torch.where(inside,self.column_types[columns],-1)
    __call__=query

def terrain_level_decisions(maximum_distance,accumulated_commands,resampling_time,
                            zero_command_probability,tile_length=9.,sub_terrain_border_width=.5):
    """Exact TerrainLevelsVelGym inequalities; no invented iteration level ramp."""
    length=max(0.,tile_length-2.*sub_terrain_border_width)
    move_up=maximum_distance>length/2.
    target_distance=accumulated_commands.norm(dim=1)*resampling_time*(1.-zero_command_probability)
    move_down=(maximum_distance<target_distance*.5)&~move_up
    return move_up,move_down

@wp.kernel
def cast_height(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),height:wp.array(dtype=float),valid:wp.array(dtype=wp.int32)):
    i=wp.tid();query=wp.mesh_query_ray(mesh,points[i],wp.vec3(0.,0.,-1.),100.)
    if query.result:
        height[i]=points[i][2]-query.t
        valid[i]=1

class CollisionHeightRaycaster:
    def __init__(self,mesh,device):
        self.device=device
        vertices=np.ascontiguousarray(mesh.vertices,dtype=np.float32)
        faces=np.ascontiguousarray(mesh.faces,dtype=np.int32)
        self.mesh=wp.Mesh(points=wp.array(vertices,dtype=wp.vec3,device=device),
                          indices=wp.array(faces.flatten(),dtype=int,device=device))
        self.identity={'vertices_sha256':hashlib.sha256(vertices.tobytes()).hexdigest(),
                       'faces_sha256':hashlib.sha256(faces.tobytes()).hexdigest(),
                       'vertex_count':len(vertices),'triangle_count':len(faces)}
    def cast(self,starts):
        shape=starts.shape[:-1];flat=starts.reshape(-1,3).contiguous().to(dtype=torch.float32)
        heights=torch.zeros(flat.shape[0],device=flat.device)
        valid=torch.zeros(flat.shape[0],dtype=torch.int32,device=flat.device)
        wp.launch(cast_height,dim=flat.shape[0],inputs=[self.mesh.id,wp.from_torch(flat,dtype=wp.vec3),
            wp.from_torch(heights),wp.from_torch(valid)],device=self.device)
        return heights.reshape(shape),valid.reshape(shape).bool()
    def height_crop(self,positions,yaw):
        """Exact full17x11 scan, source finite fallback, then5x3 crop."""
        x,y=torch.meshgrid(torch.arange(-8,9,device=positions.device)*.1,
                           torch.arange(-5,6,device=positions.device)*.1,indexing='ij')
        x,y=x.flatten(),y.flatten();c,s=yaw[:,None].cos(),yaw[:,None].sin()
        starts=torch.stack((positions[:,0:1]+c*x-s*y,positions[:,1:2]+s*x+c*y,
                            (positions[:,2:3]+20.).expand(-1,len(x))),-1)
        heights,valid=self.cast(starts)
        fallback=torch.where(valid,heights,torch.full_like(heights,torch.inf)).amin(-1)
        fallback=torch.where(torch.isfinite(fallback),fallback,positions[:,2]+19.)
        heights=torch.where(valid,heights,fallback[:,None])
        crop=(x.abs()<=.200001)&(y.abs()<=.150001)
        return positions[:,2]-heights[:,crop].mean(-1),valid[:,crop].all(-1)

def height_grid_axis(lower,upper,res):
    """Avoid float32 arange spacing drift over the source140x230m world."""
    count=int(np.ceil((upper-lower)/res))+1
    return (float(lower)+np.arange(count,dtype=np.float64)*res).astype(np.float32)

def field_from_mesh(terrain,device,res=.05):
    mesh=terrain.collision_mesh;lo,hi=mesh.bounds
    raycaster=CollisionHeightRaycaster(mesh,device)
    # Build coordinates in float64 before conversion: float32 arange starting at
    # -115m can accumulate centimetres of spacing error across the large grid.
    xs=height_grid_axis(lo[0],hi[0],res);ys=height_grid_axis(lo[1],hi[1],res)
    xs_t=torch.as_tensor(xs,device=device);ys_t=torch.as_tensor(ys,device=device)
    heights=torch.empty((len(ys)*len(xs),),device=device);misses=0
    # Bounded batches avoid a complete140x230m ray-start grid allocation on GPU.
    for first in range(0,heights.numel(),262144):
        index=torch.arange(first,min(first+262144,heights.numel()),device=device)
        x=xs_t[index.remainder(len(xs))];y=ys_t[torch.div(index,len(xs),rounding_mode='floor')]
        starts=torch.stack((x,y,torch.full_like(x,float(hi[2]+2.))),-1)
        h,valid=raycaster.cast(starts)
        interior=(x>lo[0]+res)&(x<hi[0]-res)&(y>lo[1]+res)&(y<hi[1]-res)
        if bool((~valid&interior).any()):raise ValueError('Interior ray misses in authoritative collision mesh')
        misses+=int((~valid).sum());heights[first:first+len(index)]=torch.where(valid,h,0.)
    field=HeightField(heights.reshape(len(ys),len(xs)),(float(lo[0]),float(lo[1])),res)
    field.raycaster=raycaster
    field.metadata=dict(raycaster.identity,resolution_m=res,shape=list(field.h.shape),
                        outer_boundary_misses=misses,collision_bounds=mesh.bounds.tolist())
    return field
