"""Differentiable sampling of maps built from the actual PhysX collision mesh."""
import math
import torch
from . import state as st
SCAN_X=tuple(-.42+.12*i for i in range(11));SCAN_Y=tuple(-.36+.12*j for j in range(7));SCAN_POINTS=77

class HeightField:
    def __init__(self,heights,origin,res,valid=None):
        self.h,self.origin,self.res=heights,origin,float(res)
        self.valid=torch.ones_like(heights,dtype=torch.bool) if valid is None else valid.bool()
    def sample(self,xy,return_valid=False):
        gx=(xy[...,0]-self.origin[0])/self.res;gy=(xy[...,1]-self.origin[1])/self.res
        H,W=self.h.shape;inside=(gx>=0)&(gy>=0)&(gx<=W-1)&(gy<=H-1)&torch.isfinite(xy).all(-1)
        # Invalid predicted poses must reach the learner's nonfinite handling,
        # not poison CUDA with an out-of-bounds integer index first. The original
        # inside mask remains false; finite coordinates/derivatives are unchanged.
        gx=torch.nan_to_num(gx,nan=0.,posinf=0.,neginf=0.).clamp(0,W-1.001)
        gy=torch.nan_to_num(gy,nan=0.,posinf=0.,neginf=0.).clamp(0,H-1.001)
        x,y=gx.floor().long(),gy.floor().long();fx,fy=gx-x,gy-y
        value=(self.h[y,x]*(1-fx)+self.h[y,x+1]*fx)*(1-fy)+(self.h[y+1,x]*(1-fx)+self.h[y+1,x+1]*fx)*fy
        valid=inside&self.valid[y,x]&self.valid[y,x+1]&self.valid[y+1,x]&self.valid[y+1,x+1]
        # Coordinates never silently acquire another tile's edge height.
        value=torch.where(valid,value,torch.zeros_like(value))
        return (value,valid) if return_valid else value
    def teacher_map(self,pose,scene_ids=None):return geometry_crop(None,pose,self,'teacher')
    def local_map(self,pose,scene_ids=None):return geometry_crop(None,pose,self,'world_model')
    def scene_ids(self,pose):
        # One immutable collision mesh covers all arenas. No neural-renderer atlases.
        return torch.zeros(pose.shape[:-1],device=pose.device,dtype=torch.long)

def ground(xy,field):return torch.zeros_like(xy[...,0]) if field is None else field.sample(xy)
def _query_grid(pose,x,y):
    yaw=st.yaw_from_pose(pose);c,s=yaw[:,None,None].cos(),yaw[:,None,None].sin()
    return torch.stack((pose[:,None,None,0]+c*x-s*y,pose[:,None,None,1]+s*x+c*y),-1)
def relief(pose,field):
    y,x=torch.meshgrid(pose.new_tensor(SCAN_Y),pose.new_tensor(SCAN_X),indexing='ij')
    # Preserve source77 ordering x-major for official driver observation.
    xy=_query_grid(pose,x.T,y.T).flatten(1,2)
    return ground(pose[:,:2],field)[:,None]-ground(xy,field)
def geometry_crop(s,pose,field,kind='world_model'):
    if kind=='world_model':
        xs=torch.linspace(-1.2,1.2,49,device=pose.device,dtype=pose.dtype);ys=torch.linspace(-1.,1.,41,device=pose.device,dtype=pose.dtype)
    elif kind=='teacher':
        xs=torch.linspace(0.,6.,128,device=pose.device,dtype=pose.dtype);ys=torch.linspace(-6.,6.,64,device=pose.device,dtype=pose.dtype)
    else:raise ValueError(f'Unknown geometry crop {kind}')
    y,x=torch.meshgrid(ys,xs,indexing='ij');xy=_query_grid(pose,x,y)
    if field is None:height=torch.zeros_like(xy[...,0]);valid=torch.ones_like(height,dtype=torch.bool)
    else:height,valid=field.sample(xy,return_valid=True)
    return torch.stack((height-pose[:,None,None,2],valid.to(height.dtype)),1)
