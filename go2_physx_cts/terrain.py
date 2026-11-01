"""Differentiable sampling of maps built from the actual PhysX collision mesh."""
import math
import torch
from . import state as st
from .camera_geometry import ATLAS_SPAN_M,ATLAS_STRIDE_M,frustum_world_corners
SCAN_X=tuple(-.42+.12*i for i in range(11));SCAN_Y=tuple(-.36+.12*j for j in range(7));SCAN_POINTS=77

class HeightField:
    def __init__(self,heights,origin,res,valid=None):
        self.h,self.origin,self.res=heights,origin,float(res)
        self.valid=torch.ones_like(heights,dtype=torch.bool) if valid is None else valid.bool()
        self._depth=None
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
    def depth_atlases(self):
        """Immutable25.6m scenes,12m apart, selected by the full camera frustum.

        Output sampling exactly matches the frozen generator's align-corners
        convention:512 samples spanning25.6m, not512*.05m corner rounding.
        """
        if self._depth is not None:return self._depth
        H,W=self.h.shape
        stride=ATLAS_STRIDE_M
        nx=max(1,math.ceil((W-1)*self.res/stride));ny=max(1,math.ceil((H-1)*self.res/stride))
        xs=self.h.new_tensor(self.origin[0])+torch.arange(nx,device=self.h.device)*stride+stride/2
        ys=self.h.new_tensor(self.origin[1])+torch.arange(ny,device=self.h.device)*stride+stride/2
        yy,xx=torch.meshgrid(ys,xs,indexing='ij');origins=torch.stack((xx.flatten(),yy.flatten(),torch.zeros_like(xx.flatten())),-1)
        axis=torch.linspace(-ATLAS_SPAN_M/2,ATLAS_SPAN_M/2,512,device=self.h.device);y,x=torch.meshgrid(axis,axis,indexing='ij')
        offset=torch.stack((x,y),-1);maps=[];keep=[]
        for first in range(0,len(origins),8):
            values,valid=self.sample(origins[first:first+8,None,None,:2]+offset,return_valid=True)
            included=valid.flatten(1).any(-1);keep.append(included);maps.append(values[included])
        included=torch.cat(keep);lookup=torch.full((len(origins),),-1,dtype=torch.long,device=self.h.device)
        lookup[included]=torch.arange(int(included.sum()),device=self.h.device);self._depth_lookup=lookup
        self._depth=(torch.cat(maps),origins[included]);self._depth_shape=(ny,nx)
        return self._depth
    def _scene_coverage(self,pose,corners,scene_ids,scene_exists=None):
        _,centers=self.depth_atlases()
        ids=scene_ids.reshape(pose.shape[:-1]).long()
        exists=(ids>=0)&(ids<len(centers))
        if scene_exists is not None:exists=exists&scene_exists
        # Clamping only makes indexing safe. It never changes the validity mask.
        c=centers[ids.clamp(0,len(centers)-1)]
        xy=corners[...,:2]
        atlas_margin=ATLAS_SPAN_M/2-(xy-c[...,None,:2]).abs().amax(dim=(-2,-1))
        lower,upper=xy.amin(-2),xy.amax(-2)
        field_min=pose.new_tensor(self.origin)
        H,W=self.h.shape;field_max=field_min+pose.new_tensor([(W-1)*self.res,(H-1)*self.res])
        field_margin=torch.minimum(lower-field_min,field_max-upper).amin(-1)
        finite=torch.isfinite(pose).all(-1)&(pose[...,3:7].norm(dim=-1)>1e-9)&torch.isfinite(corners).all(dim=(-2,-1))
        atlas=exists&finite&(atlas_margin>=0)
        field=finite&(field_margin>=0)
        return {'valid':atlas&field,'atlas_covered':atlas,'field_covered':field,
            'finite_pose':finite,'scene_exists':exists,'atlas_margin_m':atlas_margin,'field_margin_m':field_margin}
    def scene_selection(self,pose):
        """Return safe indices AND explicit coverage; index safety is not validity."""
        if self._depth is None:self.depth_atlases()
        corners=frustum_world_corners(pose)
        xy=corners[...,:2];centre=(xy.amin(-2)+xy.amax(-2))*.5
        ny,nx=self._depth_shape
        cell=torch.floor((centre-pose.new_tensor(self.origin))/ATLAS_STRIDE_M)
        within=torch.isfinite(cell).all(-1)&(cell>=0).all(-1)&(cell<cell.new_tensor([nx,ny])).all(-1)
        cell=torch.nan_to_num(cell,nan=0.,posinf=0.,neginf=0.)
        x=cell[...,0].clamp(0,nx-1).long();y=cell[...,1].clamp(0,ny-1).long()
        raw_ids=self._depth_lookup[y*nx+x]
        details=self._scene_coverage(pose,corners,raw_ids,within&(raw_ids>=0))
        return raw_ids.clamp_min(0),details
    def scene_ids(self,pose):
        # Existing Replay ABI. Consumers verify scene_coverage before rendering.
        return self.scene_selection(pose)[0]
    def scene_coverage(self,pose,scene_ids=None,return_details=False):
        """Strict frustum coverage in the supplied atlas and physical field bounds.

        This checks the complete projected frustum against rectangular bounds,
        not whether every interior map pixel contains a measured height hit.
        """
        if scene_ids is None:_,details=self.scene_selection(pose)
        else:details=self._scene_coverage(pose,frustum_world_corners(pose),scene_ids)
        return details if return_details else details['valid']

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
