"""Pinned source height-map sampling; PhysX mesh is the authoritative geometry."""
import torch
SCAN_X=tuple(-.42+.12*i for i in range(11))
SCAN_Y=tuple(-.36+.12*j for j in range(7))
SCAN_POINTS=77

class HeightField:
    def __init__(self,heights,origin,res):self.h,self.origin,self.res=heights,origin,res
    def sample(self,xy):
        gx=(xy[...,0]-self.origin[0])/self.res;gy=(xy[...,1]-self.origin[1])/self.res
        H,W=self.h.shape;gx=gx.clamp(0,W-1.001);gy=gy.clamp(0,H-1.001)
        x,y=gx.floor().long(),gy.floor().long();fx,fy=gx-x,gy-y
        return (self.h[y,x]*(1-fx)+self.h[y,x+1]*fx)*(1-fy)+(self.h[y+1,x]*(1-fx)+self.h[y+1,x+1]*fx)*fy

def ground(xy,field):return torch.zeros_like(xy[...,0]) if field is None else field.sample(xy)

def relief(pose,field):
    xs,ys=torch.meshgrid(pose.new_tensor(SCAN_X),pose.new_tensor(SCAN_Y),indexing='ij')
    x,y=xs.flatten(),ys.flatten();c,s=pose[:,2:3].cos(),pose[:,2:3].sin()
    xy=torch.stack((pose[:,0:1]+c*x-s*y,pose[:,1:2]+s*x+c*y),-1)
    return (ground(pose[:,:2],field)[:,None]-ground(xy,field)).clamp(-1.2,1.2)
