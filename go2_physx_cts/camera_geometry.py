"""Geometry of the frozen depth package0.1.0 camera; no observation model.

The PhysX snapshot selects an atlas before the depth adapter imports its package.
These exact constants are therefore available independently, and the adapter must
call verify_camera_contract against the loaded package before rendering.
"""
import math
import torch
from . import state as st

ATLAS_SPAN_M=25.6
ATLAS_STRIDE_M=12.
WIDTH,HEIGHT=64,48
RAW_WIDTH,RAW_HEIGHT=640,480
NEAR,FAR=.1,6.
FX_RAW=RAW_WIDTH/(2*math.tan(math.radians(87/2)))
FY_RAW=RAW_HEIGHT/(2*math.tan(math.radians(58/2)))
RAW_CX,RAW_CY=319.5,239.5
SAMPLE_OFFSET=5
FX,FY=FX_RAW/10,FY_RAW/10
CX,CY=(RAW_CX-SAMPLE_OFFSET-.5)/10,(RAW_CY-SAMPLE_OFFSET-.5)/10
MOUNT=(.101958,0.,.327)

# Match vendor.camera.rays()'s float32 arithmetic, including pixel sampling.
_u=torch.tensor([0,WIDTH-1,0,WIDTH-1])
_v=torch.tensor([0,0,HEIGHT-1,HEIGHT-1])
_ray=torch.stack(((_u-CX)/FX,(_v-CY)/FY,torch.ones_like(_u)),-1).float()
_optical=torch.cat((_ray*NEAR,_ray*FAR))
# Optical right/down/forward -> body forward/left/up, then fixed body mount.
_body=torch.stack((_optical[:,2],-_optical[:,0],-_optical[:,1]),-1)+torch.tensor(MOUNT)
_BODY_CORNERS=tuple(tuple(float(v) for v in row) for row in _body)


def frustum_world_corners(pose):
    """Eight actual near/far image corner rays under full xyz/xyzw body pose."""
    q=st.quat_normalize(pose[...,3:7])
    return pose[...,None,:3]+st.quat_apply(q[...,None,:],pose.new_tensor(_BODY_CORNERS))


def verify_camera_contract(camera):
    """Fail explicitly if the loaded frozen package and selection geometry differ."""
    scalars=dict(WIDTH=WIDTH,HEIGHT=HEIGHT,RAW_WIDTH=RAW_WIDTH,RAW_HEIGHT=RAW_HEIGHT,
        NEAR=NEAR,FAR=FAR,FX_RAW=FX_RAW,FY_RAW=FY_RAW,RAW_CX=RAW_CX,RAW_CY=RAW_CY,
        SAMPLE_OFFSET=SAMPLE_OFFSET,FX=FX,FY=FY,CX=CX,CY=CY)
    bad=[name for name,value in scalars.items() if not math.isclose(float(getattr(camera,name)),value,rel_tol=0,abs_tol=1e-12)]
    if tuple(camera.MOUNT)!=MOUNT:bad.append('MOUNT')
    rays=camera.rays(device='cpu')
    if rays.shape!=(HEIGHT,WIDTH,3) or not torch.equal(rays[[0,0,-1,-1],[0,-1,0,-1]],_ray):bad.append('corner_rays')
    optical=camera.optical_rotation(torch.eye(3))
    expected=torch.tensor([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    if not torch.equal(optical,expected):bad.append('optical_rotation')
    if bad:raise RuntimeError('Frozen depth camera contract mismatch: '+', '.join(bad))
    return {'source':'go2_depth_input0.1.0/vendor/camera.py','verified':True,
        'output_shape':[HEIGHT,WIDTH],'optical_depth_range_m':[NEAR,FAR],
        'mount_body_m':list(MOUNT),'atlas_span_m':ATLAS_SPAN_M,'atlas_stride_m':ATLAS_STRIDE_M,
        'selection':'world-XY AABB centre of eight full-pose near/far corner rays'}
