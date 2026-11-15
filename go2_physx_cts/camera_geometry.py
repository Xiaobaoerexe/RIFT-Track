"""Confirmed AS2 fixed-camera intrinsics/extrinsics and source parity check."""
import math
import torch

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
        'mount_body_m':list(MOUNT),'direction':'body horizontal forward',
        'confirmed':'2026-11-15 user-confirmed inherited model coordinates; not measured hardware calibration'}
