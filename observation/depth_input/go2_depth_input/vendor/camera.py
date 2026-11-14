"""Camera contract. XYZ body frame: forward, left, up. Optical: right, down, forward."""
import math
import torch

WIDTH, HEIGHT = 64, 48
RAW_WIDTH, RAW_HEIGHT = 640, 480
NEAR, FAR = .10, 6.0
FX_RAW = RAW_WIDTH / (2 * math.tan(math.radians(87 / 2)))
FY_RAW = RAW_HEIGHT / (2 * math.tan(math.radians(58 / 2)))
# Nearest sample at the upper of the two central pixels in each 10x10 block.
# A measured camera calibration can replace these explicitly nominal values.
RAW_CX, RAW_CY = 319.5, 239.5
SAMPLE_OFFSET = 5
FX, FY = FX_RAW / 10, FY_RAW / 10
# RTX OpenCV rays use (integer pixel + 0.5). Verified against a planar metric target.
# These output intrinsics use integer pixel-center coordinates in the neural model.
CX, CY = (RAW_CX - SAMPLE_OFFSET - .5) / 10, (RAW_CY - SAMPLE_OFFSET - .5) / 10
MOUNT = (.101958, 0., .327)


def contract():
    return dict(raw_wh=[RAW_WIDTH, RAW_HEIGHT], output_wh=[WIDTH, HEIGHT],
                raw_intrinsics=[FX_RAW, FY_RAW, RAW_CX, RAW_CY],
                intrinsics=[FX, FY, CX, CY], nominal_fov_deg=[87, 58],
                calibration_source='D455 nominal FOV; actual D457 calibration not supplied',
                mount_position_base_m=MOUNT, mount_direction='horizontal_forward',
                depth='distance_to_image_plane_m', valid_range_m=[NEAR, FAR],
                downsample='nearest_raw[5::10,5::10]; no interpolation across occlusions',
                rtx_pixel_center_offset=.5, model_pixel_center_offset=0.,
                quaternion='xyzw', optical_axes='right_down_forward', body_axes='forward_left_up')


def intrinsics(raw=False, device='cpu'):
    x,y,cx,cy = (FX_RAW,FY_RAW,RAW_CX,RAW_CY) if raw else (FX,FY,CX+.5,CY+.5)
    return torch.tensor([[x,0,cx],[0,y,cy],[0,0,1.]],device=device)


def rays(device='cpu'):
    v,u=torch.meshgrid(torch.arange(HEIGHT,device=device),torch.arange(WIDTH,device=device),indexing='ij')
    return torch.stack(((u-CX)/FX,(v-CY)/FY,torch.ones_like(u)),dim=-1).float()


def downsample(raw):
    return raw[...,SAMPLE_OFFSET::10,SAMPLE_OFFSET::10]


def rpy_matrix(rpy):
    r,p,y=rpy.unbind(-1);cr,sr=r.cos(),r.sin();cp,sp=p.cos(),p.sin();cy,sy=y.cos(),y.sin()
    return torch.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,
                        sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,
                        -sp,cp*sr,cp*cr),-1).reshape(*rpy.shape[:-1],3,3)


def optical_rotation(body_rotation):
    bc=body_rotation.new_tensor([[0,0,1],[-1,0,0],[0,-1,0]])
    return body_rotation @ bc


def sample_poses(height_maps, scene_ids, generator, paired=False):
    """75% nominal/running, 15% transition, 10% fallen orientations.

    Uses virtual body pose only; no robot articulation or dynamics is created.
    Ground is sampled conservatively at the camera and a virtual trunk footprint.
    Fallen samples are observation coverage, not a physically simulated recovery dataset.
    """
    n=len(scene_ids);dev=height_maps.device
    rnd=lambda *s:torch.rand(*s,device=dev,generator=generator)
    norm=lambda *s:torch.randn(*s,device=dev,generator=generator)
    xy=(rnd(n,2)-.5)*8.0
    bucket=rnd(n);mode=torch.where(bucket<.75,0,torch.where(bucket<.90,1,2))
    rp=norm(n,2)*torch.tensor([.10,.13],device=dev)
    rp=rp.clamp(-.40,.40)
    transition=(rnd(n,2)-.5)*torch.tensor([2*math.pi,math.pi],device=dev)
    rp=torch.where((mode==1)[:,None],transition,rp)
    # Side, back, nose-down, nose-up plus jitter; signed full yaw is always sampled.
    catalog=torch.tensor([[math.pi/2,0],[-math.pi/2,0],[math.pi,0],
                          [0,math.pi/2],[0,-math.pi/2],[math.pi,.5]],device=dev)
    idx=torch.randint(len(catalog),(n,),device=dev,generator=generator)
    rp=torch.where((mode==2)[:,None],catalog[idx]+norm(n,2)*.18,rp)
    rpy=torch.cat((rp,(rnd(n,1)-.5)*2*math.pi),-1)
    rb=rpy_matrix(rpy)
    h=height_at(height_maps,scene_ids,xy)
    body_h=torch.where(mode==0,.36+norm(n)*.035,
                       torch.where(mode==1,.25+rnd(n)*.25,.10+rnd(n)*.20))
    body=torch.cat((xy,(h+body_h)[:,None]),-1)
    # Keep a representative torso box out of the ground without creating physics.
    corners=body.new_tensor([[x,y,z] for x in [-.20,.20] for y in [-.09,.09] for z in [-.07,.07]])
    corners=body[:,None,:]+torch.einsum('bij,kj->bki',rb,corners)
    hc=height_at(height_maps,scene_ids[:,None].expand(-1,8).reshape(-1),corners[...,:2].reshape(-1,2)).reshape(n,8)
    body[:,2]+=(hc+.015-corners[:,:,2]).amax(1).clamp_min(0)
    pos=body+torch.einsum('bij,j->bi',rb,body.new_tensor(MOUNT))
    # Camera must be outside the height surface, even for upside-down poses.
    floor=height_at(height_maps,scene_ids,pos[:,:2])
    lift=(floor+.12-pos[:,2]).clamp_min(0)
    pos[:,2]+=lift
    return pos,optical_rotation(rb),mode


def height_at(height_maps,ids,xy):
    # World-local map: grid nodes [-12.8,12.8], includes complete view frustum.
    grid=xy.float()/12.8
    # Advanced gather with bilinear interpolation avoids duplicating 512x512 maps per ray.
    m=height_maps
    p=(grid+1)*.5*(m.shape[-1]-1);p=p.clamp(0,m.shape[-1]-1.0001)
    ix=p[:,0].floor().long();iy=p[:,1].floor().long();dx=p[:,0]-ix;dy=p[:,1]-iy
    return ((m[ids,iy,ix]*(1-dx)+m[ids,iy,ix+1]*dx)*(1-dy)
            +(m[ids,iy+1,ix]*(1-dx)+m[ids,iy+1,ix+1]*dx)*dy)
