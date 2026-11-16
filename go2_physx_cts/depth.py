"""50Hz exact static collision-mesh depth and frozen keyed sensor noise.

Only observed/replayed states enter this module. Future imagined observations
are WM latents, so ray intersection intentionally has no autograd interface.
"""
from pathlib import Path
import hashlib,math,sys
import torch
import warp as wp
from . import state as st
from .capture_cache import CapturedImages
from .camera_geometry import verify_camera_contract

@wp.kernel
def camera_rays(mesh:wp.uint64,origins:wp.array(dtype=wp.vec3),rotations:wp.array(dtype=wp.mat33),
                rays:wp.array(dtype=wp.vec3),output:wp.array(dtype=float)):
    i=wp.tid();row=i//3072;pixel=i%3072
    ray=rays[pixel];length=wp.length(ray);direction=rotations[row]*(ray/length)
    hit=wp.mesh_query_ray(mesh,origins[row],direction,6.0*length)
    depth=float(-6.0)
    if hit.result:
        z=hit.t/length
        if z>=0.1 and z<=6.0:depth=z
    output[i]=depth


def add_noise(noise,clean,valid,keys,counter,mixed_precision=False):
    from go2_depth_input.core import fields_from_keys
    site=torch.ones_like(keys);fields,epsilon=fields_from_keys(keys,counter)
    with torch.autocast(device_type=clean.device.type,dtype=torch.bfloat16,enabled=mixed_precision):
        logits,sigma=noise.parameters_at(clean,site)
    noisy,_,_=noise.corrupt(clean,site,logits+math.log(2.),sigma,fields,epsilon,valid,True,.5)
    return noisy[:,None]

class DepthObservation:
    def __init__(self,cfg,field):
        package=Path(cfg.depth_package)
        if str(package) not in sys.path:sys.path.insert(0,str(package))
        from go2_depth_input.core import noise_keys
        from go2_depth_input.vendor.noise_reference import DepthNoise
        from go2_depth_input.vendor import camera
        self.field=field;self.device=cfg.device;self.seed=cfg.seed;self.period=1
        self.noise_keys=noise_keys;self.camera=camera;self.chunk_size=cfg.depth_chunk_size
        self.noise_mixed_precision=cfg.depth_precision=='bf16'
        weights=package/'go2_depth_input/assets/weights.pt'
        ck=torch.load(weights,map_location='cpu',weights_only=True)
        self.noise=DepthNoise().to(cfg.device).eval().requires_grad_(False);self.noise.load_state_dict(ck['noise']);del ck
        self.counter=torch.arange(4*54*70,device=cfg.device,dtype=torch.int64)[None]
        self.rays=camera.rays(cfg.device).reshape(-1,3).contiguous()
        capacity=min(cfg.num_envs*(cfg.buffer_len+2),int(cfg.depth_cache_gib*2**30//(48*64*2+88)))
        self.images=CapturedImages(capacity,cfg.device,cfg.num_envs)
        self._noise_fn=torch.compile(add_noise,dynamic=False,fullgraph=True) if cfg.compile else add_noise
        self.noise_pool=None
        if len(cfg.noise_devices)>1:
            from .noise_parallel import FrozenNoisePool
            self.noise_pool=FrozenNoisePool(cfg.noise_devices,self.noise,add_noise,
                self.chunk_size,self.noise_mixed_precision,cfg.compile)
        self.metadata={'engine':'static_collision_mesh_gpu_first_hit','depth_hz':50,'output_shape':[1,48,64],
            'camera_contract':verify_camera_contract(camera),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'frozen':True,'geometry_precision':'fp32','noise_precision':cfg.depth_precision,'cache_precision':'fp16 final noisy depth',
            'capture_cache_capacity':self.images.capacity,'cache_bytes':self.images.storage_bytes,
            'capture_cache':'bounded direct-index CPU noisy images, keyed trajectory/episode/frame/scene/pose',
            'scene_geometry':field.raycaster.identity,'future_images':False,'robot_self_occlusion':False,
            'noise_devices':list(cfg.noise_devices) or [cfg.device],
            'multi_gpu_scope':'frozen noise only; trainable parameters on primary'}

    @staticmethod
    def rotation(pose):
        q=st.quat_normalize(pose[:,3:]);basis=torch.eye(3,device=pose.device)[None].expand(len(pose),-1,-1)
        return st.quat_apply(q[:,None,:],basis).transpose(1,2)

    @torch.no_grad()
    def render_clean(self,pose):
        rotation=self.rotation(pose);origin=pose[:,:3]+(rotation@pose.new_tensor(self.camera.MOUNT)[None,:,None]).squeeze(-1)
        optical=self.camera.optical_rotation(rotation).contiguous();origin=origin.contiguous()
        output=torch.empty(len(pose)*3072,device=pose.device,dtype=torch.float32)
        wp.launch(camera_rays,dim=len(output),inputs=[self.field.raycaster.mesh.id,wp.from_torch(origin,dtype=wp.vec3),
            wp.from_torch(optical,dtype=wp.mat33),wp.from_torch(self.rays,dtype=wp.vec3),wp.from_torch(output)],device=str(pose.device))
        return output.reshape(-1,1,48,64)

    @torch.no_grad()
    def paired(self,pose,scene_ids=None,frame_index=0,episode_ids=0,trajectory_ids=None,capture_keys=None):
        if trajectory_ids is None:raise ValueError('Explicit trajectory IDs required')
        if scene_ids is None:scene_ids=self.field.scene_ids(pose)
        noisy=self.captured(pose,scene_ids,frame_index,episode_ids,trajectory_ids,capture_keys)
        return noisy,self.clean(pose)

    @torch.no_grad()
    def clean(self,pose):
        # Match the old signed FP16 geometry-cache quantization exactly. Only
        # the small reconstruction target batch needs this clean observation.
        return self.render_clean(pose).half().float().abs()

    def _render_noisy(self,pose,keys):
        if self.noise_pool is not None:
            return self.noise_pool.render(self.render_clean,pose,keys)
        output=[]
        for start in range(0,len(pose),self.chunk_size):
            sl=slice(start,start+self.chunk_size)
            signed=self.render_clean(pose[sl]).half().float()
            z=signed[:,0].abs();valid=(signed[:,0]>0).float();count=len(z);batch_keys=keys[sl]
            if count<self.chunk_size:
                pad=self.chunk_size-count
                zz=torch.cat((z,z[:1].expand(pad,-1,-1)))
                vv=torch.cat((valid,valid[:1].expand(pad,-1,-1)))
                kk=torch.cat((batch_keys,batch_keys[:1].expand(pad)))
            else:zz,vv,kk=z,valid,batch_keys
            output.append(self._noise_fn(self.noise,zz,vv,kk,self.counter,self.noise_mixed_precision)[:count].half())
        return torch.cat(output)

    @torch.no_grad()
    def captured(self,pose,scene_ids,frame_index,episode_ids,trajectory_ids,capture_keys=None):
        n=len(pose)
        if not n:return pose.new_empty((0,1,48,64))
        def vector(x):return torch.as_tensor(x,device=pose.device).long().reshape(-1).expand(n)
        keys=self.noise_keys(vector(trajectory_ids),vector(frame_index),seed=self.seed,episode_ids=vector(episode_ids))
        meta=self.images.keys(pose,scene_ids,frame_index,episode_ids,trajectory_ids) if capture_keys is None else capture_keys
        return self.images.get(lambda points,ids:self._render_noisy(points,keys[ids]),pose,scene_ids,
            frame_index,episode_ids,trajectory_ids,capture_keys=meta).float()

    @torch.no_grad()
    def warmup(self,pose,scene_ids,episode_ids,trajectory_ids):
        return self.captured(pose,scene_ids,0,episode_ids,trajectory_ids)
