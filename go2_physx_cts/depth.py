"""Frozen depth/noise observation module with full-body-pose input derivatives.

The supplied projection/sensor model uses the accepted compact image refiner.
This adapter handles scene sharing, quaternion conversion and replay keys.
"""
from pathlib import Path
import hashlib
import sys
import torch
from . import state as st
from .camera_geometry import verify_camera_contract,ATLAS_STRIDE_M
from .capture_cache import CapturedImages


def _paired_core(core,cache,origins,position,rotation,scene_ids,keys):
    """Same pinned geometry/noise math as FrozenCore.forward, with clean output.

    This supervised-only split avoids performing the expensive surface query
    twice to obtain a noisy/clean pair. No geometry/noise parameters are trained.
    """
    import math
    from go2_depth_input.core import fields_from_keys
    clean,logit=core.geometry_cached(cache,scene_ids,position-origins[scene_ids],rotation)
    probability=logit.sigmoid()
    valid=(logit>0).float().detach()-probability.detach()+probability if core.hard else probability
    site=torch.ones_like(scene_ids)
    fields,epsilon=fields_from_keys(keys,core.noise_counter)
    logits,sigma=core.noise.parameters_at(clean,site)
    noisy,_,_=core.noise.corrupt(clean,site,logits+math.log(core.hole_strength),sigma,fields,epsilon,valid,core.hard,.5)
    return noisy[:,None],clean[:,None]

class DepthObservation:
    def __init__(self,cfg,field):
        package=Path(cfg.depth_package)
        if (package/'go2_depth_input').is_dir() and str(package) not in sys.path:
            sys.path.insert(0,str(package))
        from go2_depth_input import noise_keys
        from go2_depth_input.api import SceneCache
        from go2_depth_input.vendor import camera
        camera_contract=verify_camera_contract(camera)
        self.noise_keys=noise_keys
        self.field=field
        if not cfg.fast_depth_checkpoint:raise ValueError('An accepted compact observer is required')
        from .fast_depth.api import FastDepthGenerator
        self.engine=FastDepthGenerator(cfg.fast_depth_checkpoint,str(package),device=cfg.device,
            chunk_size=cfg.depth_chunk_size,compile_mode='default' if cfg.compile else 'off',
            precision=cfg.depth_precision,require_accepted=True)
        maps,origins=field.depth_atlases()
        maps=maps.to(device=cfg.device)
        origins=origins.to(device=cfg.device,dtype=torch.float32)
        self.scene_origins=origins.clone()
        # Identical zero-height tiles share one immutable encoded scene. Move
        # each query into its tile-local frame; pose derivatives are unchanged.
        # This avoids encoding thousands of background tiles in large arenas.
        nonflat=maps.ne(0).any(dim=(1,2))
        source_count=len(maps)
        if bool(nonflat.all()):
            # Reuse HeightField's immutable map storage; indexing+concatenation
            # here would duplicate about3GB on the measured full terrain atlas.
            self.scene_remap=torch.arange(len(maps),device=cfg.device)
            maps=maps.float()
        else:
            keep=nonflat.nonzero().flatten()
            self.scene_remap=torch.full((len(maps),),len(keep),dtype=torch.long,device=cfg.device)
            self.scene_remap[keep]=torch.arange(len(keep),device=cfg.device)
            maps=torch.cat((maps[keep],maps.new_zeros((1,512,512))),0).float()
        origins=torch.zeros((len(maps),3),device=cfg.device)
        # Encode in bounded chunks: preparing hundreds of maps in one CNN pass
        # would allocate training-sized activations despite the frozen weights.
        local,context=[],[]
        for start in range(0,len(maps),16):
            cache=self.engine.prepare_scenes(maps[start:start+16],origins[start:start+16])
            local.append(cache.tensors[4]);context.append(cache.tensors[5])
        lf=torch.cat(local,0);cf=torch.cat(context,0)
        owned=maps
        lv=lf.view(len(maps),128,128,8).permute(0,3,1,2)
        cv=cf.view(len(maps),64,64,8).permute(0,3,1,2)
        tensors=(owned,lv,cv,owned.reshape(-1,1),lf,cf)
        self.cache=SceneCache(tensors,origins.clone(),len(maps))
        cache_bytes=owned.numel()*4+lf.numel()*4+cf.numel()*4
        self.seed=cfg.seed
        self.period=cfg.depth_period
        self._paired_fn=torch.compile(_paired_core,dynamic=False,fullgraph=True) if cfg.compile else _paired_core
        self.images=CapturedImages(cfg.num_envs*((cfg.buffer_len+cfg.depth_period-1)//cfg.depth_period+2),cfg.device)
        self.metadata={'package_version':'0.1.0','frozen':True,'precision':cfg.depth_precision,
                       'weights_sha256':hashlib.sha256((package/'go2_depth_input/assets/weights.pt').read_bytes()).hexdigest(),
                       'source_lock_sha256':hashlib.sha256((package/'go2_depth_input/assets/source_lock.json').read_bytes()).hexdigest(),
                       'output_shape':[1,48,64],'depth_hz':1/(cfg.dt*cfg.depth_period),'body_pose':'xyz_xyzw',
                       'capture_cache':'FP16 frozen images keyed by trajectory/episode/control-frame/scene/full-pose',
                       'capture_cache_storage':'CPU pageable blocks; at most48MiB pinned staging',
                       'capture_cache_capacity':self.images.capacity,'noise_key_frame':'control_step',
                       'atlas_count':source_count,'encoded_atlas_count':len(maps),'atlas_span_m':25.6,
                       'atlas_stride_m':ATLAS_STRIDE_M,'camera_contract':camera_contract,
                       'atlas_selection':'full camera frustum XY AABB centre',
                       'scene_origin':'map centre xy, world z offset',
                       'recompute':True,'noise':'explicit stateless per-frame keys',
                       'cache_bytes':cache_bytes,'engine':'compact_surface_depth',
                       'direct_model':self.engine.metadata,
                       'mount_body_m':[.101958,0.,.327]}

    @torch.no_grad()
    def captured(self,pose,scene_ids,frame_index,episode_ids,trajectory_ids,capture_keys=None):
        """Real captures and imagined starting observations; future steps use WM latent."""
        return self.images.get(self.render,pose,scene_ids,frame_index,episode_ids,trajectory_ids,capture_keys=capture_keys)

    @torch.no_grad()
    def paired(self,pose,scene_ids=None,frame_index=0,episode_ids=0,trajectory_ids=None):
        """Real-state noisy/clean supervision pair in metric optical depth.

        Clean means the frozen geometry module's output before sensor noise,
        not a claim of exact ray-traced ground truth. Pairing retains exact pose,
        scene and stateless noise keys; it is never called on imagined futures.
        """
        if trajectory_ids is None:raise ValueError('Explicit trajectory IDs required for paired depth')
        if scene_ids is None:scene_ids=self.field.scene_ids(pose)
        ids=scene_ids.reshape(-1).long()
        coverage=self.field.scene_coverage(pose,scene_ids=ids,return_details=True)
        if not bool(coverage['valid'].all()):raise RuntimeError('Paired depth outside represented geometry')
        n=len(pose)
        if not n:return pose.new_empty((0,1,48,64)),pose.new_empty((0,1,48,64))
        def index(x):return torch.as_tensor(x,device=pose.device).reshape(-1).long().expand(n)
        keys=self.noise_keys(index(trajectory_ids),index(frame_index),seed=self.seed,episode_ids=index(episode_ids))
        from go2_depth_input.vendor.camera import MOUNT,optical_rotation
        rotation=self.rotation(pose).float()
        camera=pose[:,:3].float()-self.scene_origins[ids]+(rotation@pose.new_tensor(MOUNT)[None,:,None]).squeeze(-1)
        optical=optical_rotation(rotation)
        noisy,clean=[],[]
        engine=self.engine
        for lo in range(0,n,engine.chunk_size):
            sl=slice(lo,min(n,lo+engine.chunk_size))
            with engine._precision_context(),engine._autocast():
                pair=self._paired_fn(engine.core,self.cache.tensors,self.cache.origins,camera[sl],optical[sl],self.scene_remap[ids[sl]],keys[sl])
            noisy.append(pair[0]);clean.append(pair[1])
        return torch.cat(noisy).float(),torch.cat(clean).float()

    @staticmethod
    def rotation(pose):
        q=st.quat_normalize(pose[:,3:7])
        basis=torch.eye(3,device=pose.device,dtype=pose.dtype)[None].expand(len(pose),-1,-1)
        return st.quat_apply(q[:,None,:],basis).transpose(1,2)

    def render(self,pose,scene_ids=None,frame_index=0,episode_ids=0,trajectory_ids=None,mode='recompute'):
        if trajectory_ids is None:raise ValueError('Explicit trajectory IDs required for reproducible noise')
        if scene_ids is None:scene_ids=self.field.scene_ids(pose)
        ids=scene_ids.reshape(-1).long()
        # Check the actual supplied tile, including replayed capture IDs. A
        # clamped atlas lookup must never masquerade as a valid camera image.
        # This is domain validation only; the rendered pose remains attached.
        with torch.no_grad():
            coverage=self.field.scene_coverage(pose,scene_ids=ids,return_details=True)
            if not bool(coverage['valid'].all()):
                failed={name:int((~coverage[name]).sum()) for name in
                        ('valid','atlas_covered','field_covered','finite_pose','scene_exists')}
                raise RuntimeError('Depth frustum outside represented geometry: '+str(failed))
        def index(x):return x.reshape(-1).long() if isinstance(x,torch.Tensor) else int(x)
        keys=self.noise_keys(index(trajectory_ids),index(frame_index),seed=self.seed,episode_ids=index(episode_ids))
        # Freeze parameters, NEVER detach input pose on the imagination path.
        position=pose[:,:3].float()-self.scene_origins[ids]
        return self.engine.render_from_body(self.cache,position,self.rotation(pose).float(),self.scene_remap[ids],keys,mode=mode)

    def warmup(self,pose,scene_ids,episode_ids,trajectory_ids):
        from go2_depth_input.vendor.camera import MOUNT,optical_rotation
        ids=scene_ids.reshape(-1).long()
        keys=self.noise_keys(trajectory_ids.reshape(-1),0,seed=self.seed,episode_ids=episode_ids.reshape(-1))
        rotation=self.rotation(pose)
        camera=pose[:,:3]-self.scene_origins[ids]+(rotation@pose.new_tensor(MOUNT)[None,:,None]).squeeze(-1)
        self.engine.warmup(self.cache,camera,optical_rotation(rotation),self.scene_remap[ids],keys)
