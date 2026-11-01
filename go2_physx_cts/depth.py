"""Frozen depth/noise observation module with full-body-pose input derivatives.

The geometry and sensor model is the supplied versioned package. This adapter
only handles scene sharing, quaternion conversion and explicit replay keys.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib
import sys
import torch
from . import state as st
from .camera_geometry import verify_camera_contract,ATLAS_STRIDE_M


class CapturedImages:
    """Bounded cache of frozen observations, never of trainable CNN features.

    Each key includes the capture's episode, control step, scene and exact pose.
    This also makes restored diagnostic branches safe. Evicted observations can
    be reconstructed exactly from the immutable model and stateless noise key.
    Images are stored once in FP16 (not BF16), with FP32 tensors at the API.
    """
    def __init__(self,capacity,device,block_size=4096):
        self.capacity=int(capacity);self.device=device;self.block_size=min(block_size,self.capacity)
        self.entries=OrderedDict();self.blocks=[];self.hits=0;self.misses=0

    def _storage(self,slot):
        block,index=divmod(slot,self.block_size)
        while len(self.blocks)<=block:
            remaining=self.capacity-len(self.blocks)*self.block_size
            self.blocks.append(torch.empty((min(self.block_size,remaining),1,48,64),
                                           device=self.device,dtype=torch.float16))
        return self.blocks[block],index

    @torch.no_grad()
    def get(self,render,pose,scene,frame,episode,trajectory):
        n=len(pose)
        def vector(x):
            return torch.as_tensor(x,device=pose.device).long().reshape(-1).expand(n)
        scene,frame,episode,trajectory=map(vector,(scene,frame,episode,trajectory))
        # CPU indexing is small and bounded; the images remain on their device.
        ids=torch.stack((trajectory,episode,frame,scene),-1).cpu().tolist()
        poses=pose.detach().float().contiguous().cpu().numpy()
        keys=[(*row,poses[i].tobytes()) for i,row in enumerate(ids)]
        out=pose.new_empty((n,1,48,64),dtype=torch.float32)
        missing={};hits=[]
        for i,key in enumerate(keys):
            slot=self.entries.get(key)
            if slot is None:missing.setdefault(key,[]).append(i)
            else:
                self.entries.move_to_end(key);hits.append((i,slot))
        # Group by storage block to avoid one CUDA operation per observation.
        by_block={}
        for row,slot in hits:by_block.setdefault(slot//self.block_size,[]).append((row,slot%self.block_size))
        for block,rows in by_block.items():
            dst,src=zip(*rows)
            out[list(dst)]=self.blocks[block][list(src)].float()
        self.hits+=len(hits);self.misses+=len(missing)
        if missing:
            first=[rows[0] for rows in missing.values()]
            generated=render(pose[first],scene[first],frame[first],episode[first],trajectory[first],mode='inference')
            stored=generated.to(torch.float16)
            writes={};duplicate_dst=[];duplicate_src=[]
            for j,(key,rows) in enumerate(missing.items()):
                if len(self.entries)<self.capacity:slot=len(self.entries)
                else:_,slot=self.entries.popitem(last=False)
                self.entries[key]=slot
                self._storage(slot)
                # A request may exceed a small cache's capacity. Only the last
                # occupant of each slot is written, avoiding duplicate-index
                # CUDA writes whose ordering is otherwise unspecified.
                writes.setdefault(slot//self.block_size,{})[slot%self.block_size]=j
                duplicate_dst.extend(rows);duplicate_src.extend([j]*len(rows))
            for block,rows in writes.items():
                dst,src=zip(*rows.items())
                self.blocks[block][list(dst)]=stored[list(src)]
            out[duplicate_dst]=stored[duplicate_src].float()
        return out

    @property
    def storage_bytes(self):return sum(x.numel()*x.element_size() for x in self.blocks)

class DepthObservation:
    def __init__(self,cfg,field):
        package=Path(cfg.depth_package)
        if (package/'go2_depth_input').is_dir() and str(package) not in sys.path:
            sys.path.insert(0,str(package))
        from go2_depth_input import DepthInputGenerator,noise_keys
        from go2_depth_input.api import SceneCache
        from go2_depth_input.vendor import camera
        camera_contract=verify_camera_contract(camera)
        self.noise_keys=noise_keys
        self.field=field
        direct=bool(cfg.fast_depth_checkpoint)
        if direct:
            from .fast_depth.api import FastDepthGenerator
            self.engine=FastDepthGenerator(cfg.fast_depth_checkpoint,str(package),device=cfg.device,
                chunk_size=max(256,cfg.depth_chunk_size),compile_mode='default' if cfg.compile else 'off',
                precision=cfg.depth_precision,require_accepted=True)
        else:
            self.engine=DepthInputGenerator(device=cfg.device,chunk_size=cfg.depth_chunk_size,
                compile_mode='default' if cfg.compile else 'off',precision=cfg.depth_precision)
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
        if direct:
            self.cache=self.engine.prepare_scenes(maps,origins)
            cache_bytes=self.cache.embedding.numel()*self.cache.embedding.element_size()
        else:
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
        self.images=CapturedImages(cfg.num_envs*((cfg.buffer_len+cfg.depth_period-1)//cfg.depth_period+2),cfg.device)
        self.metadata={'package_version':'0.1.0','frozen':True,'precision':cfg.depth_precision,
                       'weights_sha256':hashlib.sha256((package/'go2_depth_input/assets/weights.pt').read_bytes()).hexdigest(),
                       'source_lock_sha256':hashlib.sha256((package/'go2_depth_input/assets/source_lock.json').read_bytes()).hexdigest(),
                       'output_shape':[1,48,64],'depth_hz':1/(cfg.dt*cfg.depth_period),'body_pose':'xyz_xyzw',
                       'capture_cache':'FP16 frozen images keyed by trajectory/episode/control-frame/scene/full-pose',
                       'capture_cache_capacity':self.images.capacity,'noise_key_frame':'control_step',
                       'atlas_count':source_count,'encoded_atlas_count':len(maps),'atlas_span_m':25.6,
                       'atlas_stride_m':ATLAS_STRIDE_M,'camera_contract':camera_contract,
                       'atlas_selection':'full camera frustum XY AABB centre',
                       'scene_origin':'map centre xy, world z offset',
                       'recompute':True,'noise':'explicit stateless per-frame keys',
                       'cache_bytes':cache_bytes,'engine':'direct_depth' if direct else 'source_ray_generator',
                       'direct_model':self.engine.metadata if direct else None,
                       'mount_body_m':[.101958,0.,.327]}

    @torch.no_grad()
    def captured(self,pose,scene_ids,frame_index,episode_ids,trajectory_ids):
        """Only for already collected poses; future imagination uses render()."""
        return self.images.get(self.render,pose,scene_ids,frame_index,episode_ids,trajectory_ids)

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
