"""SuperMimic observation API: GPU tensors in/out; pose gradients are retained."""
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import math
import torch
from .core import FrozenCore,noise_keys
from .vendor.geometry_reference import DepthGenerator
from .vendor.noise_reference import DepthNoise
from .vendor.camera import MOUNT,optical_rotation


@dataclass(frozen=True)
class SceneCache:
    tensors: tuple
    origins: torch.Tensor
    scene_count: int


class _Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,engine,cache,position,rotation,scene_ids,keys):
        ctx.engine=engine;ctx.cache=cache
        ctx.save_for_backward(position,rotation,scene_ids,keys,*cache.tensors,cache.origins)
        ctx.set_materialize_grads(False)
        return engine._chunks(cache,position,rotation,scene_ids,keys)

    @staticmethod
    def backward(ctx,grad):
        if grad is None:return (None,)*6
        position,rotation,ids,keys,*_=ctx.saved_tensors
        dp=torch.empty_like(position);dr=torch.empty_like(rotation)
        with torch.enable_grad():
            for start in range(0,len(position),ctx.engine.chunk_size):
                stop=min(start+ctx.engine.chunk_size,len(position))
                p=position[start:stop].detach().requires_grad_(True);r=rotation[start:stop].detach().requires_grad_(True)
                y=ctx.engine._block(ctx.cache,p,r,ids[start:stop],keys[start:stop])
                with ctx.engine._precision_context():
                    gp,gr=torch.autograd.grad(y,(p,r),grad[start:stop],create_graph=False)
                dp[start:stop]=gp;dr[start:stop]=gr
        return None,None,dp,dr,None,None


class DepthInputGenerator:
    """Frozen first-order differentiable generator, output [N,1,48,64] metres.

    mode='recompute' saves poses/keys instead of all ray/CNN activations.
    mode='autograd' keeps the normal graph (higher memory).
    mode='inference' is for real collection/evaluation only, with no input gradients.
    """
    def __init__(self,weights=None,device='cuda',chunk_size=32,compile_mode='default',precision='fp32',channels_last=True,hole_strength=2.,hard=True,optimized=True,scan='log'):
        self.device=torch.device(device)
        if precision not in ('fp32','tf32','bf16'):raise ValueError('precision must be fp32, tf32 or bf16')
        if self.device.type=='cpu' and precision!='fp32':raise ValueError('CPU path requires precision=fp32')
        if not isinstance(chunk_size,int) or chunk_size<=0:raise ValueError('chunk_size must be a positive integer')
        if scan not in ('product','log'):raise ValueError('scan must be product or log')
        if not math.isfinite(hole_strength) or hole_strength<=0:raise ValueError('hole_strength must be finite and positive')
        if compile_mode not in ('off','default','reduce-overhead','max-autotune','max-autotune-no-cudagraphs'):raise ValueError('Unsupported torch.compile mode')
        self.chunk_size=chunk_size
        self.precision=precision;self.compile_mode=compile_mode;self.channels_last=channels_last
        weight_path=Path(weights) if weights else Path(__file__).parent/'assets'/'weights.pt'
        state=torch.load(weight_path,map_location='cpu',weights_only=True)
        geometry=DepthGenerator();geometry.load_state_dict(state['geometry'])
        noise=DepthNoise();noise.load_state_dict(state['noise'])
        self.core=FrozenCore(geometry,noise,hole_strength,hard,optimized,scan).to(self.device)
        if channels_last:self.core.to(memory_format=torch.channels_last)
        self._fn=self.core if compile_mode=='off' else torch.compile(self.core,mode=compile_mode,dynamic=False,fullgraph=True)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.precision=='bf16')

    @contextmanager
    def _precision_context(self):
        # Scoped precision: do not persistently change the locomotion model's
        # backend policy. GPU operations are compiled/launched inside this scope.
        if self.precision not in ('fp32','tf32') or self.device.type!='cuda':
            yield
            return
        previous=torch.backends.cuda.matmul.allow_tf32
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                                       benchmark=torch.backends.cudnn.benchmark,
                                       benchmark_limit=torch.backends.cudnn.benchmark_limit,
                                       deterministic=torch.backends.cudnn.deterministic,
                                       allow_tf32=self.precision=='tf32'):
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                yield
            finally:
                torch.backends.cuda.matmul.allow_tf32=previous

    def prepare_scenes(self,height_maps,origins=None):
        if height_maps.ndim!=3 or len(height_maps)==0 or tuple(height_maps.shape[-2:])!=(512,512):raise ValueError('height_maps must be nonempty [S,512,512], 25.6 metre local atlases')
        # Own immutable snapshots: a terrain refresh creates a new cache. Active
        # rollout/backward references continue to point to their original maps.
        with torch.no_grad(),self._precision_context(),self._autocast():
            maps=height_maps.detach().to(self.device,dtype=torch.float32).clone()
            tensors=tuple(t.detach() for t in self.core.encode(maps))
            origin=torch.zeros(len(maps),3,device=self.device) if origins is None else origins.detach().to(self.device,dtype=torch.float32).clone()
        if origin.shape!=(len(maps),3):raise ValueError('origins must be [S,3]')
        return SceneCache(tensors,origin,len(maps))

    def _block(self,cache,p,r,ids,keys):
        n=len(p)
        if n<self.chunk_size:
            pad=self.chunk_size-n
            p=torch.cat((p,p[-1:].expand(pad,-1)))
            r=torch.cat((r,r[-1:].expand(pad,-1,-1)))
            ids=torch.cat((ids,ids[-1:].expand(pad)))
            keys=torch.cat((keys,keys[-1:].expand(pad)))
        with self._precision_context(),self._autocast():
            result=self._fn(cache.tensors,cache.origins,p,r,ids,keys)[:n]
            # CUDA graph outputs use replay-owned storage. Own each block before
            # another replay can reuse it; ordinary compiled mode needs no copy.
            return result.clone() if self.compile_mode in ('reduce-overhead','max-autotune') else result

    def _chunks(self,cache,position,rotation,ids,keys):
        return torch.cat([self._block(cache,position[s:s+self.chunk_size],rotation[s:s+self.chunk_size],ids[s:s+self.chunk_size],keys[s:s+self.chunk_size]) for s in range(0,len(position),self.chunk_size)])

    def render(self,cache,position,rotation,scene_ids,keys,mode='recompute'):
        n=len(position)
        if position.shape!=(n,3) or rotation.shape!=(n,3,3) or scene_ids.shape!=(n,) or keys.shape!=(n,):raise ValueError('Expected position[N,3], rotation[N,3,3], scene_ids[N], noise keys[N]')
        if any(t.device!=cache.origins.device for t in (position,rotation,scene_ids,keys)):raise ValueError('Poses, IDs, keys and cache must share a device')
        if position.dtype!=torch.float32 or rotation.dtype!=torch.float32:raise ValueError('Metric positions and rotations must be float32')
        if scene_ids.dtype!=torch.int64 or keys.dtype!=torch.int64:raise ValueError('scene_ids and keys must be int64')
        if mode not in ('recompute','autograd','inference'):raise ValueError('mode must be recompute, autograd or inference')
        if mode!='inference' and self.compile_mode in ('reduce-overhead','max-autotune'):raise ValueError('CUDA graph profiles are inference-only; use default or max-autotune-no-cudagraphs for BPTT')
        if n==0:return position.new_empty((0,1,48,64))
        if mode=='inference':
            with torch.no_grad():return self._chunks(cache,position,rotation,scene_ids,keys)
        if mode=='recompute':return _Recompute.apply(self,cache,position,rotation,scene_ids,keys)
        if mode=='autograd':return self._chunks(cache,position,rotation,scene_ids,keys)
        raise ValueError('mode must be recompute, autograd or inference')

    def render_from_body(self,cache,body_position,body_rotation,scene_ids,keys,mode='recompute'):
        camera_position=body_position+(body_rotation@body_position.new_tensor(MOUNT)[None,:,None]).squeeze(-1)
        return self.render(cache,camera_position,optical_rotation(body_rotation),scene_ids,keys,mode)

    def warmup(self,cache,position,rotation,scene_ids,keys):
        # Run forward and VJP compilation before entering a timed RL loop.
        p=position[:self.chunk_size].detach().requires_grad_(True);r=rotation[:self.chunk_size].detach().requires_grad_(True)
        for _ in range(2):
            y=self.render(cache,p,r,scene_ids[:len(p)],keys[:len(p)],mode='recompute')
            torch.autograd.grad(y.mean(),(p,r))
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    def save_compiler_cache(self,path):
        artifacts=torch.compiler.save_cache_artifacts()
        if artifacts is None:return False
        Path(path).write_bytes(artifacts[0]);return True

    @staticmethod
    def load_compiler_cache(path):
        """Load only trusted cache files built with matching torch/CUDA/GPU versions."""
        return torch.compiler.load_cache_artifacts(Path(path).read_bytes())
