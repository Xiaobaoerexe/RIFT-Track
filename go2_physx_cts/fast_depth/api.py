"""Frozen direct-depth candidate followed by the unchanged sensor-noise model."""
from dataclasses import dataclass
from pathlib import Path
import math
import torch
from torch import nn
from .model import DirectDepth,ModelConfig
from .source import import_package,ieee_fp32,sha256,package_fingerprints


@dataclass(frozen=True)
class FastSceneCache:
    embedding:torch.Tensor
    origins:torch.Tensor
    scene_count:int


class _Core(nn.Module):
    def __init__(self,model,noise,precision,hole_strength):
        super().__init__();self.model=model;self.noise=noise;self.precision=precision;self.hole_strength=hole_strength
        self.register_buffer('noise_counter',torch.arange(4*54*70,dtype=torch.int64)[None],persistent=False)
        self.eval();self.requires_grad_(False)

    def forward(self,embedding,origins,position,rotation,ids,keys):
        from go2_depth_input.core import fields_from_keys
        with torch.autocast(device_type=position.device.type,dtype=torch.bfloat16,enabled=self.precision=='bf16'):
            z,logit=self.model.decode(embedding[ids],position-origins[ids],rotation)
        # Existing noise coefficients, keys and hard-forward/smooth-backward
        # convention stay unchanged; BF16 is restricted to the new decoder.
        with torch.autocast(device_type=position.device.type,enabled=False):
            z,logit=z.float(),logit.float();probability=logit.sigmoid()
            valid=(logit>0).float().detach()-probability.detach()+probability
            site=torch.ones_like(ids);fields,epsilon=fields_from_keys(keys,self.noise_counter)
            logits,sigma=self.noise.parameters_at(z,site)
            noisy,_,_=self.noise.corrupt(z,site,logits+math.log(self.hole_strength),sigma,fields,epsilon,valid,True,.5)
        return noisy[:,None]


class _Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,engine,cache,p,r,ids,keys):
        ctx.engine,ctx.cache=engine,cache
        ctx.save_for_backward(p,r,ids,keys,cache.embedding,cache.origins)
        ctx.set_materialize_grads(False)
        return engine._chunks(cache,p,r,ids,keys)

    @staticmethod
    def backward(ctx,gradient):
        if gradient is None:return (None,)*6
        p,r,ids,keys,*_=ctx.saved_tensors;dp=torch.empty_like(p);dr=torch.empty_like(r)
        with torch.enable_grad():
            for start in range(0,len(p),ctx.engine.chunk_size):
                end=min(start+ctx.engine.chunk_size,len(p))
                pp=p[start:end].detach().requires_grad_();rr=r[start:end].detach().requires_grad_()
                y=ctx.engine._block(ctx.cache,pp,rr,ids[start:end],keys[start:end])
                with ieee_fp32(p.device):
                    gp,gr=torch.autograd.grad(y,(pp,rr),gradient[start:end],create_graph=False)
                dp[start:end]=gp;dr[start:end]=gr
        return None,None,dp,dr,None,None


class FastDepthGenerator:
    def __init__(self,weights,depth_package,device='cuda:0',chunk_size=256,compile_mode='default',precision='fp32',require_accepted=True):
        package=import_package(depth_package);self.device=torch.device(device)
        if precision not in ('fp32','bf16'):raise ValueError('Only explicitly measured FP32/BF16 profiles are supported')
        if self.device.type=='cpu' and precision!='fp32':raise ValueError('CPU validation uses FP32')
        ck=torch.load(weights,map_location='cpu',weights_only=False)
        if ck.get('schema')!='direct_depth_v1':raise ValueError('Not a direct-depth checkpoint')
        if require_accepted and not ck.get('acceptance',{}).get('passed',False):
            raise ValueError('Candidate has not passed the independent observation acceptance')
        if require_accepted and precision not in ck['acceptance'].get('accepted_precisions',[]):
            raise ValueError(f'Precision profile {precision} has not passed independent image/VJP acceptance')
        if compile_mode not in ('off','default','max-autotune-no-cudagraphs'):
            raise ValueError('Use off/default/max-autotune-no-cudagraphs; CUDA graph replay is not accepted here')
        self.source=package.DepthInputGenerator(device=str(self.device),compile_mode='off',precision='fp32')
        expected=ck['source_weights_sha256'];actual=sha256(Path(depth_package)/'go2_depth_input/assets/weights.pt')
        if expected!=actual:raise ValueError('Original geometry/noise package weights changed')
        if ck.get('source_code_sha256') and ck['source_code_sha256']!=package_fingerprints(depth_package):
            raise ValueError('Pinned source observation code changed after candidate training')
        model=DirectDepth(ModelConfig(**ck['model_config'])).to(self.device)
        model.load_state_dict(ck['model']);model.freeze()
        self.core=_Core(model,self.source.core.noise,precision,2.).to(self.device)
        self.core.to(memory_format=torch.channels_last)
        self.chunk_size=int(chunk_size);self.precision=precision
        if self.chunk_size<=0:raise ValueError('Positive chunk size required')
        self._fn=self.core if compile_mode=='off' else torch.compile(self.core,mode=compile_mode,dynamic=False,fullgraph=True)
        self.metadata={'schema':'direct_depth_v1','checkpoint_sha256':sha256(weights),
            'source_weights_sha256':actual,'precision':precision,'frozen':True,
            'output_shape':[1,48,64],'metric_range':[.1,6.],
            'acceptance':ck.get('acceptance',{}),'noise':'unchanged frozen package; explicit keys; hole odds2',
            'geometry':'direct pose-conditioned decoder; no per-pixel ray marching'}

    def prepare_scenes(self,maps,origins=None):
        if maps.ndim!=3 or tuple(maps.shape[-2:])!=(512,512) or not len(maps):raise ValueError('Expected nonempty [S,512,512] atlases')
        embedding=[]
        with torch.no_grad(),ieee_fp32(self.device):
            for start in range(0,len(maps),16):
                source=self.source.prepare_scenes(maps[start:start+16])
                features=DirectDepth.scene_inputs(source.tensors)
                embedding.append(self.core.model.encode_scenes(features).float())
        embedding=torch.cat(embedding).detach()
        origins=embedding.new_zeros((len(maps),3)) if origins is None else origins.detach().to(self.device,dtype=torch.float32).clone()
        if origins.shape!=(len(maps),3):raise ValueError('Expected scene origins[S,3]')
        return FastSceneCache(embedding,origins,len(maps))

    def _block(self,cache,p,r,ids,keys):
        count=len(p)
        if count<self.chunk_size:
            n=self.chunk_size-count
            p=torch.cat((p,p[-1:].expand(n,-1)));r=torch.cat((r,r[-1:].expand(n,-1,-1)))
            ids=torch.cat((ids,ids[-1:].expand(n)));keys=torch.cat((keys,keys[-1:].expand(n)))
        with ieee_fp32(self.device):return self._fn(cache.embedding,cache.origins,p,r,ids,keys)[:count]

    def _chunks(self,cache,p,r,ids,keys):
        return torch.cat([self._block(cache,p[s:s+self.chunk_size],r[s:s+self.chunk_size],ids[s:s+self.chunk_size],keys[s:s+self.chunk_size]) for s in range(0,len(p),self.chunk_size)])

    def render(self,cache,position,rotation,scene_ids,keys,mode='recompute'):
        n=len(position)
        if position.shape!=(n,3) or rotation.shape!=(n,3,3) or scene_ids.shape!=(n,) or keys.shape!=(n,):raise ValueError('Invalid pose/id/key shapes')
        if position.dtype!=torch.float32 or rotation.dtype!=torch.float32 or scene_ids.dtype!=torch.int64 or keys.dtype!=torch.int64:raise ValueError('Metric poses must be FP32; IDs/keys must be int64')
        if any(t.device!=cache.embedding.device for t in (position,rotation,scene_ids,keys)):raise ValueError('Inputs/cache device mismatch')
        if not n:return position.new_empty((0,1,48,64))
        if mode=='inference':
            with torch.no_grad():return self._chunks(cache,position,rotation,scene_ids,keys)
        if mode=='recompute':return _Recompute.apply(self,cache,position,rotation,scene_ids,keys)
        if mode=='autograd':return self._chunks(cache,position,rotation,scene_ids,keys)
        raise ValueError('Expected inference/recompute/autograd mode')

    def render_from_body(self,cache,body_position,body_rotation,scene_ids,keys,mode='recompute'):
        from go2_depth_input.vendor.camera import MOUNT,optical_rotation
        p=body_position+(body_rotation@body_position.new_tensor(MOUNT)[None,:,None]).squeeze(-1)
        return self.render(cache,p,optical_rotation(body_rotation),scene_ids,keys,mode)

    def warmup(self,cache,p,r,ids,keys):
        n=min(len(p),self.chunk_size)
        for _ in range(2):
            pp=p[:n].detach().requires_grad_();rr=r[:n].detach().requires_grad_()
            y=self.render(cache,pp,rr,ids[:n],keys[:n],mode='recompute')
            torch.autograd.grad(y.mean(),(pp,rr))
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
