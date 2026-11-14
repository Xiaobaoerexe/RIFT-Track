"""Frozen, fused geometry/noise path. FP32 geometry, autocast CNNs."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.geometry_reference import DepthGenerator
from .vendor.noise_reference import DepthNoise
from .vendor.camera import NEAR,FAR


def lookup(flat,height,width,ids,xy):
    p=(xy/12.8+1)*.5*xy.new_tensor([width-1,height-1])
    p=torch.minimum(torch.maximum(p,torch.zeros_like(p)),p.new_tensor([width-1.0001,height-1.0001]))
    ix=p[...,0].floor().long();iy=p[...,1].floor().long()
    dx=(p[...,0]-ix)[...,None].to(flat.dtype);dy=(p[...,1]-iy)[...,None].to(flat.dtype)
    index=ids.reshape(len(ids),*([1]*(ix.ndim-1)))*height*width+iy*width+ix
    return (flat[index]*(1-dx)+flat[index+1]*dx)*(1-dy)+(flat[index+width]*(1-dx)+flat[index+width+1]*dx)*dy


def mix32(x):
    x=x & 0xffffffff
    x=((x^(x>>16))*0x45d9f3b)&0xffffffff
    x=((x^(x>>16))*0x45d9f3b)&0xffffffff
    return x^(x>>16)


def noise_keys(trajectory_ids,frame_index,seed=317,episode_ids=0):
    """Explicit per-observation keys. Identical keys reproduce forward/backward noise."""
    return mix32(trajectory_ids.long()*0x9e3779b1+frame_index*0x85ebca6b+episode_ids*0xc2b2ae35+seed)


def fields_from_keys(keys,counter=None):
    # Stateless Gaussian samples, independent of chunk boundaries and global RNG.
    if counter is None:
        counter=torch.arange(4*54*70,device=keys.device,dtype=torch.int64)[None]
    # Signed spelling is identical modulo 2**32 and avoids an Inductor/Triton
    # index-expression int32 literal overflow for 0x9e3779b1.
    base=keys[:,None]+counter*(-1640531535)
    a=mix32(base);b=mix32(base+0x68bc21eb)
    u=((a>>8).float()+.5)/16777216.;v=((b>>8).float()+.5)/16777216.
    normals=(torch.sqrt(-2*torch.log(u))*torch.cos(2*math.pi*v)).reshape(-1,4,54,70)
    fields=torch.stack((normals[:,0,:48,:64],(F.avg_pool2d(normals[:,1:2,:50,:66],3,1)*3)[:,0],(F.avg_pool2d(normals[:,2:3],7,1)*7)[:,0]),1)
    return fields,normals[:,3,:48,:64]


class FrozenCore(nn.Module):
    def __init__(self,geometry,noise,hole_strength=2.,hard=True,optimized=True,scan='product'):
        super().__init__();self.geometry=geometry;self.noise=noise
        self.hole_strength=hole_strength;self.hard=hard;self.optimized=optimized
        self.scan=scan
        # Materialized int64 counters prevent Inductor's affine-index folding
        # from narrowing large hash coefficients to int32.
        self.register_buffer('noise_counter',torch.arange(4*54*70,dtype=torch.int64)[None],persistent=False)
        self.eval();self.requires_grad_(False)

    def encode(self,maps):
        local,context=self.geometry.encode(maps)
        flat=lambda x:x.permute(0,2,3,1).contiguous().reshape(-1,x.shape[1])
        lf,cf=flat(local),flat(context)
        # CHW views share the prepacked storage, avoiding duplicate cached features.
        lv=lf.view(len(maps),128,128,8).permute(0,3,1,2)
        cv=cf.view(len(maps),64,64,8).permute(0,3,1,2)
        return maps,lv,cv,flat(maps[:,None]),lf,cf

    def geometry_cached(self,cache,ids,position,rotation):
        g=self.geometry
        if not self.optimized:
            return g(cache[0],ids,position,rotation,encoded=(cache[1],cache[2]))
        height,local,context=cache[3:6];batch=len(position)
        with torch.autocast(device_type=position.device.type,enabled=False):
            direction=torch.einsum('bij,rj->bri',rotation.float(),g.ray.float())
            t=g.depth_samples.float()
            xy=position.float()[:,None,None,:2]+direction[:,:,None,:2]*t[None,None,:,None]
            z_ray=position.float()[:,None,None,2]+direction[:,:,None,2]*t[None,None,:]
            clearance=z_ray-lookup(height,512,512,ids,xy)[...,0]
            alpha=torch.sigmoid(-clearance/.035)
            survival=1-alpha+1e-7
            if self.scan=='log':
                prefix=torch.cumsum(survival.log(),-1)
                trans=torch.cat((torch.zeros_like(prefix[...,:1]),prefix[...,:-1]),-1).exp()
                background=prefix[...,-1].exp().clamp(0,1)
            else:
                trans=torch.cumprod(torch.cat((torch.ones_like(alpha[...,:1]),survival),-1),-1)[...,:-1]
                background=survival.prod(-1).clamp(0,1)
            weights=alpha*trans
            prev=torch.cat((clearance[...,:1],clearance[...,:-1]),-1)
            den=prev-clearance;safe=torch.where(den.abs()>1e-5,den,torch.ones_like(den))
            frac=(prev/safe).clamp(0,1)
            candidate=t[None,None,:]-(1-frac)*(FAR-NEAR)/(g.samples-1)
            candidate=torch.cat((torch.full_like(candidate[...,:1],NEAR),candidate[...,1:]),-1)
            depth=(weights*candidate).sum(-1)+background*FAR
            hit=(1-background).clamp(1e-5,1-1e-5)
            local_t=(depth[...,None]+depth.new_tensor([-.12,0.,.12])).clamp(NEAR,FAR)
            local_xyz=position.float()[:,None,None,:]+direction[:,:,None,:]*local_t[...,None]
            local_xy=local_xyz[...,:2]
            local_clear=local_xyz[...,2]-lookup(height,512,512,ids,local_xy)[...,0]
            rd=direction/(direction.norm(dim=-1,keepdim=True)+1e-8)
        context_features=torch.cat((lookup(local,128,128,ids,local_xy),lookup(context,64,64,ids,local_xy)),-1).flatten(-2)
        image=torch.cat((depth[...,None]/FAR,hit[...,None],rd,local_clear,context_features.float()),-1)
        image=image.transpose(1,2).reshape(batch,56,48,64)
        a=g.im1(image);c=g.im2(a);d=g.im3(c)
        e=g.up2(torch.cat((F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c),1))
        e=g.up1(torch.cat((F.interpolate(e,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        output=g.output(e).float()
        return (depth.reshape(batch,48,64)+.5*output[:,0].tanh()).clamp(NEAR,FAR),torch.logit(hit.reshape(batch,48,64))+output[:,1]

    def forward(self,cache,origins,position,rotation,scene_ids,keys):
        z,logit=self.geometry_cached(cache,scene_ids,position-origins[scene_ids],rotation)
        probability=logit.sigmoid()
        geometry_valid=(logit>0).float().detach()-probability.detach()+probability if self.hard else probability
        site=torch.ones_like(scene_ids)
        fields,epsilon=fields_from_keys(keys,self.noise_counter)
        logits,sigma=self.noise.parameters_at(z,site)
        depth,_,_=self.noise.corrupt(z,site,logits+math.log(self.hole_strength),sigma,fields,epsilon,geometry_valid,self.hard,.5)
        return depth[:,None]
