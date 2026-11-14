"""One-pass conditional geometry renderer with sparse surface features and image refinement.

During actor optimization freeze parameters, not autograd: pose derivatives remain live.
No Isaac, renderer, or simulation imports are required to use this model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .camera import rays,NEAR,FAR


class Block(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,stride,padding=1),nn.GroupNorm(4,cout),nn.SiLU(),
                               nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU())
    def forward(self,x):return self.net(x)


def query_map(m,ids,xy):
    """Bilinear gather without expanding a full map for every camera/ray."""
    s,c,h,w=m.shape;xy=xy.float()
    p=(xy/12.8+1)*.5*xy.new_tensor([w-1,h-1])
    p=torch.minimum(torch.maximum(p,torch.zeros_like(p)),p.new_tensor([w-1.0001,h-1.0001]))
    ix=p[...,0].floor().long();iy=p[...,1].floor().long();dx=(p[...,0]-ix)[...,None].to(m.dtype);dy=(p[...,1]-iy)[...,None].to(m.dtype)
    flat=m.permute(0,2,3,1).reshape(s*h*w,c)
    offset=ids.reshape(len(ids),*([1]*(ix.ndim-1)))*h*w
    i=offset+iy*w+ix
    return (flat[i]*(1-dx)+flat[i+1]*dx)*(1-dy)+(flat[i+w]*(1-dx)+flat[i+w+1]*dx)*dy


class DepthGenerator(nn.Module):
    def __init__(self,samples=32):
        super().__init__();self.samples=samples
        self.enc1=Block(1,16,2);self.enc2=Block(16,32,2);self.enc3=Block(32,64,2)
        self.local=nn.Conv2d(32,8,1);self.context=nn.Conv2d(64,8,1)
        self.im1=Block(56,32);self.im2=Block(32,64,2);self.im3=Block(64,96,2)
        self.up2=Block(160,64);self.up1=Block(96,32);self.output=nn.Conv2d(32,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('ray',rays().reshape(-1,3))
        self.register_buffer('depth_samples',torch.linspace(NEAR,FAR,samples))

    def encode(self,maps):
        a=self.enc1(maps[:,None]);b=self.enc2(a);c=self.enc3(b)
        return self.local(b),self.context(c)

    def forward(self,maps,scene_ids,position,rotation,encoded=None):
        if encoded is None:encoded=self.encode(maps)
        b=len(position)
        # Geometry stays float32 under autocast; rays are parameterized by optical Z-depth.
        with torch.autocast(device_type=position.device.type,enabled=False):
            direction=torch.einsum('bij,rj->bri',rotation.float(),self.ray.float())
            t=self.depth_samples.float()
            xyz=position.float()[:,None,None,:]+direction[:,:,None,:]*t[None,None,:,None]
            xy=xyz[...,:2]
            h=query_map(maps[:,None],scene_ids,xy)[...,0]
            clearance=xyz[...,2]-h
        with torch.autocast(device_type=position.device.type,enabled=False):
            alpha=torch.sigmoid(-clearance/.035)
            trans=torch.cumprod(torch.cat((torch.ones_like(alpha[...,:1]),1-alpha+1e-7),-1),-1)[...,:-1]
            weights=alpha*trans
            background=(1-alpha+1e-7).prod(-1).clamp(0,1)
            # Sub-bin surface estimate supplies a useful geometric initialization.
            prev=torch.cat((clearance[...,:1],clearance[...,:-1]),-1)
            den=prev-clearance
            safe=torch.where(den.abs()>1e-5,den,torch.ones_like(den))
            frac=(prev/safe).clamp(0,1)
            candidate=t[None,None,:]-(1-frac)*(FAR-NEAR)/(self.samples-1)
            candidate[...,0]=NEAR
            depth=(weights*candidate).sum(-1)+background*FAR
            hit=(1-background).clamp(1e-5,1-1e-5)
            # Expensive learned features are queried around the first surface only,
            # not at all 32 mostly empty-space samples. Geometry remains differentiable.
            local_t=(depth[...,None]+depth.new_tensor([-.12,0.,.12])).clamp(NEAR,FAR)
            local_xyz=position.float()[:,None,None,:]+direction[:,:,None,:]*local_t[...,None]
            local_xy=local_xyz[...,:2]
            local_clear=local_xyz[...,2]-query_map(maps[:,None],scene_ids,local_xy)[...,0]
            rd=direction/(direction.norm(dim=-1,keepdim=True)+1e-8)
        context=torch.cat([query_map(e,scene_ids,local_xy) for e in encoded],-1).flatten(-2)
        image=torch.cat((depth[...,None]/FAR,hit[...,None],rd,local_clear,context.float()),-1)
        image=image.transpose(1,2).reshape(b,56,48,64)
        a=self.im1(image);c=self.im2(a);d=self.im3(c)
        e=self.up2(torch.cat((F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c),1))
        e=self.up1(torch.cat((F.interpolate(e,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        out=self.output(e).float()
        z=(depth.reshape(b,48,64)+.5*out[:,0].tanh()).clamp(NEAR,FAR)
        logit=torch.logit(hit.reshape(b,48,64))+out[:,1]
        return z,logit

    def freeze(self):
        self.eval()
        for p in self.parameters():p.requires_grad_(False)
        return self
