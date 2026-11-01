"""Compact image refiner; pinned full-resolution geometric projection is retained."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
REFINER_NAMES = ('im1','im2','im3','up2','up1','output')

@dataclass(frozen=True)
class ModelConfig:
    width: int = 16

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(cin,cout,3,stride,1),nn.GroupNorm(4,cout),nn.SiLU(),
                               nn.Conv2d(cout,cout,3,1,1),nn.GroupNorm(4,cout),nn.SiLU())
    def forward(self,x):return self.net(x)

def refiner_output(model,image):
    a=model.im1(image);c=model.im2(a);d=model.im3(c)
    e=model.up2(torch.cat((F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c),1))
    e=model.up1(torch.cat((F.interpolate(e,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
    return model.output(e).float()

class SurfaceRefiner(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or ModelConfig();w=self.config.width
        if w%4:raise ValueError('Width must be divisible by four')
        self.im1=Block(56,w);self.im2=Block(w,2*w,2);self.im3=Block(2*w,3*w,2)
        self.up2=Block(5*w,2*w);self.up1=Block(3*w,w);self.output=nn.Conv2d(w,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,image):return refiner_output(self,image)
    def configuration(self):return asdict(self.config)
    def install(self,geometry):
        for name in REFINER_NAMES:setattr(geometry,name,getattr(self,name))
    def freeze(self):
        self.eval();self.requires_grad_(False);return self

def decode(output,base,hit):
    return (base+.5*output[:,0].tanh()).clamp(.1,6.),torch.logit(hit)+output[:,1]

def surface_features(core,cache,ids,position,rotation):
    """Exact offline split of source geometry_cached before its image refiner.

    Production still executes the original FrozenCore; this split is checked
    against its complete output before supervised fitting.
    """
    from go2_depth_input.core import lookup
    g=core.geometry;height,local,context=cache[3:6];n=len(position)
    with torch.autocast(device_type=position.device.type,enabled=False):
        direction=torch.einsum('bij,rj->bri',rotation.float(),g.ray.float());t=g.depth_samples.float()
        xy=position.float()[:,None,None,:2]+direction[:,:,None,:2]*t[None,None,:,None]
        zray=position.float()[:,None,None,2]+direction[:,:,None,2]*t[None,None,:]
        clearance=zray-lookup(height,512,512,ids,xy)[...,0]
        alpha=torch.sigmoid(-clearance/.035);prefix=torch.cumsum((1-alpha+1e-7).log(),-1)
        trans=torch.cat((torch.zeros_like(prefix[...,:1]),prefix[...,:-1]),-1).exp()
        background=prefix[...,-1].exp().clamp(0,1);weights=alpha*trans
        prev=torch.cat((clearance[...,:1],clearance[...,:-1]),-1)
        den=prev-clearance;safe=torch.where(den.abs()>1e-5,den,torch.ones_like(den));frac=(prev/safe).clamp(0,1)
        candidate=t[None,None,:]-(1-frac)*5.9/(g.samples-1)
        candidate=torch.cat((torch.full_like(candidate[...,:1],.1),candidate[...,1:]),-1)
        base=(weights*candidate).sum(-1)+background*6.;hit=(1-background).clamp(1e-5,1-1e-5)
        lt=(base[...,None]+base.new_tensor([-.12,0.,.12])).clamp(.1,6.)
        xyz=position.float()[:,None,None,:]+direction[:,:,None,:]*lt[...,None];lxy=xyz[...,:2]
        clear=xyz[...,2]-lookup(height,512,512,ids,lxy)[...,0];rd=direction/(direction.norm(dim=-1,keepdim=True)+1e-8)
    features=torch.cat((lookup(local,128,128,ids,lxy),lookup(context,64,64,ids,lxy)),-1).flatten(-2)
    image=torch.cat((base[...,None]/6.,hit[...,None],rd,clear,features.float()),-1)
    return image.transpose(1,2).reshape(n,56,48,64),base.reshape(n,48,64),hit.reshape(n,48,64)
