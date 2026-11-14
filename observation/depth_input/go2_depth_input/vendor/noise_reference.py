"""Continuous small depth noise and correlated holes, with explicit surrogate gradients."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),
                                 nn.Conv2d(b,b,3,padding=1),nn.SiLU())
    def forward(self,x):
        return self.net(x)


class DepthNoise(nn.Module):
    """Input metres. One means hole in the internal corruption mask.

    The continuous error is <=20 mm. Hard masks use a biased Concrete
    straight-through derivative; soft mode has the actual continuous derivative.
    Camera/site calibrations are separate from the shared spatial predictor.
    """
    def __init__(self):
        super().__init__()
        self.a=Block(5,16); self.b=Block(16,32); self.c=Block(32,64)
        self.u=Block(96,32); self.v=Block(48,16); self.out=nn.Conv2d(16,2,1)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad(): self.out.bias.copy_(torch.tensor([-3.,-1.5]))
        # D435i shared prior; local D455 calibration modifies only these small tensors.
        self.local_adjust=nn.Parameter(torch.zeros(2))
        self.corr_logits=nn.Parameter(torch.tensor([[1.5,0.,-1.],[1.5,0.,-1.]]))
        self.register_buffer('xy',torch.stack(torch.meshgrid(torch.linspace(-1,1,48),torch.linspace(-1,1,64),indexing='ij'))[None])

    def parameters_at(self,depth,site):
        d=depth[:,None].float().clamp(.1,6.)
        # No observed hole mask is provided to the predictor during fitting.
        dx=F.pad(d[:,:,:,1:]-d[:,:,:,:-1],(0,1,0,0))
        dy=F.pad(d[:,:,1:,:]-d[:,:,:-1,:],(0,0,0,1))
        x=torch.cat((d/6.,(.2/d),dx.clamp(-1,1),dy.clamp(-1,1),self.xy[:,0:1].expand(len(d),-1,-1,-1)),1)
        a=self.a(x); b=self.b(F.avg_pool2d(a,2)); c=self.c(F.avg_pool2d(b,2))
        u=self.u(torch.cat((F.interpolate(c,scale_factor=2,mode='bilinear',align_corners=False),b),1))
        v=self.v(torch.cat((F.interpolate(u,scale_factor=2,mode='bilinear',align_corners=False),a),1))
        out=self.out(v).float()
        adjust=site[:,None,None,None].float()*self.local_adjust[None,:,None,None]
        logits=out[:,0:1]+adjust[:,0:1]
        sigma=.0005+.0095*torch.sigmoid(out[:,1:2]+adjust[:,1:2])
        return logits[:,0],sigma[:,0]

    @staticmethod
    def random_fields(batch,device,generator=None):
        # Valid convolutions of independent Gaussian fields have unit variance.
        fields=[]
        for k in (1,3,7):
            n=torch.randn(batch,1,48+k-1,64+k-1,device=device,generator=generator)
            fields.append((F.avg_pool2d(n,k,stride=1)*k)[:,0])
        return torch.stack(fields,1),torch.randn(batch,48,64,device=device,generator=generator)

    def corrupt(self,depth,site,logits,sigma,fields,epsilon,geometry_valid=None,hard=True,temperature=.5):
        weights=self.corr_logits[site.long()].softmax(-1)
        normal=(fields*weights[:,:,None,None]).sum(1)/(weights.square().sum(-1).sqrt()[:,None,None])
        uniform=(.5*(1+torch.erf(normal/math.sqrt(2)))).clamp(1e-5,1-1e-5)
        threshold=torch.logit(uniform)
        soft=torch.sigmoid((logits-threshold)/temperature)
        holes=(soft>=.5).float().detach()-soft.detach()+soft if hard else soft
        # Small inverse-depth residual, then an explicit absolute metric bound.
        d=depth.float().clamp(.1,6.)
        inv=1/d+(sigma/d.square())*epsilon
        candidate=inv.clamp_min(1/20.).reciprocal()
        delta=.020*torch.tanh((candidate-d)/.020)
        measured=(d+delta).clamp(.1,6.)
        valid=torch.ones_like(d) if geometry_valid is None else geometry_valid
        return measured*(1-holes)*valid,holes,delta

    def forward(self,depth,site,geometry_valid=None,random=None,hard=True,temperature=.5,hole_strength=1.):
        """hole_strength multiplies hole odds; 1.0 reproduces the trained model.

        Only hole logits change. Continuous noise and the spatial random fields
        are unchanged. This is augmentation intensity, not sensor recalibration.
        """
        if not math.isfinite(hole_strength) or hole_strength<=0:
            raise ValueError('hole_strength must be a finite positive scalar')
        logits,sigma=self.parameters_at(depth,site)
        logits=logits+math.log(hole_strength)
        fields,eps=self.random_fields(len(depth),depth.device) if random is None else random
        noisy,holes,delta=self.corrupt(depth,site,logits,sigma,fields,eps,geometry_valid,hard,temperature)
        return dict(depth=noisy,holes=holes,delta=delta,logits=logits,sigma=sigma)

    def freeze(self):
        self.eval()
        for p in self.parameters(): p.requires_grad_(False)
        return self
