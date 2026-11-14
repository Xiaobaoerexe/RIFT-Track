"""Visual CTS encoders and selectable plain/dense-MoE shared actor.

The shared depth CNN belongs to WM reconstruction; other student parameters
belong to distillation. Freezing their parameters must
not disable autograd through observations or recurrent state during imagination.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .moe import MLP, MoE
from . import state as st
from .precision import network_autocast


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-4, clip=10.):
        super().__init__(); self.clip=clip
        self.register_buffer('mean',torch.zeros(dim)); self.register_buffer('var',torch.ones(dim))
        self.register_buffer('count',torch.tensor(eps))
    @torch.no_grad()
    def update(self,x):
        x=x.float().reshape(-1,x.shape[-1])
        if not x.shape[0]: return
        b_mean,b_var,b_n=x.mean(0),x.var(0,unbiased=False),x.shape[0]
        delta=b_mean-self.mean; total=self.count+b_n
        self.var.copy_((self.var*self.count+b_var*b_n+delta.square()*self.count*b_n/total)/total)
        self.mean.add_(delta*b_n/total); self.count.copy_(total)
    def forward(self,x): return ((x.float()-self.mean)/(self.var.sqrt()+1e-6)).clamp(-self.clip,self.clip)
    def unnormalize(self,y): return y.float()*(self.var.sqrt()+1e-6)+self.mean


def mlp(input_dim,hidden,out_dim,act=nn.SiLU):
    layers=[]
    for h in hidden: layers += [nn.Linear(input_dim,h),act()]; input_dim=h
    return nn.Sequential(*layers,nn.Linear(input_dim,out_dim))


class GeometryEncoder(nn.Module):
    """Metric relative height and validity; no flattening of a large raw map."""
    def __init__(self,out_dim=128,channels=(8,16,32),first_kernel=3,mixed_precision=True):
        super().__init__();self.mixed_precision=mixed_precision
        a,b,c=channels
        self.net=nn.Sequential(nn.Conv2d(2,a,first_kernel,2,first_kernel//2),nn.ELU(),nn.Conv2d(a,b,3,2,1),nn.ELU(),
            nn.Conv2d(b,c,3,2,1),nn.ELU(),nn.AdaptiveAvgPool2d((3,4)),nn.Flatten(),nn.Linear(c*12,out_dim),nn.ELU())
    def forward(self,height):
        height=height.float()
        with network_autocast(height,self.mixed_precision):encoded=self.net(height)
        return encoded.float()


class DepthEncoder(nn.Module):
    def __init__(self,mixed_precision=True):
        super().__init__(); self.cnn=GeometryEncoder(128,channels=(16,32,64),first_kernel=5,mixed_precision=mixed_precision)
    def forward(self,depth):
        if depth.ndim==3: depth=depth[:,None]
        depth=depth.float()
        return self.cnn(torch.cat((depth/6.,(depth>0).to(depth.dtype)),1))


class Student(nn.Module):
    def __init__(self,mixed_precision=True):
        super().__init__();self.mixed_precision=mixed_precision
        self.moe=MoE(8,450,(512,256,256),32,'elu')
        self.depth_encoder=DepthEncoder(mixed_precision); self.gru=nn.GRUCell(128+45,256)
        self.fusion=nn.Sequential(nn.Linear(288,32),nn.LayerNorm(32))
    def distillation_parameters(self):
        """CNN has exactly one owner: the WM optimizer's reconstruction loss."""
        yield from self.moe.parameters()
        yield from self.gru.parameters()
        yield from self.fusion.parameters()
    def encode_proprio(self,history):
        with network_autocast(history,self.mixed_precision):
            logits=self.moe.gating_network[0](history);experts=self.moe.experts(history)
        gates=logits.float().softmax(-1)
        return (experts.float()*gates[:,:,None]).sum(1),gates
    def encode_depth(self,depth):
        return self.depth_encoder(depth)
    def advance_memory(self,single,depth_features,hidden,reset=None):
        if reset is not None: hidden=hidden*(~reset.bool()).to(hidden.dtype)[:,None]
        with network_autocast(single,self.mixed_precision):
            updated=self.gru(torch.cat((depth_features,single),-1),hidden)
        return updated.float()
    def fuse_features(self,proprio,single,depth_features,hidden,reset=None):
        hidden=self.advance_memory(single,depth_features,hidden,reset)
        with network_autocast(hidden,self.mixed_precision):projected=self.fusion[0](torch.cat((proprio,hidden),-1))
        return self.fusion[1](projected.float()),hidden
    def forward_features(self,history,single,depth_features,hidden,reset=None):
        proprio,gates=self.encode_proprio(history)
        z,hidden=self.fuse_features(proprio,single,depth_features,hidden,reset)
        return z,gates,hidden
    def forward(self,history,single,depth,hidden,reset=None):
        return self.forward_features(history,single,self.encode_depth(depth),hidden,reset)


class Teacher(nn.Module):
    def __init__(self,mixed_precision=True):
        super().__init__();self.mixed_precision=mixed_precision
        self.physical=MLP(st.PRIV_DIM,32,(512,256),'elu')
        self.geometry=GeometryEncoder(128,mixed_precision=mixed_precision); self.fusion=nn.Sequential(nn.Linear(160,32),nn.LayerNorm(32))
    def forward(self,priv,height,task_ids=None):
        # Intent must be visible in commands/observations, never hidden task IDs.
        with network_autocast(priv,self.mixed_precision):
            projected=self.fusion[0](torch.cat((self.physical(priv),self.geometry(height)),-1))
        return self.fusion[1](projected.float())


class PlainActor(nn.Module):
    """AS2 baseline MLP, with no redundant one-expert router or action squashing."""
    def __init__(self,hidden=(512,256,128),mixed_precision=True):
        super().__init__();self.mixed_precision=mixed_precision;self.experts=1
        self.net=MLP(77,12,hidden,'elu')
    def forward(self,latent,single,return_gates=False):
        with network_autocast(latent,self.mixed_precision):
            action=self.net(torch.cat((latent,single),-1))
        action=action.float()
        return (action,action.new_ones((len(action),1))) if return_gates else action


class DenseActionMoE(nn.Module):
    """Three dense independent action experts; router sees the fused latent."""
    def __init__(self,hidden=(512,256,128),experts=3,mixed_precision=True):
        super().__init__(); self.experts=experts;self.mixed_precision=mixed_precision
        self.router=MLP(32,experts,(128,64),'elu')
        dims=(77,*hidden,12)
        self.weights=nn.ParameterList([nn.Parameter(torch.empty(experts,a,b)) for a,b in zip(dims[:-1],dims[1:])])
        self.biases=nn.ParameterList([nn.Parameter(torch.empty(experts,b)) for b in dims[1:]])
        for weight,bias in zip(self.weights,self.biases):
            bound=1/math.sqrt(weight.shape[1]); nn.init.uniform_(weight,-bound,bound); nn.init.uniform_(bias,-bound,bound)
    def forward(self,latent,single,return_gates=False):
        with network_autocast(latent,self.mixed_precision):logits=self.router(latent)
        gates=logits.float().softmax(-1)
        with network_autocast(latent,self.mixed_precision):
            x=torch.cat((latent,single),-1).unsqueeze(0).expand(self.experts,-1,-1)
            for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
                x=torch.bmm(x,weight);x=x+bias.to(x.dtype)[:,None]
                if i<len(self.weights)-1: x=F.elu(x)
        action=(x.float().permute(1,0,2)*gates[:,:,None]).sum(1)
        return (action,gates) if return_gates else action


class Policy(nn.Module):
    def __init__(self,hidden=(512,256,128),experts=3,mixed_precision=True):
        super().__init__();self.mixed_precision=mixed_precision
        self.history_norm=RunningNorm(450); self.single_norm=RunningNorm(45); self.teacher_norm=RunningNorm(st.PRIV_DIM)
        self.teacher_encoder=Teacher(mixed_precision); self.student_encoder=Student(mixed_precision)
        if experts not in (1,3):raise ValueError('Supported shared actors: plain MLP (1) or dense MoE (3)')
        self.actor=PlainActor(hidden,mixed_precision) if experts==1 else DenseActionMoE(hidden,experts,mixed_precision)
    def teacher_latent(self,priv,height,task_ids=None):
        return self.teacher_encoder(self.teacher_norm(priv),height,task_ids)
    def student_latent(self,hist,depth,hidden,reset=None,depth_features=None):
        if depth_features is None:depth_features=self.student_encoder.encode_depth(depth)
        return self.student_encoder.forward_features(self.history_norm(hist),self.single_norm(st.single_from_history(hist)),depth_features,hidden,reset)
    def step(self,hist,priv,branch,*,depth,hidden,height,task_ids=None,reset=None,depth_features=None):
        single=self.single_norm(st.single_from_history(hist))
        if isinstance(branch,tuple):
            n=branch[0]
            zt=self.teacher_latent(priv[:n],height[:n])
            zs,g,hs=self.student_latent(hist[n:],None if depth is None else depth[n:],hidden[n:],None if reset is None else reset[n:],
                None if depth_features is None else depth_features[n:])
            z=torch.cat((zt,zs)); nh=torch.cat((hidden[:n],hs))
        elif branch==0:
            z=self.teacher_latent(priv,height,task_ids); nh=hidden; g=None
        else: z,g,nh=self.student_latent(hist,depth,hidden,reset,depth_features)
        return self.actor(z,single),nh,z,g
    def forward(self,hist,priv,branch,*,depth,hidden,height,task_ids=None):
        return self.step(hist,priv,branch,depth=depth,hidden=hidden,height=height,task_ids=task_ids)[0]
    @torch.no_grad()
    def update_stats(self,hist,priv):
        self.history_norm.update(hist); self.single_norm.update(st.single_from_history(hist)); self.teacher_norm.update(priv)


class StudentExport(nn.Module):
    """Deployed recurrent policy: history, metric noisy depth and GRU memory."""
    def __init__(self,p):
        super().__init__(); self.history_norm=p.history_norm; self.single_norm=p.single_norm
        self.student_encoder=p.student_encoder; self.actor=p.actor
    def forward(self,hist,depth,hidden,reset):
        single=self.single_norm(st.single_from_history(hist))
        z,_,hidden=self.student_encoder(self.history_norm(hist),single,depth,hidden,reset)
        return self.actor(z,single),hidden


def __getattr__(name):
    # Backwards import spelling only; Critic has a single implementation in value.py.
    if name=='Critic':
        from .value import Critic
        return Critic
    raise AttributeError(name)
