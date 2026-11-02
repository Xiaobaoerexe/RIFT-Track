"""Visual CTS encoders, dense action MoE and task-owned PopArt critics.

Student parameters belong only to distillation. Freezing their parameters must
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
        self.geometry=GeometryEncoder(128,mixed_precision=mixed_precision); self.fusion=nn.Sequential(nn.Linear(170,32),nn.LayerNorm(32))
    def forward(self,priv,height,task_ids):
        task=F.one_hot(task_ids.long().reshape(-1),10).to(priv.dtype)
        with network_autocast(priv,self.mixed_precision):
            projected=self.fusion[0](torch.cat((self.physical(priv),self.geometry(height),task),-1))
        return self.fusion[1](projected.float())


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
        self.teacher_encoder=Teacher(mixed_precision); self.student_encoder=Student(mixed_precision); self.actor=DenseActionMoE(hidden,experts,mixed_precision)
    def teacher_latent(self,priv,height,task_ids):
        return self.teacher_encoder(self.teacher_norm(priv),height,task_ids)
    def student_latent(self,hist,depth,hidden,reset=None,depth_features=None):
        if depth_features is None:depth_features=self.student_encoder.encode_depth(depth)
        return self.student_encoder.forward_features(self.history_norm(hist),self.single_norm(st.single_from_history(hist)),depth_features,hidden,reset)
    def step(self,hist,priv,branch,*,depth,hidden,height,task_ids,reset=None,depth_features=None):
        single=self.single_norm(st.single_from_history(hist))
        if isinstance(branch,tuple):
            n=branch[0]
            zt=self.teacher_latent(priv[:n],height[:n],task_ids[:n])
            zs,g,hs=self.student_latent(hist[n:],None if depth is None else depth[n:],hidden[n:],None if reset is None else reset[n:],
                None if depth_features is None else depth_features[n:])
            z=torch.cat((zt,zs)); nh=torch.cat((hidden[:n],hs))
        elif branch==0:
            z=self.teacher_latent(priv,height,task_ids); nh=hidden; g=None
        else: z,g,nh=self.student_latent(hist,depth,hidden,reset,depth_features)
        return self.actor(z,single),nh,z,g
    def forward(self,hist,priv,branch,*,depth,hidden,height,task_ids):
        return self.step(hist,priv,branch,depth=depth,hidden=hidden,height=height,task_ids=task_ids)[0]
    @torch.no_grad()
    def update_stats(self,hist,priv):
        self.history_norm.update(hist); self.single_norm.update(st.single_from_history(hist)); self.teacher_norm.update(priv)


class TaskValue(nn.Module):
    def __init__(self,physical_dim,hidden,mixed_precision=True):
        super().__init__();self.mixed_precision=mixed_precision
        self.net=mlp(physical_dim+64,hidden,2,nn.ELU)
    @property
    def output_layer(self): return self.net[-1]
    def forward(self,physical,geometry):
        with network_autocast(physical,self.mixed_precision):value=self.net(torch.cat((physical,geometry),-1))
        return value.float()


def _task_value_forward(net,physical,height):
    """One shared compilable call; task modules retain their original ownership."""
    return net(physical,height)


def _encode_critic_geometry(encoder,height):
    """Compile the shared CNN without wrapping or duplicating its parameters."""
    return encoder(height)


def _critic_geometry(pose,field):
    from .terrain import geometry_crop
    return geometry_crop(None,pose,field,kind='teacher')


class Critic(nn.Module):
    """Ten independent task values, each with teacher/student PopArt heads.

    Feature records contain physical state and task memory, never policy latents.
    Raw map inputs may be cached within one TD update; learned features never are.
    """
    def __init__(self,hidden=(256,128),field=None,context_dim=21,compile=False,mixed_precision=True):
        super().__init__(); self.field=field; self.context_dim=context_dim;self.mixed_precision=mixed_precision
        self.physical_dim=st.PRIV_DIM+context_dim
        self.feature_dim=self.physical_dim+7+2+1
        self.norm=RunningNorm(self.physical_dim)
        # One Critic-owned geometry encoder; no policy latent or policy-owned
        # encoder is used. Its features are rebuilt after every optimizer step.
        self.geometry=GeometryEncoder(64,mixed_precision=mixed_precision)
        self.tasks=nn.ModuleList([TaskValue(self.physical_dim,hidden,mixed_precision) for _ in range(10)])
        self.register_buffer('pop_mean',torch.zeros(10,2)); self.register_buffer('pop_second',torch.ones(10,2))
        self.register_buffer('pop_scale',torch.ones(10,2)); self.register_buffer('pop_count',torch.zeros(10,2))
        # Compile functions rather than wrapping submodules, preserving state_dict
        # keys and optimizer ownership. Dynamic batch sizes cover per-task splits.
        self._value_fn=torch.compile(_task_value_forward,dynamic=True,fullgraph=True) if compile else _task_value_forward
        self._encode_geometry_fn=torch.compile(_encode_critic_geometry,dynamic=True,fullgraph=True) if compile else _encode_critic_geometry
        self._geometry_fn=torch.compile(_critic_geometry,dynamic=True,fullgraph=True) if compile else _critic_geometry
    def features(self,priv,branch,task_ids,context,pose):
        tag=priv.new_zeros((len(priv),2))
        if isinstance(branch,tuple): tag[:branch[0],0]=1; tag[branch[0]:,1]=1
        else: tag[:,branch]=1
        return torch.cat((priv,context,pose,tag,task_ids.reshape(-1,1).to(priv.dtype)),-1)
    def decode(self,x):
        flat=x.reshape(-1,self.feature_dim)
        return flat,flat[:,:self.physical_dim],flat[:,self.physical_dim:self.physical_dim+7],flat[:,-3:-1].argmax(-1),flat[:,-1].long()
    def forward(self,x,normalized=False,geometry=None,geometry_indices=None):
        flat,physical,pose,branch,tasks=self.decode(x)
        if geometry is not None and x.requires_grad:
            raise ValueError('Detached geometry cache is forbidden on the Actor state-gradient path')
        norm=self.norm(physical)
        if geometry is None:height=self._geometry_fn(pose,self.field)
        else:
            take=torch.arange(len(flat),device=flat.device) if geometry_indices is None else geometry_indices
            height=geometry.gather(take)
        encoded=self._encode_geometry_fn(self.geometry,height)
        # One task-count transfer replaces ten CUDA nonzero synchronizations.
        # Stable sorting preserves the original within-task sample order exactly.
        order=torch.argsort(tasks,stable=True)
        counts=torch.bincount(tasks,minlength=len(self.tasks)).tolist()
        values=[];offset=0
        for task,(net,count) in enumerate(zip(self.tasks,counts)):
            if not count: continue
            ids=order[offset:offset+count];offset+=count
            val=self._value_fn(net,norm[ids],encoded[ids]).gather(1,branch[ids,None]).squeeze(1)
            if not normalized: val=val*self.pop_scale[task,branch[ids]]+self.pop_mean[task,branch[ids]]
            values.append(val)
        out=flat.new_zeros(len(flat))
        if values: out=out.index_copy(0,order,torch.cat(values))
        return out.reshape(x.shape[:-1])
    def forward_for_task(self,x,task,normalized=False,geometry=None,geometry_indices=None):
        """Execute an already grouped task block without redispatching its rows.

        The caller owns grouping within the original optimization minibatch.
        Parameters, normalization and teacher/student head selection are identical
        to forward; no task losses or optimizer updates are combined or dropped.
        """
        flat,physical,pose,branch,_=self.decode(x)
        if geometry is not None and x.requires_grad:
            raise ValueError('Detached geometry cache is forbidden on the Actor state-gradient path')
        if geometry is None:
            height=self._geometry_fn(pose,self.field)
        else:
            take=torch.arange(len(flat),device=x.device) if geometry_indices is None else geometry_indices
            height=geometry.gather(take)
        encoded=self._encode_geometry_fn(self.geometry,height)
        val=self._value_fn(self.tasks[task],self.norm(physical),encoded).gather(1,branch[:,None]).squeeze(1)
        if not normalized: val=val*self.pop_scale[task,branch]+self.pop_mean[task,branch]
        return val.reshape(x.shape[:-1])
    def loss_for_task(self,x,targets,task,geometry=None,geometry_indices=None):
        _,_,_,branch,_=self.decode(x)
        y=(targets.reshape(-1)-self.pop_mean[task,branch])/self.pop_scale[task,branch]
        value=self.forward_for_task(x,task,normalized=True,geometry=geometry,geometry_indices=geometry_indices)
        return F.smooth_l1_loss(value.float().reshape(-1),y.float())
    @torch.no_grad()
    def prepare_geometry(self,x,chunk_size=2048):
        """Ephemeral detached inputs, quantized at the CNN's BF16 boundary."""
        if x.requires_grad: raise ValueError('Only detached Critic training samples may be cached')
        from .critic_cache import CriticGeometryCache
        flat,physical,pose,_,_=self.decode(x)
        dtype=torch.bfloat16 if flat.is_cuda and self.mixed_precision else torch.float32
        cache=CriticGeometryCache(len(flat),flat.device,dtype,chunk_size)
        # Reserved-minus-allocated is not a promise of reusable contiguous space.
        # Issue bounded real allocations; do not authorize a giant request using
        # that optimistic sum, or flush the allocator every training iteration.
        free_before=torch.cuda.mem_get_info(flat.device)[0] if flat.is_cuda else None
        try:
            for begin in range(0,len(flat),chunk_size):
                crop=self._geometry_fn(pose[begin:begin+chunk_size],self.field)
                cache.append(crop)
                del crop
            cache.finish()
        except torch.cuda.OutOfMemoryError:
            cache.heights.clear();cache.validity.clear()
            raise
        self.geometry_cache_metadata={'bytes':cache.storage_bytes,'query_chunk':chunk_size,
            'height_dtype':str(dtype),'validity_dtype':'uint8 bit-packed, exact',
            'blocks':len(cache.heights),'device_free_before':free_before,
            'allocation_policy':'bounded allocations; reserved slack is not credited',
            'scope':'one detached Critic update; no learned feature caching'}
        return cache

    @torch.no_grad()
    def update_popart(self,x,targets,beta=.001):
        _,physical,_,branch,tasks=self.decode(x)
        targets=targets.reshape(-1)
        for task,net in enumerate(self.tasks):
            for b in range(2):
                ids=(tasks==task)&(branch==b)
                if not ids.any(): continue
                y=targets[ids]; alpha=1. if self.pop_count[task,b]==0 else 1-(1-beta)**len(y)
                old_mean=self.pop_mean[task,b].clone(); old_scale=self.pop_scale[task,b].clone()
                mean=(1-alpha)*old_mean+alpha*y.mean()
                second=(1-alpha)*self.pop_second[task,b]+alpha*y.square().mean()
                scale=(second-mean.square()).clamp_min(1e-4).sqrt()
                net.output_layer.weight[b].mul_(old_scale/scale)
                net.output_layer.bias[b].copy_((old_scale*net.output_layer.bias[b]+old_mean-mean)/scale)
                self.pop_mean[task,b]=mean; self.pop_second[task,b]=second; self.pop_scale[task,b]=scale; self.pop_count[task,b]+=len(y)
    def loss(self,x,targets,geometry=None,geometry_indices=None):
        _,_,_,branch,tasks=self.decode(x)
        y=(targets.reshape(-1)-self.pop_mean[tasks,branch])/self.pop_scale[tasks,branch]
        return F.smooth_l1_loss(self(x,normalized=True,geometry=geometry,geometry_indices=geometry_indices).float().reshape(-1),y.float())


class StudentExport(nn.Module):
    """Deployed recurrent policy: history, metric noisy depth and GRU memory."""
    def __init__(self,p):
        super().__init__(); self.history_norm=p.history_norm; self.single_norm=p.single_norm
        self.student_encoder=p.student_encoder; self.actor=p.actor
    def forward(self,hist,depth,hidden,reset):
        single=self.single_norm(st.single_from_history(hist))
        z,_,hidden=self.student_encoder(self.history_norm(hist),single,depth,hidden,reset)
        return self.actor(z,single),hidden
