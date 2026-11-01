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


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-4, clip=10.):
        super().__init__(); self.clip=clip
        self.register_buffer('mean',torch.zeros(dim)); self.register_buffer('var',torch.ones(dim))
        self.register_buffer('count',torch.tensor(eps))
    @torch.no_grad()
    def update(self,x):
        x=x.reshape(-1,x.shape[-1])
        if not x.shape[0]: return
        b_mean,b_var,b_n=x.mean(0),x.var(0,unbiased=False),x.shape[0]
        delta=b_mean-self.mean; total=self.count+b_n
        self.var.copy_((self.var*self.count+b_var*b_n+delta.square()*self.count*b_n/total)/total)
        self.mean.add_(delta*b_n/total); self.count.copy_(total)
    def forward(self,x): return ((x-self.mean)/(self.var.sqrt()+1e-6)).clamp(-self.clip,self.clip)
    def unnormalize(self,y): return y*(self.var.sqrt()+1e-6)+self.mean


def mlp(input_dim,hidden,out_dim,act=nn.SiLU):
    layers=[]
    for h in hidden: layers += [nn.Linear(input_dim,h),act()]; input_dim=h
    return nn.Sequential(*layers,nn.Linear(input_dim,out_dim))


class GeometryEncoder(nn.Module):
    """Metric relative height and validity; no flattening of a large raw map."""
    def __init__(self,out_dim=128):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(2,16,5,2,2),nn.ELU(),nn.Conv2d(16,32,3,2,1),nn.ELU(),
            nn.Conv2d(32,64,3,2,1),nn.ELU(),nn.AdaptiveAvgPool2d((3,4)),nn.Flatten(),nn.Linear(768,out_dim),nn.ELU())
    def forward(self,height): return self.net(height)


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.cnn=GeometryEncoder(128)
    def forward(self,depth):
        if depth.ndim==3: depth=depth[:,None]
        return self.cnn(torch.cat((depth/6.,(depth>0).to(depth.dtype)),1))


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe=MoE(8,450,(512,256,256),32,'elu')
        self.depth_encoder=DepthEncoder(); self.gru=nn.GRUCell(128+45,256)
        self.fusion=nn.Sequential(nn.Linear(288,32),nn.LayerNorm(32))
    def forward(self,history,single,depth,hidden,reset=None):
        if reset is not None: hidden=hidden*(~reset.bool()).to(hidden.dtype)[:,None]
        proprio,gates=self.moe(history)
        hidden=self.gru(torch.cat((self.depth_encoder(depth),single),-1),hidden)
        return self.fusion(torch.cat((proprio,hidden),-1)),gates,hidden


class Teacher(nn.Module):
    def __init__(self):
        super().__init__(); self.physical=MLP(st.PRIV_DIM,32,(512,256),'elu')
        self.geometry=GeometryEncoder(128); self.fusion=nn.Sequential(nn.Linear(170,32),nn.LayerNorm(32))
    def forward(self,priv,height,task_ids):
        task=F.one_hot(task_ids.long().reshape(-1),10).to(priv.dtype)
        return self.fusion(torch.cat((self.physical(priv),self.geometry(height),task),-1))


class DenseActionMoE(nn.Module):
    """Eight independent action experts; router sees only the fused latent."""
    def __init__(self,hidden=(512,256,128),experts=8):
        super().__init__(); self.experts=experts
        self.router=MLP(32,experts,(128,64),'elu')
        dims=(77,*hidden,12)
        self.weights=nn.ParameterList([nn.Parameter(torch.empty(experts,a,b)) for a,b in zip(dims[:-1],dims[1:])])
        self.biases=nn.ParameterList([nn.Parameter(torch.empty(experts,b)) for b in dims[1:]])
        for weight,bias in zip(self.weights,self.biases):
            bound=1/math.sqrt(weight.shape[1]); nn.init.uniform_(weight,-bound,bound); nn.init.uniform_(bias,-bound,bound)
    def forward(self,latent,single,return_gates=False):
        gates=self.router(latent).softmax(-1)
        x=torch.cat((latent,single),-1).unsqueeze(0).expand(self.experts,-1,-1)
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            x=torch.bmm(x,weight)+bias[:,None]
            if i<len(self.weights)-1: x=F.elu(x)
        action=(x.permute(1,0,2)*gates[:,:,None]).sum(1)
        return (action,gates) if return_gates else action


class Policy(nn.Module):
    def __init__(self,hidden=(512,256,128)):
        super().__init__()
        self.history_norm=RunningNorm(450); self.single_norm=RunningNorm(45); self.teacher_norm=RunningNorm(st.PRIV_DIM)
        self.teacher_encoder=Teacher(); self.student_encoder=Student(); self.actor=DenseActionMoE(hidden)
    def teacher_latent(self,priv,height,task_ids):
        return self.teacher_encoder(self.teacher_norm(priv),height,task_ids)
    def student_latent(self,hist,depth,hidden,reset=None):
        return self.student_encoder(self.history_norm(hist),self.single_norm(st.single_from_history(hist)),depth,hidden,reset)
    def step(self,hist,priv,branch,*,depth,hidden,height,task_ids,reset=None):
        single=self.single_norm(st.single_from_history(hist))
        if isinstance(branch,tuple):
            n=branch[0]
            zt=self.teacher_latent(priv[:n],height[:n],task_ids[:n])
            zs,g,hs=self.student_latent(hist[n:],depth[n:],hidden[n:],None if reset is None else reset[n:])
            z=torch.cat((zt,zs)); nh=torch.cat((hidden[:n],hs))
        elif branch==0:
            z=self.teacher_latent(priv,height,task_ids); nh=hidden; g=None
        else: z,g,nh=self.student_latent(hist,depth,hidden,reset)
        return self.actor(z,single),nh,z,g
    def forward(self,hist,priv,branch,*,depth,hidden,height,task_ids):
        return self.step(hist,priv,branch,depth=depth,hidden=hidden,height=height,task_ids=task_ids)[0]
    @torch.no_grad()
    def update_stats(self,hist,priv):
        self.history_norm.update(hist); self.single_norm.update(st.single_from_history(hist)); self.teacher_norm.update(priv)


class TaskValue(nn.Module):
    def __init__(self,physical_dim,hidden):
        super().__init__(); self.physical=mlp(physical_dim,(256,128),128,nn.ELU)
        self.geometry=GeometryEncoder(); self.net=mlp(256,hidden,2,nn.ELU)
    @property
    def output_layer(self): return self.net[-1]
    def forward(self,physical,height): return self.net(torch.cat((self.physical(physical),self.geometry(height)),-1))


def _task_value_forward(net,physical,height):
    """One shared compilable call; task modules retain their original ownership."""
    return net(physical,height)


def _critic_geometry(pose,field):
    from .terrain import geometry_crop
    return geometry_crop(None,pose,field,kind='teacher')


class Critic(nn.Module):
    """Ten independent task values, each with teacher/student PopArt heads.

    Feature records contain physical state and task memory, never policy latents.
    Raw map inputs may be cached within one TD update; learned features never are.
    """
    def __init__(self,hidden=(512,256,128),field=None,context_dim=21,compile=False):
        super().__init__(); self.field=field; self.context_dim=context_dim
        self.physical_dim=st.PRIV_DIM+context_dim
        self.feature_dim=self.physical_dim+7+2+1
        self.norm=RunningNorm(self.physical_dim)
        self.tasks=nn.ModuleList([TaskValue(self.physical_dim,hidden) for _ in range(10)])
        self.register_buffer('pop_mean',torch.zeros(10,2)); self.register_buffer('pop_second',torch.ones(10,2))
        self.register_buffer('pop_scale',torch.ones(10,2)); self.register_buffer('pop_count',torch.zeros(10,2))
        # Compile functions rather than wrapping submodules, preserving state_dict
        # keys and optimizer ownership. Dynamic batch sizes cover per-task splits.
        self._value_fn=torch.compile(_task_value_forward,dynamic=True,fullgraph=True) if compile else _task_value_forward
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
        from .terrain import geometry_crop
        flat,physical,pose,branch,tasks=self.decode(x)
        if geometry is not None and x.requires_grad:
            raise ValueError('Detached geometry cache is forbidden on the Actor state-gradient path')
        norm=self.norm(physical)
        # One task-count transfer replaces ten CUDA nonzero synchronizations.
        # Stable sorting preserves the original within-task sample order exactly.
        order=torch.argsort(tasks,stable=True)
        counts=torch.bincount(tasks,minlength=len(self.tasks)).tolist()
        values=[];offset=0
        for task,(net,count) in enumerate(zip(self.tasks,counts)):
            if not count: continue
            ids=order[offset:offset+count];offset+=count
            if geometry is None:
                height=geometry_crop(physical[ids,:st.STATE_DIM],pose[ids],self.field,kind='teacher')
            else:
                take=ids if geometry_indices is None else geometry_indices[ids]
                height=torch.stack((geometry[0][take],geometry[1][take].to(physical.dtype)),1)
            val=self._value_fn(net,norm[ids],height).gather(1,branch[ids,None]).squeeze(1)
            if not normalized: val=val*self.pop_scale[task,branch[ids]]+self.pop_mean[task,branch[ids]]
            values.append(val)
        out=flat.new_zeros(len(flat))
        if values: out=out.index_copy(0,order,torch.cat(values))
        return out.reshape(x.shape[:-1])
    @torch.no_grad()
    def prepare_geometry(self,x,chunk_size=2048):
        """Ephemeral exact input cache for TD fitting, never learned features."""
        if x.requires_grad: raise ValueError('Only detached Critic training samples may be cached')
        flat,physical,pose,_,_=self.decode(x)
        cache_bytes=len(flat)*64*128*(flat.element_size()+1)
        if flat.is_cuda:
            free,_=torch.cuda.mem_get_info(flat.device)
            reusable=max(0,torch.cuda.memory_reserved(flat.device)-torch.cuda.memory_allocated(flat.device))
            # Keep space for Critic activations/gradients and optimizer kernels.
            workspace=free+reusable-cache_bytes-2*1024**3
            bytes_per_row=64*128*128  # conservative temporary bilinear-query workspace
            allowed=int(workspace//bytes_per_row)
            if allowed<64:
                raise MemoryError(f'Exact Critic terrain cache needs {cache_bytes/1024**3:.2f}GiB; '
                    f'available including reusable allocator blocks {(free+reusable)/1024**3:.2f}GiB; '
                    'insufficient safe query/fitting workspace')
            chunk_size=min(chunk_size,allowed)
        self.geometry_cache_metadata={'bytes':cache_bytes,'query_chunk':chunk_size,
            'height_dtype':str(flat.dtype),'validity_dtype':'bool','scope':'one Critic update; no learned feature caching'}
        heights=flat.new_empty((len(flat),64,128))
        valid=torch.empty((len(flat),64,128),device=flat.device,dtype=torch.bool)
        for begin in range(0,len(flat),chunk_size):
            end=min(begin+chunk_size,len(flat))
            crop=self._geometry_fn(pose[begin:end],self.field)
            heights[begin:end]=crop[:,0];valid[begin:end]=crop[:,1].bool()
        return heights,valid

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
        return F.smooth_l1_loss(self(x,normalized=True,geometry=geometry,geometry_indices=geometry_indices).reshape(-1),y)


class StudentExport(nn.Module):
    """Deployed recurrent policy: history, metric noisy depth and GRU memory."""
    def __init__(self,p):
        super().__init__(); self.history_norm=p.history_norm; self.single_norm=p.single_norm
        self.student_encoder=p.student_encoder; self.actor=p.actor
    def forward(self,hist,depth,hidden,reset):
        single=self.single_norm(st.single_from_history(hist))
        z,_,hidden=self.student_encoder(self.history_norm(hist),single,depth,hidden,reset)
        return self.actor(z,single),hidden
