"""Task-separated velocity grids using Rapid Locomotion's weighted-cell sampler.

Grid/within-cell sampling and neighbor expansion are ports of
Improbable-AI/rapid-locomotion-rl@f5143ef940e934849c00284e34caf164d6ce7b6e,
mini_gym/envs/base/curriculum.py. Per-terrain macro statistics, downgrade,
phase caps and .5/.75 thresholds are the explicitly approved project adaptation.
"""
import math
import torch
from .task_allocation import BASE_COUNTS
SOURCE_COMMIT='f5143ef940e934849c00284e34caf164d6ce7b6e'
CMD_STATE_DIM=20
REQUEST,COMMAND=slice(0,3),slice(3,6)
TIMER,ACCUMULATION,EPISODE_ELAPSED=6,slice(7,9),9
LAST_LIMIT,COMMAND_COUNTER,MAX_DISTANCE=10,11,12
ORIGIN,RANGE_STAGE,LIMIT_STAGE=slice(13,15),15,16
GRID_BIN,TASK_ID,TERRAIN_ID=17,18,19
COMMAND_NAMES=('lin_vel_x','lin_vel_y','ang_vel_yaw')
TERRAIN_NAMES=('wave','slope_up','slope_down','rough_slope','stairs_up','stairs_down','obstacles','flat','wheels','small_squares','gap','indoor_obstacles','obstacle_up','obstacle_down')
FINAL_TASK_COUNTS=BASE_COUNTS
TASK_TERRAINS=((7,1,2,3),(7,),(0,6,8,9),(4,),(5,),(10,),(7,0,1,2,3,4,5,6,8,9),(11,),(12,),(13,))

def phase(iteration):return 0 if iteration<2500 else 1 if iteration<5000 else 2 if iteration<8000 else 3
def task_assignments(n,iteration,device):
    active=1 if iteration<5000 else 5 if iteration<8000 else 10
    weights=torch.tensor(FINAL_TASK_COUNTS[:active],dtype=torch.float64)
    counts=torch.floor(weights/weights.sum()*n).long()
    remainder=n-int(counts.sum());fractions=weights/weights.sum()*n-counts
    if remainder:counts[torch.argsort(fractions,descending=True)[:remainder]]+=1
    return torch.repeat_interleave(torch.arange(active,device=device),counts.to(device))
def speed_cap(task,iteration):
    if task==0:
        return .5 if iteration<5000 else 1. if iteration<10000 else 1.5 if iteration<20000 else 2. if iteration<30000 else 2.5 if iteration<35000 else 3.
    if task==1:return 1.5 if iteration<10000 else 2.5 if iteration<20000 else 4. if iteration<30000 else 5.
    return 3.
def zero_command_prob(iteration):return min(max(float(iteration)/1500.,0.),1.)*.1

def coupled_clip(command):
    x,y,w=command.unbind(-1);fast=x.abs()>2.
    return torch.stack((x,torch.where(fast,y.clamp(-.5,.5),y),torch.where(fast,w.clamp(-1.,1.),w)),-1)

class VelocityGrid:
    """Checkpointed per-task grid, training-data-only per-terrain statistics."""
    def __init__(self,device='cpu'):
        axes=(torch.arange(-10,11,device=device)*.5,torch.arange(-5,6,device=device)*.2,torch.arange(-10,11,device=device)*.1)
        self.shape=tuple(len(a) for a in axes)
        self.grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        self.width=self.grid.new_tensor([.5,.2,.1]);g=len(self.grid)
        self.weights=torch.zeros(10,g,device=device);self.extent=self.grid.new_tensor([[.5,.2,.1]]*10)
        self.extent[1,0]=1.5
        self.extent[[5,8,9],0]=1.
        initial=(self.grid.abs()<=self.extent[0]+1e-6).all(-1)
        self.weights[:,initial]=1.;self.weights[[5,8,9]]=(((self.grid[:,0]>=1.)&(self.grid[:,0]<=1.)&(self.grid[:,1:].abs()<=self.extent[0,1:]+1e-6).all(-1)).float())
        self.weights[1,(self.grid[:,0]>=1.)&(self.grid[:,0]<=1.5)&(self.grid[:,1:].abs()<=self.extent[1,1:]+1e-6).all(-1)]=1.
        self.sum=torch.zeros(10,14,g,device=device);self.count=torch.zeros_like(self.sum)
        self.last_update=0;self.last_statistics={}
    def bin_indices(self,command):
        ix=torch.round(command/self.width).long()+command.new_tensor([10,5,10],dtype=torch.long)
        ix=torch.maximum(torch.minimum(ix,ix.new_tensor([20,10,20])),torch.zeros_like(ix))
        return (ix[:,0]*11+ix[:,1])*21+ix[:,2]
    def state_dict(self):return {k:getattr(self,k).clone() for k in ('weights','extent','sum','count')}|{'last_update':self.last_update,'statistics':self.last_statistics}
    def load_state_dict(self,s):
        for k in ('weights','extent','sum','count'):getattr(self,k).copy_(s[k])
        self.last_update=int(s['last_update']);self.last_statistics=s.get('statistics',{})
    @torch.no_grad()
    def record(self,task,terrain,bins,progress,own):
        mask=own&(task!=6)&(terrain>=0)&(terrain<14)&torch.isfinite(progress)&(bins>=0)
        idx=(task[mask]*14+terrain[mask])*len(self.grid)+bins[mask]
        self.sum.view(-1).scatter_add_(0,idx,progress[mask].clamp(0,1));self.count.view(-1).scatter_add_(0,idx,torch.ones_like(progress[mask]))
    @torch.no_grad()
    def update(self,iteration):
        if iteration<self.last_update+1000:return {}
        result={}
        for t in range(10):
            if t==6:continue
            c=self.count[t];seen=c>0;means=self.sum[t]/c.clamp_min(1)
            # Standing/low-speed cells cannot alone unlock a new fast frontier.
            frontier=(self.grid[:,0].abs()>=max(.25,float(self.extent[t,0])-.25))
            tested=c*frontier[None];perterrain=(self.sum[t]*frontier[None]).sum(-1)/tested.sum(-1).clamp_min(1);terrains=tested.sum(-1)>0
            if not bool(terrains.any()):continue
            score=perterrain[terrains].mean();old=self.extent[t].clone()
            if score>.75:self.extent[t]+=self.width
            elif score<=.5:self.extent[t]=torch.maximum(self.extent[t]-self.width,self.width)
            self.extent[t,0].clamp_(max=speed_cap(t,iteration));self.extent[t,1].clamp_(max=1.);self.extent[t,2].clamp_(max=1.)
            if t in (5,8,9):self.extent[t,0].clamp_(min=1.,max=1.5 if iteration<20000 else 2.)
            if t==1:self.extent[t,0].clamp_(min=min(speed_cap(1,iteration),max(1.5,float(self.extent[0,0])+.5)))
            # Same-cell learning remains available; successful neighboring cells
            # gain weight. Failure retracts only the outer envelope, not low bins.
            macro=(means*seen).sum(0)/seen.sum(0).clamp_min(1);success=(seen.any(0))&(macro>.75)
            w=self.weights[t].reshape(self.shape);good=success.reshape(self.shape)
            neighbors=torch.nn.functional.max_pool3d(good.float()[None,None],3,stride=1,padding=1)[0,0].bool()
            w.add_(neighbors*.2).clamp_(max=1.)
            self.weights[t,(self.grid.abs()<=self.width+1e-6).all(-1)]=1.
            result[str(t)]={'macro_progress':float(score),'terrain_progress':{TERRAIN_NAMES[i]:float(perterrain[i]) for i in terrains.nonzero().flatten().tolist()},'old_extent':old.tolist(),'extent':self.extent[t].tolist()}
        self.sum.zero_();self.count.zero_();self.last_update=int(iteration);self.last_statistics=result
        return result
    def sample(self,task,terrain,iteration,generator=None):
        # Ten shared distributions, rather than B copies of the entire grid.
        key=(int(iteration),int(self.last_update))
        if getattr(self,'_cache_key',None)!=key:
            caps=self.grid.new_tensor([speed_cap(t,iteration) for t in range(10)])
            extent=self.extent.clone();extent[:,0]=torch.minimum(extent[:,0],caps)
            extent[[5,8,9],0]=extent[[5,8,9],0].clamp(min=1.,max=1.5 if iteration<20000 else 2.)
            extent[[5,8,9],1:]=extent.new_tensor([.15,.1])
            enabled=(self.grid[None].abs()<=extent[:,None]+1e-6).all(-1)
            lower=self.extent[0,0].clamp(max=speed_cap(0,iteration))
            enabled[1]&=self.grid[:,0]>=lower
            for t in (3,4,5,7,8,9):enabled[t]&=self.grid[:,0]>=0.
            for t in (5,8,9):enabled[t]&=self.grid[:,0]>=1.
            weight=self.weights*enabled
            empty=weight.sum(-1)==0;weight=torch.where(empty[:,None],enabled.float(),weight)
            self._sample_cdf=weight.cumsum(-1)/weight.sum(-1,keepdim=True).clamp_min(1e-9)
            self._sample_extent=extent;self._sample_lower=lower;self._cache_key=key
        n=len(task);u=torch.rand(10,n,device=task.device,generator=generator)
        choices=torch.searchsorted(self._sample_cdf.contiguous(),u.contiguous()).clamp_max(len(self.grid)-1)
        ix=choices[task,torch.arange(n,device=task.device)]
        extent=self._sample_extent[task];sample=self.grid[ix]+(torch.rand(n,3,device=task.device,generator=generator)-.5)*self.width
        sample=torch.maximum(torch.minimum(sample,extent),-extent)
        lower=torch.where(task==1,self._sample_lower.expand_as(sample[:,0]),-extent[:,0])
        forward=(task==3)|(task==4)|(task==5)|(task==7)|(task==8)|(task==9)
        lower=torch.where(forward,torch.full_like(lower,.2),lower)
        special=(task==5)|(task==8)|(task==9)
        lower=torch.where(special,torch.full_like(lower,1.),lower)
        vx=torch.maximum(sample[:,0],lower)
        vx=torch.where(special,vx.clamp(min=1.,max=1.5 if iteration<20000 else 2.),vx)
        vy=torch.where(special,sample[:,1].clamp(-.15,.15),sample[:,1])
        wz=torch.where(special,sample[:,2].clamp(-.1,.1),sample[:,2])
        membership=torch.nn.functional.one_hot(task,10)
        rank=(membership.cumsum(0)*membership).sum(-1)-1;totals=membership.sum(0).clamp_min(1)
        offset=(torch.rand(10,device=task.device,generator=generator)*totals).long()
        pure=((rank-offset[task])%totals[task])<torch.ceil(.3*totals[task]).long()
        vy=torch.where(pure,torch.zeros_like(vy),vy);wz=torch.where(pure,torch.zeros_like(wz),wz)
        vx=torch.where(task==6,torch.full_like(vx,.5),vx);vy=torch.where(task==6,torch.zeros_like(vy),vy);wz=torch.where(task==6,torch.zeros_like(wz),wz)
        return coupled_clip(torch.stack((vx,vy,wz),-1)),ix

class OfficialCommands:
    """Functional command state shared by real and imagined rollout; task grids own sampling."""
    resampling_time=5.;num_steps_per_iter=1;zero_command_prob=staticmethod(zero_command_prob)
    def __init__(self,dt=.02,episode_seconds=25.,terrain_lookup=None):
        self.dt,self.episode_seconds=float(dt),float(episode_seconds);self.terrain_lookup=terrain_lookup;self.grid=None
    def _grid(self,device):
        if self.grid is None:self.grid=VelocityGrid(device)
        return self.grid
    def command(self,s):return s[:,COMMAND]
    def timer(self,s):return s[:,TIMER:TIMER+1]
    def initial(self,position_xy,origins_xy,iteration=0,*,task_id=None,terrain_id=None,**kw):
        s=position_xy.new_zeros(len(position_xy),CMD_STATE_DIM)
        if task_id is not None:s[:,TASK_ID]=task_id
        if terrain_id is not None:s[:,TERRAIN_ID]=terrain_id
        return self.reset(s,torch.ones(len(s),device=s.device,dtype=torch.bool),position_xy,origins_xy,iteration,**kw)
    def reset(self,s,mask,position_xy,origins_xy,iteration,*,task_id=None,terrain_id=None,generator=None,random=None):
        out=s.clone();out[mask,:17]=0.;out[mask,ORIGIN]=origins_xy[mask,:2]
        if task_id is not None:out[mask,TASK_ID]=task_id[mask].to(out.dtype)
        if terrain_id is not None:out[mask,TERRAIN_ID]=terrain_id[mask].to(out.dtype)
        return self._resample(out,mask,iteration,generator)
    def _resample(self,s,mask,iteration,generator=None):
        # Sampling draws are independent of states: no detach on useful state paths.
        sampled,bins=self._grid(s.device).sample(s[:,TASK_ID].long(),s[:,TERRAIN_ID].long(),iteration,generator)
        zeros=(torch.rand(len(s),device=s.device,generator=generator)<zero_command_prob(iteration))&(s[:,TASK_ID]==0)
        sampled=torch.where(zeros[:,None],torch.zeros_like(sampled),sampled);bins=self.grid.bin_indices(sampled)
        request=torch.where(mask[:,None],sampled,s[:,REQUEST]);out=s.clone();out[:,REQUEST]=request;out[:,COMMAND]=request
        out[:,TIMER]=torch.where(mask,torch.full_like(s[:,TIMER],self.resampling_time),s[:,TIMER])
        out[:,ACCUMULATION]=s[:,ACCUMULATION]+torch.where(mask[:,None],request[:,:2],torch.zeros_like(request[:,:2]))
        out[:,COMMAND_COUNTER]=s[:,COMMAND_COUNTER]+mask;out[:,RANGE_STAGE]=phase(iteration)
        out[:,GRID_BIN]=torch.where(mask,bins,s[:,GRID_BIN]);return out
    def advance(self,s,position_xy,iteration,*,generator=None,random=None,task_ids=None,terrain_ids=None):
        out=s.clone();out[:,TIMER]=s[:,TIMER]-self.dt;out[:,EPISODE_ELAPSED]=s[:,EPISODE_ELAPSED]+self.dt
        out=self._resample(out,out[:,TIMER]<=0,iteration,generator)
        out[:,MAX_DISTANCE]=torch.maximum(s[:,MAX_DISTANCE],(position_xy[:,:2]-s[:,ORIGIN]).norm(dim=-1));return out
    def state_dict(self):return None if self.grid is None else self.grid.state_dict()
    def load_state_dict(self,s):
        if s is not None:self._grid(s['weights'].device).load_state_dict(s)
