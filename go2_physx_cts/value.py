"""Independent AS2 MLP task values for total future reward; no event channel."""
import torch
from torch import nn
from . import state as st
from .networks import RunningNorm, mlp
from .task import TASK_CONTEXT_DIM, GOAL, ORIGIN, ROUTE, ROUTE_YAW, PREVIOUS_COMMAND, CHANGE_AGE, CHANGE_STRENGTH

class Critic(nn.Module):
    def __init__(self,hidden=(512,256,128),field=None,context_dim=TASK_CONTEXT_DIM,
                 compile=False,mixed_precision=False,variant='simple',task_count=None):
        super().__init__();self.variant=variant;self.task_count=1 if variant=='simple' else 4
        if task_count is not None and task_count!=self.task_count:raise ValueError('Critic task count mismatch')
        self.context_dim=5 if variant=='simple' else TASK_CONTEXT_DIM+14
        self.physical_dim=st.PRIV_DIM+self.context_dim
        self.feature_dim=self.physical_dim+3
        self.norms=nn.ModuleList([RunningNorm(self.physical_dim+2) for _ in range(self.task_count)])
        self.nets=nn.ModuleList([mlp(self.physical_dim+2,hidden,1,nn.ELU) for _ in range(self.task_count)])

    @staticmethod
    def _tags(priv,branch):
        tag=priv.new_zeros((len(priv),2))
        if isinstance(branch,tuple):tag[:branch[0],0]=1.;tag[branch[0]:,1]=1.
        elif torch.is_tensor(branch):tag.scatter_(1,branch.reshape(-1,1).long(),1.)
        else:tag[:,int(branch)]=1.
        return tag

    def features(self,priv,branch,task_ids=None,context=None,pose=None,contacts=None):
        if context is None:raise ValueError('Value requires reward memory')
        if self.variant=='simple':
            canonical=context[:,PREVIOUS_COMMAND.start:CHANGE_STRENGTH+1]
            tasks=priv.new_zeros((len(priv),1))
        else:
            if pose is None or contacts is None or task_ids is None:raise ValueError('Task value requires complete physical context')
            canonical=context.clone();yaw=st.yaw_from_pose(pose)
            canonical[:,GOAL]=st.yaw_inverse(context[:,GOAL]-pose[:,:3],yaw)
            d=torch.cat((pose[:,:2]-context[:,ORIGIN],pose.new_zeros(len(pose),1)),-1)
            canonical[:,ORIGIN]=st.yaw_inverse(d,yaw)[:,:2]
            route=torch.cat((context[:,ROUTE]-pose[:,:2],pose.new_zeros(len(pose),1)),-1)
            canonical[:,ROUTE]=st.yaw_inverse(route,yaw)[:,:2]
            canonical[:,ROUTE_YAW]=torch.atan2(torch.sin(context[:,ROUTE_YAW]-yaw),torch.cos(context[:,ROUTE_YAW]-yaw))
            canonical=torch.cat((canonical,contacts),-1);tasks=task_ids.reshape(-1,1).to(priv.dtype)
        return torch.cat((priv,canonical,tasks,self._tags(priv,branch)),-1)

    def norm_input(self,x):return torch.cat((x[...,:self.physical_dim],x[...,-2:]),-1)

    @torch.no_grad()
    def update_stats(self,x):
        for task,norm in enumerate(self.norms):norm.update(self.norm_input(x[x[...,-3]==task]))

    def forward(self,x):
        shape=x.shape[:-1];flat=x.reshape(-1,self.feature_dim).float();out=flat.new_zeros(len(flat))
        for task,(norm,net) in enumerate(zip(self.norms,self.nets)):
            ids=(flat[:,-3]==task).nonzero().flatten()
            if ids.numel():out=out.index_copy(0,ids,net(norm(self.norm_input(flat[ids])))[:,0])
        return out.reshape(shape)
