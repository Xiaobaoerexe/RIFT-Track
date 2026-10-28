"""Source RunningNorm/MLP and retained Go2RLLab MoE-CTS parameter ownership."""
import torch
from torch import nn
from .moe import MLP, MoE, L2Norm
from . import state as st

class RunningNorm(nn.Module):
    def __init__(self,dim,eps=1e-4,clip=10.):
        super().__init__();self.clip=clip
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('var',torch.ones(dim))
        self.register_buffer('count',torch.tensor(eps))
    @torch.no_grad()
    def update(self,x):
        x=x.reshape(-1,x.shape[-1])
        if not x.shape[0]:return
        b_mean,b_var,b_n=x.mean(0),x.var(0,unbiased=False),x.shape[0]
        delta=b_mean-self.mean;tot=self.count+b_n
        self.mean+=delta*b_n/tot
        self.var=(self.var*self.count+b_var*b_n+delta**2*self.count*b_n/tot)/tot
        self.count=tot
    def forward(self,x):return ((x-self.mean)/(self.var.sqrt()+1e-6)).clamp(-self.clip,self.clip)
    def unnormalize(self,y):return y*(self.var.sqrt()+1e-6)+self.mean

def mlp(input_dim,hidden,out_dim,act=nn.SiLU):
    layers=[]
    for h in hidden:layers += [nn.Linear(input_dim,h),act()];input_dim=h
    return nn.Sequential(*layers,nn.Linear(input_dim,out_dim))

class Student(nn.Module):
    def __init__(self):
        super().__init__();self.moe=MoE(8,450,(512,256,256),32,'elu');self.norm_layer=L2Norm()
    def forward(self,x):
        z,g=self.moe(x);return self.norm_layer(z),g

class Policy(nn.Module):
    def __init__(self,hidden=(512,256,128)):
        super().__init__()
        self.history_norm=RunningNorm(450);self.single_norm=RunningNorm(45);self.teacher_norm=RunningNorm(st.PRIV_DIM)
        self.teacher_encoder=nn.Sequential(MLP(st.PRIV_DIM,32,(512,256),'elu'),L2Norm())
        self.student_encoder=Student();self.actor=MLP(77,12,hidden,'elu')
    def forward(self,hist,priv,branch):
        if isinstance(branch,tuple):
            split=branch[0]
            z=torch.cat((self.teacher_encoder(self.teacher_norm(priv[:split])),self.student_encoder(self.history_norm(hist[split:]))[0]),0)
        else:z=self.teacher_encoder(self.teacher_norm(priv)) if branch==0 else self.student_encoder(self.history_norm(hist))[0]
        return self.actor(torch.cat((z,self.single_norm(st.single_from_history(hist))),-1))
    @torch.no_grad()
    def update_stats(self,hist,priv):
        self.history_norm.update(hist);self.single_norm.update(st.single_from_history(hist));self.teacher_norm.update(priv)

class Critic(nn.Module):
    def __init__(self,hidden=(512,256,128)):
        super().__init__();self.norm=RunningNorm(st.PRIV_DIM+2);self.net=mlp(st.PRIV_DIM+2,hidden,1,nn.ELU)
    def forward(self,x):return self.net(self.norm(x)).squeeze(-1)
    @staticmethod
    def features(priv,branch):
        tag=priv.new_zeros((priv.shape[0],2))
        if isinstance(branch,tuple):tag[:branch[0],0]=1;tag[branch[0]:,1]=1
        else:tag[:,branch]=1
        return torch.cat((priv,tag),-1)

class StudentExport(nn.Module):
    """Stateless450 history input,12 raw actions; no privileged inputs."""
    def __init__(self,p):
        super().__init__();self.history_norm=p.history_norm;self.single_norm=p.single_norm
        self.student_encoder=p.student_encoder;self.actor=p.actor
    def forward(self,hist):
        z,_=self.student_encoder(self.history_norm(hist))
        return self.actor(torch.cat((z,self.single_norm(st.single_from_history(hist))),-1))
