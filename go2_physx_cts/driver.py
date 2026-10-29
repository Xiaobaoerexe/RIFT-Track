"""Pinned upstream MoE-CTS driver adapted only to our simulator state API."""
import torch
from . import state as st
from .networks import Policy

def official_default(device):
    return torch.tensor([(.1 if leg[1]=='L' else -.1) if j=='hip' else (.8 if leg[0]=='F' else 1.) if j=='thigh' else -1.5 for leg in st.LEGS for j in ('hip','thigh','calf')],device=device)

class MoeCtsDriver:
    def __init__(self,env,path):
        self.env=env;self.policy=torch.jit.load(path,map_location=env.device).eval()
        self.default=official_default(env.device)
        self.hist=torch.zeros(env.num_envs,450,device=env.device)
        self.last=torch.zeros(env.num_envs,12,device=env.device)
        if tuple(self.policy.feature_dims)!=st.TERM_DIMS or int(self.policy.history_len)!=10:
            raise ValueError('Official driver observation contract mismatch')
    @torch.no_grad()
    def act(self,command):
        s=self.env.snapshot()['s'].clone();s[:,st.Q]+=self.env.consts.default_q-self.default
        one=st.frame(s,self.last,command);self.hist=st.push_history(self.hist,one)
        p=self.policy;z,_=p.student_moe_encoder(p.actor_obs_normalizer(self.hist))
        self.last=p.actor(torch.cat((z,p.single_obs_normalizer(one)),-1))
        return (self.default+.25*self.last-self.env.consts.default_q)/self.env.consts.action_scale
    def reset(self,done):self.hist[done]=0;self.last[done]=0

class FlatExpert:
    def __init__(self,path,device):
        ck=torch.load(path,map_location=device,weights_only=False)
        if ck.get('schema')!='physx-cts-source-port-v1' or ck['cfg']['stage']!='flat':
            raise ValueError('Terrain expert must be this velocity task\'s flat checkpoint')
        self.policy=Policy(tuple(ck['cfg']['actor_hidden'])).to(device).eval()
        self.policy.load_state_dict(ck['policy'])
    @torch.no_grad()
    def __call__(self,hist):return self.policy(hist,None,1)
