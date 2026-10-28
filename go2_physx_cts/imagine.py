"""Source BPTT/soft-survival return with three-velocity command and CTS adapters."""
import torch
from . import state as st
from .terrain import relief
from .task import reward,advance_commands

class Imagination:
    def __init__(self,cfg,consts,field=None):
        self.cfg,self.c,self.field=cfg,consts,field
        self._step_fn=torch.compile(self._step,dynamic=False) if cfg.compile else self._step
    def _step(self,wm,policy,critic,s,pose,a_prev,hist,command,timer,xi,branch):
        cfg=self.cfg;rlf=relief(pose,self.field);priv=st.privileged(s,a_prev,command,xi,rlf)
        feature=critic.features(priv,branch)
        a=policy(hist,priv,branch)
        ns,dp=wm(s,a,a_prev,xi,rlf);np=st.apply_pose_delta(pose,dp)
        r,_,done=reward(s,ns,a,a_prev,command,self.c,cfg.stage,soft=True)
        command,timer=advance_commands(command,timer,cfg.command_max)
        hist=st.push_history(hist,st.frame(ns,a,command,noise=True))
        return ns,np,a,hist,command,timer,r,done,feature
    def rollout(self,wm,policy,critic,seed,branch):
        cfg=self.cfg;s,pose,a_prev,hist=seed['s'],seed['pose'],seed['a_prev'],seed['obs']
        command,timer,xi=seed['command'],seed['timer'],seed['xi']
        surv=torch.ones(len(s),device=s.device);ret=torch.zeros_like(surv);disc=1.
        rewards=[];survivals=[];features=[]
        for _ in range(cfg.horizon):
            survivals.append(surv)
            s,pose,a_prev,hist,command,timer,r,done,feature=self._step_fn(wm,policy,critic,s,pose,a_prev,hist,command,timer,xi,branch)
            features.append(feature)
            ret=ret+disc*surv*r;rewards.append(r);surv=surv*(1-done);disc*=cfg.gamma
        last_priv=st.privileged(s,a_prev,command,xi,relief(pose,self.field))
        features.append(critic.features(last_priv,branch));tail=disc*surv*critic(features[-1])
        return ret+tail,{'rewards':torch.stack(rewards,1).detach(),'surv':torch.stack(survivals+[surv],1).detach(),
                         'critic_in':torch.stack(features,1).detach(),'immediate':ret.detach(),'bootstrap':tail.detach()}

    @torch.no_grad()
    def targets(self,critic,data):
        cfg=self.cfg;r=data['rewards'];surv=data['surv'];v=critic(data['critic_in'])
        nxt=v[:,-1];targets=[]
        for t in reversed(range(cfg.horizon)):
            cont=surv[:,t+1]/surv[:,t].clamp(min=1e-6)
            nxt=r[:,t]+cfg.gamma*cont*((1-cfg.lam)*v[:,t+1]+cfg.lam*nxt);targets.append(nxt)
        return torch.stack(targets[::-1],1)
