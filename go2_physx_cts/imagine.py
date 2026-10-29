"""Source BPTT/soft-survival return with three-velocity command and CTS adapters."""
import torch
from . import state as st
from .terrain import relief
from .task import reward
from .commands import OfficialCommands,EPISODE_ELAPSED


def time_limit_return(rewards,soft_done,timeout,values,gamma,valid_start=None):
    """Differentiable short return: time limits bootstrap, falls end survival.

    rewards/soft_done/timeout are[B,H], values[B,H+1]. The terminal
    transition still earns its reward. Values retain state-input gradients.
    """
    continuation=(1-soft_done)*(1-timeout.to(rewards.dtype))
    after=continuation.cumprod(-1)
    before=torch.cat((torch.ones_like(after[:,:1]),after[:,:-1]),-1)
    if valid_start is not None:
        before=before*valid_start[:,None];after=after*valid_start[:,None]
    discount=gamma**torch.arange(rewards.shape[1],device=rewards.device,dtype=rewards.dtype)
    immediate=(discount*before*rewards).sum(-1)
    timeout_bootstrap=(gamma*discount*before*(1-soft_done)*timeout*values[:,1:]).sum(-1)
    horizon_bootstrap=gamma**rewards.shape[1]*after[:,-1]*values[:,-1]
    bootstrap=timeout_bootstrap+horizon_bootstrap
    initial=torch.ones_like(after[:,:1]) if valid_start is None else valid_start[:,None].to(after.dtype)
    return immediate+bootstrap,torch.cat((initial,after),-1),immediate,bootstrap


@torch.no_grad()
def time_limit_lambda_targets(rewards,soft_done,timeout,values,gamma,lam):
    """TD-lambda cannot propagate a return across an episode time limit."""
    nxt=values[:,-1];targets=[]
    for t in reversed(range(rewards.shape[1])):
        tail=torch.where(timeout[:,t],values[:,t+1],(1-lam)*values[:,t+1]+lam*nxt)
        nxt=rewards[:,t]+gamma*(1-soft_done[:,t])*tail
        targets.append(nxt)
    return torch.stack(targets[::-1],1)

class Imagination:
    def __init__(self,cfg,consts,field=None,commands=None):
        self.cfg,self.c,self.field=cfg,consts,field
        from .robots import profile
        self.reward_height=profile(cfg.robot).reward_height
        self.commands=commands or OfficialCommands(cfg.dt,cfg.episode_seconds)
        self._step_fn=torch.compile(self._step,dynamic=False) if cfg.compile else self._step
    def _step(self,wm,policy,critic,s,pose,a_prev,hist,cmd_state,xi,branch,iteration,active):
        cfg=self.cfg;command=self.commands.command(cmd_state)
        rlf=relief(pose,self.field);priv=st.privileged(s,a_prev,command,xi,rlf)
        feature=critic.features(priv,branch)
        a=policy(hist,priv,branch)
        ns,dp=wm(s,a,a_prev,xi,rlf);np=st.apply_pose_delta(pose,dp)
        r,_,done=reward(s,ns,a,a_prev,command,self.c,cfg.stage,soft=True,iteration=iteration,pose=np,field=self.field,base_height_target=self.reward_height)
        # An inactive row is still evaluated in the fixed-size compiled graph.
        # Give only that discarded command candidate a safe elapsed time; keep
        # its actual terminal memory frozen, avoiding negative remaining-time
        # denominators after an already completed episode.
        elapsed=torch.where(active,cmd_state[:,EPISODE_ELAPSED],torch.zeros_like(cmd_state[:,EPISODE_ELAPSED]))
        candidate_command=torch.cat((cmd_state[:,:EPISODE_ELAPSED],elapsed[:,None],cmd_state[:,EPISODE_ELAPSED+1:]),-1)
        candidate_command=self.commands.advance(candidate_command,np[:,:2],iteration)
        command=self.commands.command(candidate_command)
        nhist=st.push_history(hist,st.frame(ns,a,command,noise=True))
        def select(new,old):return torch.where(active[:,None],new,old)
        return (select(ns,s),select(np,pose),select(a,a_prev),select(nhist,hist),
                select(candidate_command,cmd_state),torch.where(active,r,torch.zeros_like(r)),
                torch.where(active,done,torch.zeros_like(done)),feature)
    def rollout(self,wm,policy,critic,seed,branch,iteration=0):
        cfg=self.cfg;s,pose,a_prev,hist=seed['s'],seed['pose'],seed['a_prev'],seed['obs']
        cmd_state,xi=seed['cmd_state'],seed['xi']
        iteration=torch.as_tensor(iteration,device=s.device)
        max_steps=round(cfg.episode_seconds/cfg.dt)
        active=torch.round(cmd_state[:,EPISODE_ELAPSED]/cfg.dt)<max_steps
        valid_start=active.to(s.dtype)
        rewards=[];soft_done=[];timeouts=[];valid=[];features=[]
        for _ in range(cfg.horizon):
            valid.append(active)
            timeout=active&(torch.round(cmd_state[:,EPISODE_ELAPSED]/cfg.dt)+1>=max_steps)
            s,pose,a_prev,hist,cmd_state,r,done,feature=self._step_fn(wm,policy,critic,s,pose,a_prev,hist,cmd_state,xi,branch,iteration,active)
            features.append(feature)
            rewards.append(r);soft_done.append(done);timeouts.append(timeout)
            active=active&~timeout
        last_priv=st.privileged(s,a_prev,self.commands.command(cmd_state),xi,relief(pose,self.field))
        features.append(critic.features(last_priv,branch));features=torch.stack(features,1)
        rewards=torch.stack(rewards,1);soft_done=torch.stack(soft_done,1);timeouts=torch.stack(timeouts,1)
        objective,surv,immediate,bootstrap=time_limit_return(rewards,soft_done,timeouts,critic(features),cfg.gamma,valid_start)
        return objective,{'rewards':rewards.detach(),'surv':surv.detach(),'soft_done':soft_done.detach(),
                         'timeout':timeouts,'critic_valid':torch.stack(valid,1),'critic_in':features.detach(),
                         'immediate':immediate.detach(),'bootstrap':bootstrap.detach()}

    @torch.no_grad()
    def targets(self,critic,data):
        cfg=self.cfg
        return time_limit_lambda_targets(data['rewards'],data['soft_done'],data['timeout'],
                                         critic(data['critic_in']),cfg.gamma,cfg.lam)
