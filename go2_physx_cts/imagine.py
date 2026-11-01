"""Full-pose visual imagination with differentiable physics, depth and reward.

A frozen module still passes derivatives to its inputs. Only replay burn-in and
TD target construction use no_grad; the Actor rollout never detaches visual
observations, proprioception, recurrent memory, task completion or pose.
"""
import torch
from torch.utils.checkpoint import checkpoint
from . import state as st
from .terrain import relief,geometry_crop
from .task import task_reward,ELAPSED,EPISODE_LIMIT


def time_limit_return(rewards,soft_done,timeout,values,gamma,valid_start=None):
    continuation=(1-soft_done)*(1-timeout.to(rewards.dtype))
    after=continuation.cumprod(-1)
    before=torch.cat((torch.ones_like(after[:,:1]),after[:,:-1]),-1)
    if valid_start is not None:
        before=before*valid_start[:,None]; after=after*valid_start[:,None]
    discount=gamma**torch.arange(rewards.shape[1],device=rewards.device,dtype=rewards.dtype)
    immediate=(discount*before*rewards).sum(-1)
    timeout_bootstrap=(gamma*discount*before*(1-soft_done)*timeout*values[:,1:]).sum(-1)
    horizon_bootstrap=gamma**rewards.shape[1]*after[:,-1]*values[:,-1]
    bootstrap=timeout_bootstrap+horizon_bootstrap
    initial=torch.ones_like(after[:,:1]) if valid_start is None else valid_start[:,None].to(after.dtype)
    return immediate+bootstrap,torch.cat((initial,after),-1),immediate,bootstrap


@torch.no_grad()
def time_limit_lambda_targets(rewards,soft_done,timeout,values,gamma,lam):
    nxt=values[:,-1]; targets=[]
    for t in reversed(range(rewards.shape[1])):
        tail=torch.where(timeout[:,t],values[:,t+1],(1-lam)*values[:,t+1]+lam*nxt)
        nxt=rewards[:,t]+gamma*(1-soft_done[:,t])*tail; targets.append(nxt)
    return torch.stack(targets[::-1],1)


def flat_ids(x): return x.reshape(-1).long()


class Imagination:
    def __init__(self,cfg,consts,field,commands,depth):
        self.cfg,self.c,self.field,self.commands,self.depth=cfg,consts,field,commands,depth
        # Only the deterministic differentiable core is compiled. Random command
        # draws and observation noise retain their original eager RNG ordering.
        self._core_fn=self._core
        if cfg.compile:
            compiled=torch.compile(self._core,dynamic=False,fullgraph=True)
            # Replay starts, first recursive states and later recurrent states
            # have different input requires_grad patterns. Teacher, student,
            # warmup and mixed LR probes share this code object's bounded cache.
            # Scope the capacity to this call; fullgraph still fails explicitly
            # if an unexpected unbounded source of specializations is introduced.
            limits=torch._dynamo.config
            key='recompile_limit' if hasattr(limits,'recompile_limit') else 'cache_size_limit'
            self._core_fn=limits.patch({key:32})(compiled)
    def render(self,pose,seed,frame):
        return self.depth.render(pose,flat_ids(seed['scene_ids']),flat_ids(frame),flat_ids(seed['episode']),
                                 flat_ids(seed['trajectory_ids']),mode='recompute')
    @torch.no_grad()
    def burnin(self,policy,sequence):
        """Rebuild current-encoder memory from exact25Hz camera captures."""
        b,t=sequence['obs'].shape[:2]; hidden=sequence['obs'].new_zeros((b,256),dtype=torch.float32)
        depth=hidden.new_zeros((b,1,48,64)); previous=None
        for j in range(t):
            item={k:v[:,j] for k,v in sequence.items() if torch.is_tensor(v) and v.ndim>=2 and v.shape[1]==t}
            key=torch.cat((item['depth_frame_index'],item['episode'],item['depth_scene_ids']),-1)
            refresh=torch.ones(b,dtype=torch.bool,device=hidden.device) if previous is None else (key!=previous).any(-1)
            ids=refresh.nonzero().flatten()
            if len(ids):
                depth[ids]=self.depth.render(item['depth_pose'][ids],flat_ids(item['depth_scene_ids'])[ids],
                    flat_ids(item['depth_frame_index'])[ids],flat_ids(item['episode'])[ids],sequence['trajectory_ids'][ids],mode='inference')
            previous=key
            _,_,new=policy.student_latent(item['obs'].float(),depth,hidden,item['reset'].reshape(-1).bool())
            hidden=torch.where(item['valid'].reshape(-1,1).bool(),new,hidden)
        return hidden
    def _core(self,wm,policy,s,pose,a_prev,a_prev2,hist,cmd_state,xi,hidden,context,depth,branch,tasks,terrains,iteration):
        command=self.commands.command(cmd_state)
        priv=st.privileged(s,a_prev,command,xi,relief(pose,self.field))
        height=geometry_crop(s,pose,self.field,kind='teacher') if branch!=1 else s.new_zeros((len(s),2,1,1))
        a,nhidden,_,_=policy.step(hist,priv,branch,depth=depth,hidden=hidden,height=height,task_ids=tasks)
        raw,dp,telemetry=wm.predict(s,a,a_prev,xi,geometry_crop(s,pose,self.field,kind='world_model'),pose)
        np=st.apply_pose_delta(pose,dp); ns=st.reconcile_state(raw,np,self.field,self.c.default_q)
        lookup=getattr(self.commands,'terrain_lookup',None)
        current_terrain=lookup.query(np[:,:2]) if lookup is not None else terrains
        r,terms,ncontext,done,_=task_reward(s,ns,a,a_prev,command,self.c,task_id=tasks,terrain_id=current_terrain,
            context=context,pose0=pose,pose=np,telemetry=telemetry,previous2=a_prev2,
            iteration=iteration,field=self.field,soft=True)
        return ns,np,a,nhidden,ncontext,r,done
    def _step(self,wm,policy,s,pose,a_prev,a_prev2,hist,cmd_state,xi,hidden,context,depth,branch,tasks,terrains,iteration,reward_iteration,active):
        ns,np,a,nhidden,ncontext,r,done=self._core_fn(wm,policy,s,pose,a_prev,a_prev2,hist,cmd_state,xi,
            hidden,context,depth,branch,tasks,terrains,reward_iteration)
        ncmd=self.commands.advance(cmd_state,np[:,:2],iteration,task_ids=tasks,terrain_ids=terrains)
        nhist=st.push_history(hist,st.frame(ns,a,self.commands.command(ncmd),noise=True))
        def select(new,old): return torch.where(active[:,None],new,old)
        return (select(ns,s),select(np,pose),select(a,a_prev),select(a_prev,a_prev2),select(nhist,hist),
            select(ncmd,cmd_state),select(nhidden,hidden),select(ncontext,context),
            torch.where(active,r,torch.zeros_like(r)),torch.where(active,done,torch.zeros_like(done)))
    def rollout(self,wm,policy,critic,seed,branch,iteration=0):
        cfg=self.cfg; s,pose,a_prev,hist=seed['s'],seed['pose'],seed['a_prev'],seed['obs'].float()
        a_prev2=seed['a_prev2']; cmd_state,xi=seed['cmd_state'],seed['xi']; context=seed['task_context']
        tasks,terrains=flat_ids(seed['task_ids']),flat_ids(seed['terrain_ids'])
        frame=flat_ids(seed['frame_index']); b=len(s)
        reward_iteration=torch.as_tensor(iteration,device=s.device,dtype=s.dtype)
        hidden=s.new_zeros((b,256)); depth=s.new_zeros((b,1,48,64))
        nteacher=branch[0] if isinstance(branch,tuple) else (b if branch==0 else 0)
        student=slice(nteacher,b)
        if nteacher<b:
            prefix={k:v[student] for k,v in seed['prefix'].items()}
            hidden=torch.cat((hidden[:nteacher],self.burnin(policy,prefix)),0)
            subset={k:v[student] for k,v in seed.items() if torch.is_tensor(v)}
            view=dict(subset,scene_ids=subset['depth_scene_ids'])
            depth=torch.cat((depth[:nteacher],self.render(subset['depth_pose'],view,subset['depth_frame_index'])),0)
        active=context[:,ELAPSED]<context[:,EPISODE_LIMIT]; valid_start=active.to(s.dtype)
        rewards=[]; soft_done=[]; timeouts=[]; valid=[]; features=[]; values=[s.new_zeros(b)]
        for t in range(cfg.horizon):
            command=self.commands.command(cmd_state)
            priv=st.privileged(s,a_prev,command,xi,relief(pose,self.field))
            features.append(critic.features(priv,branch,tasks,context,pose))
            valid.append(active)
            timeout=active&(context[:,ELAPSED]+cfg.dt>=context[:,EPISODE_LIMIT]-1e-6)
            if t and nteacher<b:
                refresh=((frame[student]+t)%cfg.depth_period==0)
                # Every refreshed image corresponds to this imagined pose; held
                # images keep the gradient of their most recent capture pose.
                ids=refresh.nonzero().flatten()+nteacher
                if ids.numel():
                    subset={k:v[ids] for k,v in seed.items() if torch.is_tensor(v)}
                    if self.field is not None: subset['scene_ids']=self.field.scene_ids(pose[ids])
                    new=self.render(pose[ids],subset,(frame+t)[ids])
                    depth=depth.index_copy(0,ids,new)
            args=(s,pose,a_prev,a_prev2,hist,cmd_state,xi,hidden,context,depth)
            # Bind active for recomputation; loop-variable capture would replay a
            # different mask in backward after the rollout has advanced.
            current_active=active
            def step(*args,current_active=current_active):
                return self._step(wm,policy,*args,branch,tasks,terrains,iteration,reward_iteration,current_active)
            out=checkpoint(step,*args,use_reentrant=False,preserve_rng_state=True)
            s,pose,a_prev,a_prev2,hist,cmd_state,hidden,context,r,done=out
            rewards.append(r); soft_done.append(done); timeouts.append(timeout)
            active=active&~timeout
            p=st.privileged(s,a_prev,self.commands.command(cmd_state),xi,relief(pose,self.field))
            next_feature=critic.features(p,branch,tasks,context,pose)
            if t==cfg.horizon-1: values.append(critic(next_feature))
            else:
                ids=timeout.nonzero().flatten(); v=s.new_zeros(b)
                if ids.numel(): v=v.index_copy(0,ids,critic(next_feature[ids]))
                values.append(v)
        features.append(next_feature); features=torch.stack(features,1)
        rewards=torch.stack(rewards,1); soft_done=torch.stack(soft_done,1); timeouts=torch.stack(timeouts,1)
        objective,surv,immediate,bootstrap=time_limit_return(rewards,soft_done,timeouts,torch.stack(values,1),cfg.gamma,valid_start)
        return objective,{'rewards':rewards.detach(),'surv':surv.detach(),'soft_done':soft_done.detach(),
            'timeout':timeouts,'critic_valid':torch.stack(valid,1),'critic_in':features.detach(),
            'immediate':immediate.detach(),'bootstrap':bootstrap.detach()}
    @torch.no_grad()
    def targets(self,critic,data,geometry=None):
        flat=data['critic_in'].reshape(-1,critic.feature_dim)
        chunk_size=getattr(self.cfg,'critic_microbatch',4096)
        pieces=[]
        for start in range(0,len(flat),chunk_size):
            stop=min(start+chunk_size,len(flat))
            indices=torch.arange(start,stop,device=flat.device)
            pieces.append(critic(flat[start:stop],geometry=geometry,geometry_indices=indices))
        values=torch.cat(pieces).reshape(data['critic_in'].shape[:2])
        return time_limit_lambda_targets(data['rewards'],data['soft_done'],data['timeout'],values,self.cfg.gamma,self.cfg.lam)
