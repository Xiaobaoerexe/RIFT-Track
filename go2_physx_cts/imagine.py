"""Full-pose visual imagination with differentiable physics, depth and reward.

A frozen module still passes derivatives to its inputs. Only replay burn-in and
TD target construction use no_grad; the Actor rollout never detaches visual
observations, proprioception, recurrent memory, task completion or pose.
"""
import torch
from torch.utils.checkpoint import checkpoint
from .activation import matrix_checkpoint
from . import state as st
from .terrain import relief,geometry_crop
from .task import task_reward,episode_time_reached,ELAPSED,EPISODE_LIMIT
from .sequence_execution import depth_sequence_features


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
        self._core_fn=matrix_checkpoint(self._core)
        if cfg.compile:
            compiled=torch.compile(self._core_fn,dynamic=False,fullgraph=True)
            # Replay starts, first recursive states and later recurrent states
            # have different input requires_grad patterns. Teacher, student,
            # warmup and mixed LR probes share this code object's bounded cache.
            # Scope the capacity to this call; fullgraph still fails explicitly
            # if an unexpected unbounded source of specializations is introduced.
            limits=torch._dynamo.config
            key='recompile_limit' if hasattr(limits,'recompile_limit') else 'cache_size_limit'
            self._core_fn=limits.patch({key:32})(compiled)
    @torch.no_grad()
    def burnin(self,policy,sequence):
        """Rebuild current-encoder memory from exact configured5Hz captures."""
        b,t=sequence['obs'].shape[:2]; hidden=sequence['obs'].new_zeros((b,256),dtype=torch.float32)
        features=depth_sequence_features(policy.student_encoder,self.depth,sequence,0,t)
        obs=sequence['obs'].float().reshape(b*t,-1)
        singles=policy.single_norm(st.single_from_history(obs)).reshape(b,t,-1)
        for j in range(t):
            new=policy.student_encoder.advance_memory(singles[:,j],features[:,j],hidden,sequence['reset'][:,j].reshape(-1).bool())
            hidden=torch.where(sequence['valid'][:,j].reshape(-1,1).bool(),new,hidden)
        return hidden
    def encode_depth(self,policy,images):
        # A held feature is shared by all held control steps without detaching its
        # capture-pose derivative. Recompute its CNN only once during backward.
        return checkpoint(policy.student_encoder.encode_depth,images,use_reentrant=False) if torch.is_grad_enabled() else policy.student_encoder.encode_depth(images)
    def _core(self,wm,policy,s,pose,a_prev,a_prev2,hist,cmd_state,xi,hidden,context,depth_features,frame,capture_pose,branch,tasks,terrains,iteration):
        command=self.commands.command(cmd_state)
        nt=branch[0] if isinstance(branch,tuple) else (len(s) if branch==0 else 0)
        if nt:
            priv=st.privileged(s[:nt],a_prev[:nt],command[:nt],xi[:nt],relief(pose[:nt],self.field))
            height=geometry_crop(s[:nt],pose[:nt],self.field,kind='teacher')
        else:priv=s.new_empty((0,st.PRIV_DIM));height=s.new_empty((0,2,1,1))
        a,nhidden,_,_=policy.step(hist,priv,branch,depth=None,depth_features=depth_features,hidden=hidden,height=height,task_ids=tasks)
        raw,dp,telemetry,next_latent=wm.predict(s,a,a_prev,xi,geometry_crop(s,pose,self.field,kind='world_model'),pose,
            latent=depth_features,frame_index=frame,capture_pose=capture_pose)
        np=st.apply_pose_delta(pose,dp); ns=st.reconcile_state(raw,np,self.field,self.c.default_q)
        lookup=getattr(self.commands,'terrain_lookup',None)
        current_terrain=lookup.query(np[:,:2]) if lookup is not None else terrains
        r,terms,ncontext,done,_=task_reward(s,ns,a,a_prev,command,self.c,task_id=tasks,terrain_id=current_terrain,
            context=context,pose0=pose,pose=np,telemetry=telemetry,previous2=a_prev2,
            iteration=iteration,field=self.field,soft=True,variant=self.cfg.variant,base_height_target=.30)
        return ns,np,a,nhidden,ncontext,terms['continuous_reward'],done,next_latent,telemetry['contact_prob']
    def _step(self,wm,policy,s,pose,a_prev,a_prev2,hist,cmd_state,xi,hidden,context,depth_features,frame,capture_pose,branch,tasks,terrains,iteration,reward_iteration,active):
        ns,np,a,nhidden,ncontext,r,done,next_latent,contact=self._core_fn(wm,policy,s,pose,a_prev,a_prev2,hist,cmd_state,xi,
            hidden,context,depth_features,frame,capture_pose,branch,tasks,terrains,reward_iteration)
        ncmd=self.commands.advance(cmd_state,np[:,:2],iteration,task_ids=tasks,completed=ncontext[:,3])
        nhist=st.push_history(hist,st.frame(ns,a,self.commands.command(ncmd),noise=True))
        def select(new,old): return torch.where(active[:,None],new,old)
        return (select(ns,s),select(np,pose),select(a,a_prev),select(a_prev,a_prev2),select(nhist,hist),
            select(ncmd,cmd_state),select(nhidden,hidden),select(ncontext,context),
            torch.where(active,r,torch.zeros_like(r)),torch.where(active,done,torch.zeros_like(done)),
            select(next_latent,depth_features),contact)
    def rollout(self,wm,policy,critic,seed,branch,iteration=0):
        cfg=self.cfg; s,pose,a_prev,hist=seed['s'],seed['pose'],seed['a_prev'],seed['obs'].float()
        a_prev2=seed['a_prev2']; cmd_state,xi=seed['cmd_state'],seed['xi']; context=seed['task_context']
        tasks,terrains=flat_ids(seed['task_ids']),flat_ids(seed['terrain_ids'])
        frame=flat_ids(seed['frame_index']); b=len(s)
        reward_iteration=torch.as_tensor(iteration,device=s.device,dtype=s.dtype)
        hidden=s.new_zeros((b,256))
        capture_pose=seed['depth_pose']
        images=self.depth.captured(capture_pose,flat_ids(seed['depth_scene_ids']),flat_ids(seed['depth_frame_index']),
            flat_ids(seed['episode']),flat_ids(seed['trajectory_ids']))
        with torch.no_grad():depth_features=policy.student_encoder.encode_depth(images)
        contacts=seed['contact_prob'];root_event=s.new_zeros(b)
        nteacher=branch[0] if isinstance(branch,tuple) else (b if branch==0 else 0)
        student=slice(nteacher,b)
        if nteacher<b:
            prefix=seed['student_prefix'] if 'student_prefix' in seed else {k:v[student] for k,v in seed['prefix'].items()}
            hidden=torch.cat((hidden[:nteacher],self.burnin(policy,prefix)),0)
        active=~episode_time_reached(context[:,ELAPSED],context[:,EPISODE_LIMIT]); valid_start=active.to(s.dtype)
        rewards=[]; soft_done=[]; timeouts=[]; valid=[]; features=[]; values=[s.new_zeros(b)]
        for t in range(cfg.horizon):
            command=self.commands.command(cmd_state)
            priv=st.privileged(s,a_prev,command,xi,relief(pose,self.field))
            features.append(critic.features(priv,branch,tasks,context,pose,contacts))
            valid.append(active)
            args=(s,pose,a_prev,a_prev2,hist,cmd_state,xi,hidden,context,depth_features,frame+t,capture_pose)
            pre_context=context
            # The core owns selective activation storage inside torch.compile.
            # Command/noise/history operations retain their ordinary small graph;
            # a second outer checkpoint would discard the selected expensive work.
            current_active=active
            def step(*args,current_active=current_active):
                return self._step(wm,policy,*args,branch,tasks,terrains,iteration,reward_iteration,current_active)
            out=step(*args)
            s,pose,a_prev,a_prev2,hist,cmd_state,hidden,context,r,done,depth_features,contacts=out
            refresh=((frame+t+1)%cfg.depth_period==0)&current_active
            capture_pose=torch.where(refresh[:,None],pose,capture_pose)
            if t==0 and cfg.variant!='simple':
                after_priv=st.privileged(s,a_prev,command,xi,relief(pose,self.field))
                root_event=critic.forward_event(critic.event_features(after_priv,branch,tasks,pre_context,pose,contacts))
            timeout=current_active&episode_time_reached(context[:,ELAPSED],context[:,EPISODE_LIMIT])
            rewards.append(r); soft_done.append(done); timeouts.append(timeout)
            active=active&~timeout
            p=st.privileged(s,a_prev,self.commands.command(cmd_state),xi,relief(pose,self.field))
            next_feature=critic.features(p,branch,tasks,context,pose,contacts)
            if t==cfg.horizon-1: values.append(critic(next_feature))
            else:
                ids=timeout.nonzero().flatten(); v=s.new_zeros(b)
                if ids.numel(): v=v.index_copy(0,ids,critic(next_feature[ids]))
                values.append(v)
        features.append(next_feature); features=torch.stack(features,1)
        rewards=torch.stack(rewards,1); soft_done=torch.stack(soft_done,1); timeouts=torch.stack(timeouts,1)
        # Finite event-task deadlines are terminal; ordinary AS2 episode truncations bootstrap.
        finite=(tasks>=3) if cfg.variant!='simple' else torch.zeros_like(tasks,dtype=torch.bool)
        value_tensor=torch.stack(values,1)
        value_tensor=torch.where(torch.cat((torch.zeros_like(timeouts[:,:1]),timeouts&finite[:,None]),1),0.,value_tensor)
        objective,surv,immediate,bootstrap=time_limit_return(rewards,soft_done,timeouts,value_tensor,cfg.gamma,valid_start)
        return objective+root_event*valid_start,{'rewards':rewards.detach(),'surv':surv.detach(),'soft_done':soft_done.detach(),
            'timeout':timeouts,'critic_valid':torch.stack(valid,1),'critic_in':features.detach(),
            'finite':finite,'immediate':immediate.detach(),'bootstrap':bootstrap.detach(),'root_event':root_event.detach()}
    @torch.no_grad()
    def targets(self,critic,data):
        flat=data['critic_in'].reshape(-1,critic.feature_dim)
        chunk_size=getattr(self.cfg,'critic_microbatch',4096)
        values=torch.cat([critic(part) for part in flat.split(chunk_size)]).reshape(data['critic_in'].shape[:2])
        zero_tail=torch.cat((torch.zeros_like(data['timeout'][:,:1]),data['timeout']&data['finite'][:,None]),1)
        values=values.masked_fill(zero_tail,0.)
        return time_limit_lambda_targets(data['rewards'],data['soft_done'],data['timeout'],values,self.cfg.gamma,self.cfg.lam)
