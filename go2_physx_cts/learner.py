"""AS2 source-paced learning with 50Hz visual latents and total-return critics."""
import copy,json,math,os,time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from . import UPSTREAM_COMMIT,state as st
from .networks import Policy,StudentExport
from .value import Critic
from .world_model import WorldModel,supervised_loss
from .latent import prepare_latent_targets
from .buffer import Replay,Archive,mixed_windows
from .imagine import Imagination,flat_ids
from .terrain import relief,geometry_crop
from .depth import DepthObservation
from .schedules import LearningRateController
from .sequence_execution import depth_sequence_features,batched_apply,ENCODING_CHUNK


@contextmanager
def frozen(*modules):
    params=list({id(p):p for m in modules for p in m.parameters()}.values())
    flags=[p.requires_grad for p in params]
    for p in params:p.requires_grad_(False);p.grad=None
    try:yield
    finally:
        for p,flag in zip(params,flags):p.requires_grad_(flag)


def clip_grad(params,global_norm=1.,elem=float('inf')):
    params=[p for p in params if p.grad is not None]
    norm=torch.nn.utils.clip_grad_norm_(params,global_norm)
    for p in params:p.grad.clamp_(-elem,elem)
    return norm


def scalar_finite(x):return bool(torch.isfinite(x).all())


def branch_loss(objectives,branch,sizes):
    if isinstance(branch,tuple):
        return -objectives[:branch[0]].sum()/sizes[0]-objectives[branch[0]:].sum()/sizes[1]
    return -objectives.sum()/sizes[branch]


def critic_loss(pred,target,tags,delta=1.,denominators=None):
    per=2*F.huber_loss(pred,target,reduction='none',delta=delta)
    denominators=tags.sum(0).clamp_min(1) if denominators is None else denominators
    return .5*sum((per*tags[:,b]).sum()/denominators[b] for b in (0,1))


def subset(data,index):
    return {k:subset(v,index) if isinstance(v,dict) else v[index] for k,v in data.items()}


class Learner:
    schema='as2-perceptive-wm50-policy10-v3'
    def __init__(self,cfg,env,output,driver=None):
        self.cfg,self.env,self.output=cfg,env,Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.device=cfg.device;torch.manual_seed(cfg.seed)
        self.policy=Policy(cfg.actor_hidden,experts=cfg.actor_experts,mixed_precision=cfg.mixed_precision).to(cfg.device)
        self.critic=Critic(cfg.critic_hidden,variant=cfg.variant,mixed_precision=False).to(cfg.device)
        self.wm=WorldModel(cfg.wm_hidden,mixed_precision=cfg.mixed_precision,depth_period=cfg.depth_period).to(cfg.device)
        self.depth=DepthObservation(cfg,env.field)
        self.actor_params=list(self.policy.teacher_encoder.parameters())+list(self.policy.actor.parameters())
        self.cts_params=list(self.policy.student_encoder.distillation_parameters())
        self.wm_params=list(self.wm.parameters())+list(self.policy.student_encoder.depth_encoder.parameters())
        self.opt_wm=torch.optim.Adam(self.wm_params,lr=cfg.wm_lr)
        self.opt_actor=torch.optim.Adam(self.actor_params,lr=cfg.lr,betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=cfg.lr)
        self.opt_cts=torch.optim.Adam(self.cts_params,lr=cfg.cts_lr)
        owners=[{id(p) for g in opt.param_groups for p in g['params']} for opt in
                (self.opt_wm,self.opt_actor,self.opt_critic,self.opt_cts)]
        if any(owners[i]&owners[j] for i in range(4) for j in range(i)):
            raise RuntimeError('A parameter has multiple optimizer owners')
        self._wm_loss_fn=torch.compile(supervised_loss,dynamic=False,fullgraph=True) if cfg.compile else supervised_loss
        storage='cpu' if cfg.replay_storage=='cpu' else None
        self.buf=Replay(cfg.num_envs,cfg.buffer_len,cfg.device,cfg.lr_holdout_modulus,storage_device=storage)
        self.archive=Archive(cfg.archive_size,cfg.wm_window,cfg.device,storage_device=storage)
        self.imag=Imagination(cfg,env.consts,env.field,env.command_sampler,self.depth)
        self.schedule=LearningRateController(cfg)
        self.index=torch.arange(cfg.num_envs,device=cfg.device)
        self.source=torch.zeros(cfg.num_envs,dtype=torch.uint8,device=cfg.device)
        self.ou=torch.zeros((cfg.num_envs,12),device=cfg.device)
        self.real_hidden=torch.zeros((cfg.num_envs,256),device=cfg.device)
        self.real_depth=torch.zeros((cfg.num_envs,1,48,64),device=cfg.device)
        self.depth_pose=torch.zeros((cfg.num_envs,7),device=cfg.device)
        self.depth_scene_ids=torch.zeros(cfg.num_envs,dtype=torch.long,device=cfg.device)
        self.depth_frame_index=torch.zeros_like(self.depth_scene_ids)
        self.camera_episode=torch.full_like(self.depth_scene_ids,-1)
        self.it=0;self.nonfinite=0;self.last_checkpoint=None
        self.counts=dict(collect=0,transitions=0,wm=0,actor=0,critic=0,cts=0)
        self.metrics=[];self.window_stats=[];self.start_time=time.monotonic()
        self._quality={};self._validation_phase=None;self._wm_validation=None;self._anchor=None
        self._anchor_reference=None;self._cts_stats={};self.cts_stable=False;self.fixed_reports=[]
        self.allocator=None
        self.action_quality=torch.zeros(13,device=cfg.device)
        self.refresh_providers()
        (self.output/'resolved_config.json').write_text(json.dumps(asdict(cfg),indent=2))
        (self.output/'run_identity.json').write_text(json.dumps({'schema':self.schema,'upstream_commit':UPSTREAM_COMMIT,
            'baseline_commit':'2061be6b0d592afe9e81223206cf5f864e45b9f4','physics':'isaacsim_physx','robot':'as2',
            'variant':cfg.variant,'imaginary_branch_samples':[int(cfg.imag_batch*.75),int(cfg.imag_batch*.25)],
            'branch_loss':'teacher_loss.mean() + student_loss.mean()','replay_tensor_bytes':self.buf.storage_bytes,
            'replay_storage_device':str(self.buf.storage_device),'replay_compute_device':str(self.buf.device),
            'depth_cnn_owner':'world_model_reconstruction_only','student_sequence_effective_frames':20480,
            'init':'cold','external_providers':0,'future_depth':'50Hz WM latent, 10Hz held policy visual input; no imagined rendering',
            'imagination_depth_period':cfg.imagination_depth_period,'latent_actor_stop_gradient':cfg.latent_actor_stop_gradient},indent=2))

    def refresh_providers(self,before=None):
        before=self.env.snapshot() if before is None else before
        tasks=flat_ids(before['task_ids']);old=self.source.clone();self.source.zero_()
        for task in range(4):
            ids=(tasks==task).nonzero().flatten();ns=len(ids)-int(len(ids)*self.cfg.teacher_fraction)
            if ns:self.source[ids[torch.linspace(0,len(ids)-1,ns,device=self.device).round().long()]]=1
        changed=self.source!=old;self.real_hidden[changed]=0;self.camera_episode[changed]=-1
        self._provider_task_version=self.env.task_assignment_version
        self.provider_counts=torch.bincount(self.source.long(),minlength=2).tolist()
        self.env.own_mask=torch.ones_like(self.source,dtype=torch.bool);self.env.source=self.source

    def learning_rates(self):
        rates=self.schedule.rates(self.it)
        for name in ('wm','actor','critic','cts'):
            for group in getattr(self,'opt_'+name).param_groups:group['lr']=rates[name]
        return rates

    @torch.no_grad()
    def camera_inputs(self,before):
        frame=flat_ids(before['frame_index']);episode=flat_ids(before['episode'])
        changed=episode!=self.camera_episode;self.real_hidden[changed]=0
        refresh=(frame%self.cfg.depth_period==0)|changed
        self.depth_pose[refresh]=before['pose'][refresh]
        self.depth_scene_ids[refresh]=flat_ids(before['scene_ids'])[refresh]
        self.depth_frame_index[refresh]=frame[refresh];self.camera_episode.copy_(episode)
        ids=(refresh&(self.source==1)).nonzero().flatten()
        if len(ids):self.real_depth[ids]=self.depth.captured(self.depth_pose[ids],self.depth_scene_ids[ids],self.depth_frame_index[ids],episode[ids],self.index[ids])
        return self.real_depth

    @torch.no_grad()
    def collect(self):
        cfg=self.cfg;before=self.env.snapshot()
        if self.env.task_assignment_version!=self._provider_task_version:self.refresh_providers(before)
        priv=st.privileged(before['s'],before['a_prev'],before['command'],before['xi'],before['relief'])
        depth=self.camera_inputs(before);a=torch.empty((cfg.num_envs,12),device=self.device)
        for branch in (0,1):
            ids=(self.source==branch).nonzero().flatten()
            if not len(ids):continue
            height=geometry_crop(before['s'][ids],before['pose'][ids],self.env.field,kind='teacher') if branch==0 else before['s'].new_zeros((len(ids),2,1,1))
            action,hidden,_,_=self.policy.step(before['obs'][ids],priv[ids],branch,depth=depth[ids],hidden=self.real_hidden[ids],height=height)
            a[ids]=action;self.real_hidden[ids]=hidden
        sigma=cfg.explore_noise*math.sqrt(2*cfg.explore_theta)
        self.ou=self.ou*(1-cfg.explore_theta*cfg.dt)+sigma*math.sqrt(cfg.dt)*torch.randn_like(a)
        a=a+torch.where((self.index<int(cfg.num_envs*cfg.explore_frac))[:,None],self.ou,cfg.action_noise*torch.randn_like(a))
        if not scalar_finite(a):raise FloatingPointError('Nonfinite real action')
        after,done,metrics=self.env.step(a,defer_metrics=True)
        values={k:before[k] for k in self.buf.data if k in before}
        values.update(s_next=after['s'],pose_next=after['pose'],a=a,source=self.source,done=done,
            terminated=after['terminated'],timeout=after['timeout'],success=after['success'],
            command_next=after['command'],cmd_state_next=after['cmd_state'],task_context_next=after['task_context'],
            continuous_reward=after['continuous_reward'],event_reward=after['event_reward'],
            depth_pose=self.depth_pose,depth_scene_ids=self.depth_scene_ids,depth_frame_index=self.depth_frame_index)
        names=('contact_prob','applied_torque','foot_velocity','joint_acceleration','log_contact_force')
        for name in names:values['next_'+name]=after[name]
        self.buf.add(**values)
        training=~self.buf.validation_episodes(before['episode']);telemetry={k:after[k][training] for k in names}
        self.wm.update_stats(before['s'][training],a[training],before['a_prev'][training],before['xi'][training],None,after['s'][training],
            st.pose_delta(before['pose'][training],after['pose'][training]),telemetry,pose=before['pose'][training])
        self.policy.update_stats(before['obs'][training],priv[training])
        self.ou[done]=0;self.real_hidden[done]=0;self.camera_episode[done]=-1
        self.counts['collect']+=1;self.counts['transitions']+=cfg.num_envs
        self.metrics.append(metrics)

    def fit_world_model(self):
        losses=[]
        for _ in range(self.cfg.wm_updates_for(self.it)):
            w=mixed_windows(self.buf,self.archive,self.cfg,self.env.xi)
            latent=prepare_latent_targets(w,self.depth,self.policy.student_encoder.depth_encoder,self.cfg.depth_period)
            loss=self._wm_loss_fn(self.wm,w,self.env.field,self.env.consts.default_q,
                encoder=self.policy.student_encoder.depth_encoder,latent_targets=latent)
            self.opt_wm.zero_grad(set_to_none=True);loss.backward();gn=clip_grad(self.wm_params,self.cfg.grad_clip)
            if not scalar_finite(gn) or not scalar_finite(loss):raise FloatingPointError('Nonfinite WM update')
            self.opt_wm.step();self.counts['wm']+=1;losses.append(loss.detach())
        self.opt_wm.zero_grad(set_to_none=True)
        return float(torch.stack(losses).mean())

    def _actor_ready(self,seed):
        tasks=flat_ids(seed['task_ids'])
        starts=torch.where(tasks==0,self.cfg.policy_start,self.cfg.all_terrain_iteration+self.cfg.critic_warmup_iterations)
        return (self.it>=starts).to(seed['s'].dtype)

    def improve_policy(self,*,detailed_timing=True):
        cfg=self.cfg;nt=int(cfg.imag_batch*cfg.teacher_fraction);sizes=(nt,cfg.imag_batch-nt)
        active=self.it>=cfg.policy_start;all_data=[];means=[0.,0.];start=time.monotonic()
        self.opt_actor.zero_grad(set_to_none=True)
        with frozen(self.wm,self.critic,self.policy.student_encoder):
            groups=[(sizes,cfg.imag_batch)] if cfg.actor_microbatch>=cfg.imag_batch else list(enumerate(sizes))
            for branch,batch in groups:
                if isinstance(branch,tuple):
                    teacher=self.buf.seeds(nt,burnin=0);student=self.buf.seeds(sizes[1],burnin=cfg.cts_burnin)
                    seed={k:torch.cat((v,student[k]),0) for k,v in teacher.items()};seed['student_prefix']=student['prefix']
                else:seed=self.buf.seeds(batch,burnin=cfg.cts_burnin if branch else 0)
                for lo in range(0,batch,cfg.actor_microbatch):
                    sub=seed if isinstance(branch,tuple) else subset(seed,slice(lo,lo+cfg.actor_microbatch))
                    with torch.enable_grad() if active else torch.no_grad():
                        obj,data=self.imag.rollout(self.wm,self.policy,self.critic,sub,branch,iteration=self.it)
                        if active:branch_loss(obj*self._actor_ready(sub),branch,sizes).backward()
                    all_data.append(data)
                    if isinstance(branch,tuple):means=[float(obj[:nt].detach().mean()),float(obj[nt:].detach().mean())]
                    else:means[branch]+=float(obj.detach().sum())/batch
                    del obj,sub
                del seed
            gn=clip_grad(self.actor_params,cfg.grad_clip,cfg.grad_clip_elem) if active else torch.tensor(0.)
            if not scalar_finite(gn) or not all(math.isfinite(v) for v in means):raise FloatingPointError('Nonfinite Actor update')
        actor_seconds=time.monotonic()-start
        data={k:torch.cat([d[k] for d in all_data]) for k in all_data[0]}
        target=self.imag.targets(self.critic,data)
        if active:self.opt_actor.step();self.counts['actor']+=1
        self.opt_actor.zero_grad(set_to_none=True)
        x=data['critic_in'][:,:-1].reshape(-1,self.critic.feature_dim);y=target.reshape(-1)
        valid=data['critic_valid'].reshape(-1).bool();x,y=x[valid],y[valid]
        self.critic.update_stats(x)
        losses=[];fit_start=time.monotonic()
        for _ in range(cfg.critic_epochs):
            order=torch.randperm(len(x),device=self.device)
            for ids in torch.tensor_split(order,cfg.critic_minibatches):
                self.opt_critic.zero_grad(set_to_none=True);total=x.new_zeros(())
                denominators=x[ids,-2:].sum(0).clamp_min(1)
                for sub in ids.split(cfg.critic_microbatch):
                    loss=critic_loss(self.critic(x[sub]),y[sub],x[sub,-2:],cfg.critic_huber_delta,denominators)
                    loss.backward();total+=loss.detach()
                cgn=clip_grad(self.critic.parameters(),cfg.grad_clip)
                if not scalar_finite(cgn) or not scalar_finite(total):raise FloatingPointError('Nonfinite Critic update')
                self.opt_critic.step();self.counts['critic']+=1;losses.append(float(total))
        self.opt_critic.zero_grad(set_to_none=True)
        if detailed_timing:
            with torch.no_grad():
                residual=torch.cat([self.critic(part) for part in x.split(cfg.critic_microbatch)])-y
                mse=[residual[x[:,-2+b]>.5].square().mean() for b in (0,1)]
                self._quality={'critic_error':float((.5*sum(mse)).sqrt()/y.std().clamp_min(1.)),
                    'critic_explained_variance':float(1-residual.var()/y.var().clamp_min(1e-8)),
                    'critic_target_max':float(y.abs().max()),'value_guard':self.imag.last_value_guard}
        return {**self._quality,'branch_objectives':means,'objective':sum(means),'actor_grad_norm':float(gn),
            'critic_loss':sum(losses)/len(losses),
            'critic_valid_samples':len(x),'critic_samples_processed':len(x)*cfg.critic_epochs,
            'actor_seconds':actor_seconds,'critic_seconds':time.monotonic()-fit_start,
            'immediate':float(data['immediate'].mean()),'bootstrap':float(data['bootstrap'].mean())}

    def sequence_loss(self,student,sequence,training=True):
        cfg=self.cfg;b,total=sequence['obs'].shape[:2];burn=cfg.cts_burnin;steps=total-burn
        hidden=sequence['obs'].new_zeros((b,256),dtype=torch.float32)
        with torch.no_grad():
            single=self.policy.single_norm(st.single_from_history(sequence['obs'].float().reshape(b*total,-1))).reshape(b,total,-1)
            features=depth_sequence_features(student,self.depth,sequence,0,total,training=False)
            for j in range(burn):hidden=student.advance_memory(single[:,j],features[:,j],hidden,sequence['reset'][:,j].reshape(-1).bool())
            history=self.policy.history_norm(sequence['obs'][:,burn:].float().reshape(b*steps,-1))
            fields={k:sequence[k][:,burn:].reshape(b*steps,*sequence[k].shape[2:]) for k in ('s','a_prev','command','xi','relief','pose')}
            labels=[]
            for lo in range(0,b*steps,ENCODING_CHUNK):
                item={k:v[lo:lo+ENCODING_CHUNK] for k,v in fields.items()}
                priv=st.privileged(item['s'],item['a_prev'],item['command'],item['xi'],item['relief'])
                height=geometry_crop(item['s'],item['pose'],self.env.field,kind='teacher')
                labels.append(self.policy.teacher_latent(priv,height))
            label=torch.cat(labels).reshape(b,steps,-1)
        proprio,gates=student.encode_proprio(history);proprio=proprio.reshape(b,steps,-1)
        latent_losses=[];action_losses=[];action_mse=[]
        for j in range(steps):
            reset=sequence['reset'][:,burn+j].reshape(-1).bool()
            args=(proprio[:,j],single[:,burn+j],features[:,burn+j],hidden,reset)
            z,hidden=checkpoint(student.fuse_features,*args,use_reentrant=False) if training else student.fuse_features(*args)
            latent_losses.append((z.float()-label[:,j]).square().mean(-1))
            with torch.no_grad():target=self.policy.actor(label[:,j],single[:,burn+j])
            prediction=self.policy.actor(z,single[:,burn+j]);delta=(prediction-target)*self.env.consts.action_scale/.25
            per=2*F.huber_loss(delta,torch.zeros_like(delta),reduction='none',delta=1.).mean(-1)
            kind=flat_ids(sequence['terrain_ids'][:,burn+j]).clamp(0,len(self.action_quality)-1)
            action_losses.append(per*self.action_quality[kind]);action_mse.append(delta.detach().square().mean(-1))
        mask=sequence['valid'][:,burn:].reshape(b,steps).float();den=mask.sum().clamp_min(1)
        latent=(torch.stack(latent_losses,1)*mask).sum()/den
        action=(torch.stack(action_losses,1)*mask).sum()/den
        balance=(gates.float().mean(0)-1/8).square().mean();ramp=min(1.,max(0.,(self.it-2500)/2500))
        loss=latent+cfg.cts_balance*balance+cfg.cts_action_weight*ramp*action
        self._cts_stats={'cts_latent_loss':float(latent.detach()),'cts_balance_loss':float(balance.detach()),
            'cts_action_loss':float(action.detach()),'cts_action_mse':float(torch.stack(action_mse).mean()),'cts_action_coefficient':cfg.cts_action_weight*ramp}
        return loss

    def update_cts(self):
        losses=[]
        with frozen(self.wm,self.critic,self.policy.actor,self.policy.teacher_encoder,self.policy.student_encoder.depth_encoder):
            for _ in range(self.cfg.cts_updates):
                seq=self.buf.sequences(self.cfg.cts_batch,self.cfg.cts_sequence_steps,self.cfg.cts_burnin,source=1)
                loss=self.sequence_loss(self.policy.student_encoder,seq)
                self.opt_cts.zero_grad(set_to_none=True);loss.backward();gn=clip_grad(self.cts_params,self.cfg.grad_clip)
                if not scalar_finite(gn) or not scalar_finite(loss):raise FloatingPointError('Nonfinite CTS update')
                self.opt_cts.step();self.counts['cts']+=1;losses.append(loss.detach())
        self.opt_cts.zero_grad(set_to_none=True)
        return float(torch.stack(losses).mean())

    @torch.no_grad()
    def quality_diagnostics(self):
        if self.buf.size<max(self.cfg.wm_window,self.cfg.cts_burnin+1):return {}
        phase=(self.it//5000,int(self.it>=2500),int(self.it>=5000))
        if self._validation_phase!=phase:
            self._validation_phase=phase
            self._wm_validation=self.buf.windows(256,self.cfg.wm_window,validation=True)
            self._validation_scales=(self.wm.out_norm.var.sqrt().clamp_min(1e-4).clone(),self.wm.telemetry_norm.var.sqrt().clamp_min(1e-3).clone())
            self._anchor=self.buf.seeds(128,source=1,burnin=self.cfg.cts_burnin,validation=True)
            self._anchor_reference=None;self._validation_identity=str(time.time_ns())
        w=self._wm_validation;targets=prepare_latent_targets(w,self.depth,self.policy.student_encoder.depth_encoder,self.cfg.depth_period)
        errors=self.wm.validation_errors(w,self.env.field,self.env.consts.default_q,self._validation_scales,latent_targets=targets)
        a=self._anchor;priv=st.privileged(a['s'],a['a_prev'],a['command'],a['xi'],a['relief'])
        height=geometry_crop(a['s'],a['pose'],self.env.field,kind='teacher')
        hidden=self.imag.burnin(self.policy,a['prefix'])
        image=self.depth.captured(a['depth_pose'],flat_ids(a['depth_scene_ids']),flat_ids(a['depth_frame_index']),flat_ids(a['episode']),flat_ids(a['trajectory_ids']))
        ta=self.policy.step(a['obs'],priv,0,depth=None,hidden=hidden,height=height)[0]*self.env.consts.action_scale
        sa=self.policy.step(a['obs'],priv,1,depth=image,hidden=hidden,height=height)[0]*self.env.consts.action_scale
        if self._anchor_reference is None:self._anchor_reference=ta.clone()
        drift=float((ta-self._anchor_reference).square().mean().sqrt())
        if self.it%1000==0:self._anchor_reference=ta.clone()
        return {'wm_error':float(errors['total'].mean()),'latent_validation_error':float(errors.get('latent',ta.new_zeros(1)).mean()),
            'teacher_drift':drift,'cts_error':float((sa-ta).square().mean().sqrt()),'context':self._validation_identity,'cts_stable':self.cts_stable}

    def log(self,stats):
        from .physx_env import materialize_metrics
        from .task import reward_weights
        self.metrics=materialize_metrics(self.metrics);rows=self.metrics;n=max(1,len(rows))
        metrics={k:sum(m.get(k,0.) for m in rows)/n for k in ('height','xy_rmse','yaw_rmse','reward','falls')}
        quality=self.quality_diagnostics()
        groups=self.env.flush_training_groups()
        result={'iteration':self.it,'elapsed_seconds':time.monotonic()-self.start_time,'counts':self.counts.copy(),
            'metrics':metrics,'falls_per_env_min':metrics['falls']/self.cfg.num_envs*60/self.cfg.dt,
            'archive_size':self.archive.size,'archive_retention':self.archive.statistics(self.buf.total_collections),
            'nonfinite_skipped':0,'terrain_levels':self.env.terrain_level_statistics,**stats,**quality,
            'reward_weights':reward_weights(self.it),'task_metrics':[r.get('task_metrics',{}) for r in rows]}
        if quality and self.it%50==0:
            self.schedule.observe({'iteration':self.it,**quality,'reward':metrics['reward'],'xy_rmse':metrics['xy_rmse'],
                'falls':result['falls_per_env_min'],'critic_error':self._quality.get('critic_error'),'groups':groups})
            adjustment=self.schedule.adjust(self.it)
            if adjustment:
                with (self.output/'lr_feedback.jsonl').open('a') as f:f.write(json.dumps(adjustment,allow_nan=False)+'\n')
                result['lr_feedback']=adjustment
        with (self.output/'training.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
        (self.output/'progress.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)
        self.metrics=[];self.window_stats=[]

    def save(self,path=None):
        path=Path(path) if path else self.output/f'model_{self.it:08d}.pt'
        value={'schema':self.schema,'upstream_commit':UPSTREAM_COMMIT,'cfg':asdict(self.cfg),'iteration':self.it,
            'policy':self.policy.state_dict(),'critic':self.critic.state_dict(),'world_model':self.wm.state_dict(),
            'optimizers':{k:getattr(self,'opt_'+k).state_dict() for k in ('wm','actor','critic','cts')},
            'counts':self.counts,'schedule':self.schedule.state_dict(),'curriculum':self.env.curriculum_state_dict(),
            'value_guard':{'streak':self.imag.value_guard_streak,'last':self.imag.last_value_guard},
            'action_quality':self.action_quality,'cts_stable':self.cts_stable,'fixed_reports':self.fixed_reports,
            'allocator':None if self.allocator is None else self.allocator.state_dict(),
            'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'physics':'isaacsim_physx','depth':self.depth.metadata,'consts':{k:v.detach().cpu() for k,v in vars(self.env.consts).items()},
            'resume_mode':'fresh simulator episodes and Replay, current models/optimizers/normalizers/curriculum retained'}
        partial=path.with_suffix('.partial');torch.save(value,partial);os.replace(partial,path);self.last_checkpoint=path
        return path

    def load(self,path,continue_run=False):
        ck=torch.load(path,map_location=self.device,weights_only=False)
        if ck['schema']!=self.schema or ck['cfg']['variant']!=self.cfg.variant:raise ValueError('Architecture/experiment mismatch')
        for name,expected in ck['consts'].items():
            if not torch.allclose(getattr(self.env.consts,name),expected.to(self.device)):raise ValueError('Physical constants changed: '+name)
        self.policy.load_state_dict(ck['policy']);self.wm.load_state_dict(ck['world_model']);self.critic.load_state_dict(ck['critic'])
        for k in ('wm','actor','critic','cts'):getattr(self,'opt_'+k).load_state_dict(ck['optimizers'][k])
        self.schedule.load_state_dict(ck['schedule']);self.it=ck['iteration'];self.counts=ck['counts']
        guard=ck.get('value_guard',{});self.imag.value_guard_streak=int(guard.get('streak',0));self.imag.last_value_guard=guard.get('last',{})
        self.action_quality.copy_(ck['action_quality']);self.cts_stable=ck['cts_stable'];self.fixed_reports=ck['fixed_reports']
        if ck.get('allocator') is not None:
            from .terrain_allocation import TerrainAllocator
            self.allocator=TerrainAllocator(ck['allocator']['base']);self.allocator.load_state_dict(ck['allocator'])
        self.env.load_curriculum_state_dict(ck['curriculum']);self.env.iteration=self.it
        torch.set_rng_state(ck['rng'].cpu())
        if ck['cuda_rng']:torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
        self.env.reset(self.env.ids);self.env._snapshot=self.env._observe();self.refresh_providers();self.reset_rollout_state()
        self._resume_refill_steps=max(self.cfg.wm_window,self.cfg.cts_burnin+self.cfg.cts_sequence_steps)
        self.last_checkpoint=Path(path)

    @torch.no_grad()
    def reset_rollout_state(self):
        self.real_hidden.zero_();self.real_depth.zero_();self.ou.zero_();self.camera_episode.fill_(-1)

    @torch.no_grad()
    def policy_action(self,snapshot,branch,hidden,depth=None,policy=None):
        policy=self.policy if policy is None else policy
        if depth is None and branch==1:
            depth=self.depth.captured(snapshot['pose'],flat_ids(snapshot['scene_ids']),flat_ids(snapshot['frame_index']),
                flat_ids(snapshot['episode']),torch.arange(len(snapshot['s']),device=self.device))
        priv=st.privileged(snapshot['s'],snapshot['a_prev'],snapshot['command'],snapshot['xi'],snapshot['relief'])
        height=geometry_crop(snapshot['s'],snapshot['pose'],self.env.field,kind='teacher') if branch==0 else snapshot['s'].new_zeros((len(priv),2,1,1))
        action,hidden,_,_=policy.step(snapshot['obs'],priv,branch,depth=depth,hidden=hidden,height=height)
        return action,hidden

    def run(self,iterations=None):
        end=self.cfg.iterations if iterations is None else self.it+iterations
        for _ in range(getattr(self,'_resume_refill_steps',0)):self.collect()
        self._resume_refill_steps=0
        while self.it<end:
            if getattr(self,'stop_requested',False):self.save();break
            self.it+=1;self.env.iteration=self.it;stats={'lr':self.learning_rates()};timings={}
            tick=time.monotonic();self.collect();timings['collect_seconds']=time.monotonic()-tick
            if self.buf.size>=self.cfg.wm_window:
                tick=time.monotonic();stats['wm_loss']=self.fit_world_model();timings['wm_seconds']=time.monotonic()-tick
            if self.it>=self.cfg.warmup_iterations:
                tick=time.monotonic();stats.update(self.improve_policy(detailed_timing=self.it%self.cfg.log_interval==0));timings['actor_critic_seconds']=time.monotonic()-tick
            if self.it>=self.cfg.policy_start and self.buf.size>=self.cfg.cts_burnin+self.cfg.cts_sequence_steps:
                tick=time.monotonic();stats['cts_loss']=self.update_cts();stats.update(self._cts_stats);timings['cts_seconds']=time.monotonic()-tick
            stats['phase_times']=timings
            if self.it==1 or self.it%self.cfg.log_interval==0 or self.it==end:self.log(stats)
            if self.it==1 or self.it%self.cfg.save_interval==0 or self.it==end:self.save()
            if self.it%self.cfg.evaluation_interval==0:
                from .diagnostics import native_diagnostics
                report=native_diagnostics(self)
                (self.output/f'native_{self.it:08d}.json').write_text(json.dumps(report,indent=2))
                self.consume_fixed_report(report)
        return self.last_checkpoint

    def consume_fixed_report(self,report):
        terrains=report.get('terrains',{})
        for k,r in terrains.items():
            if r.get('valid'):self.action_quality[int(k)]=max(0.,min(1.,(r['branch_rates']['teacher']-.5)/.3))
        old=self.fixed_reports[-1] if self.fixed_reports else None
        self.cts_stable=bool(old and terrains and all(r.get('valid') and old.get(k,{}).get('cohort_id')==r['cohort_id']
            and r['branch_rates']['teacher']>=.8 and old[k]['branch_rates']['teacher']>=.8
            and r['branch_rates']['teacher']-r['branch_rates']['student']<=.1
            and abs(r['branch_rates']['teacher']-old[k]['branch_rates']['teacher'])<.1 for k,r in terrains.items()))
        self.fixed_reports=(self.fixed_reports+[terrains])[-2:]
        if self.it>=self.cfg.all_terrain_iteration:
            from .terrain_allocation import TerrainAllocator
            counts=self.env.allocation_counts()
            if self.allocator is None:self.allocator=TerrainAllocator(counts)
            record=self.allocator.observe(self.it,terrains,counts,self.env.allocation_pending())
            if record:
                if record['changed']:self.env.queue_allocation(record['target'])
                with (self.output/'terrain_allocation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')

    def export(self,path):
        module=copy.deepcopy(StudentExport(self.policy)).eval().cpu()
        inputs=(torch.zeros(1,450),torch.zeros(1,1,48,64),torch.zeros(1,256),torch.zeros(1,dtype=torch.bool))
        torch.jit.trace(module,inputs,check_trace=True).save(str(path));return path
