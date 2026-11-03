"""Source-paced visual multitask BPTT with explicitly separated parameter owners."""
import copy,json,math,os,time
from contextlib import contextmanager,nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from . import UPSTREAM_COMMIT,state as st
from .networks import Policy,Critic,StudentExport
from .world_model import WorldModel,supervised_loss
from .buffer import Replay,Archive,mixed_windows
from .imagine import Imagination,flat_ids
from .terrain import relief,geometry_crop
from .depth import DepthObservation
from .schedules import LearningRateController,ProviderSchedule
from .lr_feedback import UpdateFeedback
from .task_allocation import TaskAllocator
from .commands import phase
from .sequence_execution import depth_sequence_features,batched_apply,ENCODING_CHUNK


@contextmanager
def frozen(*modules):
    params=[p for m in modules for p in m.parameters()]; flags=[p.requires_grad for p in params]
    for p in params: p.requires_grad_(False); p.grad=None
    try: yield
    finally:
        for p,flag in zip(params,flags): p.requires_grad_(flag)


def clip_grad(params,global_norm=1.,elem=float('inf')):
    params=[p for p in params if p.grad is not None]
    norm=torch.nn.utils.clip_grad_norm_(params,global_norm)
    for p in params: p.grad.clamp_(-elem,elem)
    return norm


def scalar_finite(x): return bool(torch.isfinite(x).all())


def provider_assignment(n,driver_frac,teacher_fraction,device):
    source=torch.zeros(n,dtype=torch.uint8,device=device)
    nd=int(n*driver_frac)
    if nd: source[torch.linspace(0,n-1,nd,device=device).round().long()]=2
    own=(source==0).nonzero().flatten(); ns=len(own)-int(len(own)*teacher_fraction)
    if ns: source[own[torch.linspace(0,len(own)-1,ns,device=device).round().long()]]=1
    return source


def branch_loss(objectives,branch,sizes):
    if isinstance(branch,tuple):
        return -objectives[:sizes[0]].sum()/sizes[0]-objectives[sizes[0]:].sum()/sizes[1]
    return -objectives.sum()/sizes[branch]


def subset(data,index):
    return {k:subset(v,index) if isinstance(v,dict) else v[index] for k,v in data.items()}


class Learner:
    def __init__(self,cfg,env,output,driver=None):
        self.cfg,self.env,self.output=cfg,env,Path(output); self.output.mkdir(parents=True,exist_ok=True)
        self.device=cfg.device; torch.manual_seed(cfg.seed)
        mixed_precision=getattr(cfg,'mixed_precision',True)
        self.policy=Policy(cfg.actor_hidden,experts=cfg.actor_experts,mixed_precision=mixed_precision).to(cfg.device)
        self.critic=Critic(cfg.critic_hidden,env.field,cfg.task_context_dim,compile=cfg.compile,mixed_precision=mixed_precision).to(cfg.device)
        self.wm=WorldModel(cfg.wm_hidden,mixed_precision=mixed_precision).to(cfg.device)
        self._wm_loss_fn=torch.compile(supervised_loss,dynamic=False,fullgraph=True) if cfg.compile else supervised_loss
        self.depth=DepthObservation(cfg,env.field)
        self.actor_params=list(self.policy.teacher_encoder.parameters())+list(self.policy.actor.parameters())
        self.opt_wm=torch.optim.Adam(self.wm.parameters(),lr=cfg.wm_lr)
        self._new_policy_optimizers()
        self.buf=Replay(cfg.num_envs,cfg.buffer_len,cfg.device,cfg.lr_holdout_modulus)
        self.archive=Archive(cfg.archive_size,cfg.wm_window,cfg.device)
        self.imag=Imagination(cfg,env.consts,env.field,env.command_sampler,self.depth)
        self.schedule=LearningRateController(cfg)
        self.feedback=UpdateFeedback(self)
        self.task_allocator=TaskAllocator()
        self.driver=driver
        if cfg.driver_frac and driver is None: raise ValueError('Declared official driver is missing')
        self.index=torch.arange(cfg.num_envs,device=cfg.device)
        self.source=torch.zeros(cfg.num_envs,dtype=torch.uint8,device=cfg.device)
        self.providers=ProviderSchedule(cfg)
        self.ou=torch.zeros((cfg.num_envs,12),device=cfg.device)
        self.real_hidden=torch.zeros((cfg.num_envs,256),device=cfg.device)
        self.real_depth=torch.zeros((cfg.num_envs,1,48,64),device=cfg.device)
        self.depth_pose=torch.zeros((cfg.num_envs,7),device=cfg.device)
        self.depth_scene_ids=torch.zeros(cfg.num_envs,dtype=torch.long,device=cfg.device)
        self.depth_frame_index=torch.zeros_like(self.depth_scene_ids)
        self.camera_episode=torch.full_like(self.depth_scene_ids,-1)
        self.it=0; self.actor_frozen_until=0
        self.last_good=None; self.collapse_strikes=0; self.nonfinite=0; self.teacher_stable=False
        self.collapse_history=[]
        self.counts=dict(collect=0,transitions=0,wm=0,actor=0,critic=0,cts=0)
        self.metrics=[]; self.start_time=time.monotonic(); self.provider_counts=[0,0,0]
        self.refresh_providers()
        (self.output/'resolved_config.json').write_text(json.dumps(asdict(cfg),indent=2))
        (self.output/'run_identity.json').write_text(json.dumps({'upstream_commit':UPSTREAM_COMMIT,
            'physics':'isaacsim_physx','imaginary_branch_samples':[6144,2048],
            'branch_loss':'teacher_loss.mean() + student_loss.mean()',
            'replay_tensor_bytes':self.buf.storage_bytes,'init':'cold',
            'expert_missing':['gap','recovery','visual_avoidance','obstacle_up','obstacle_down'],
            'student_update':'latent MSE only; input and GRU derivatives retained in actor BPTT',
            'curriculum_clock':'learner iteration, one real vector step per iteration'},indent=2))
    def _new_policy_optimizers(self):
        self.opt_actor=torch.optim.Adam([{'params':self.policy.actor.parameters(),'lr':self.cfg.lr,'name':'actor'},
            {'params':self.policy.teacher_encoder.parameters(),'lr':self.cfg.lr,'name':'teacher'}],betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=self.cfg.lr)
        self.opt_cts=torch.optim.Adam(self.policy.student_encoder.parameters(),lr=self.cfg.cts_lr)
    def refresh_providers(self,before=None):
        before=self.env.snapshot() if before is None else before
        tasks=flat_ids(before['task_ids']); old=self.source.clone(); self.source.zero_()
        for task in range(10):
            ids=(tasks==task).nonzero().flatten()
            if not len(ids): continue
            frac=self.providers.fraction(task,self.it)
            self.source[ids]=provider_assignment(len(ids),frac,self.cfg.teacher_fraction,self.device)
        self.provider_requested=self.source.clone()
        self._provider_task_version=self.env.task_assignment_version
        self._provider_refresh_bucket=self.it//1000
        self._provider_phase=phase(self.env.iteration)
        self._provider_before_stop=self.it<self.providers.stop_iteration
        changed=self.source!=old; self.real_hidden[changed]=0; self.camera_episode[changed]=-1
        self.provider_counts=[int((self.source==i).sum()) for i in range(3)]
        self.env.own_mask=self.source<2
    def sync_providers(self,before):
        # snapshot() may change tasks/reset episodes even when the caller skips
        # a curriculum boundary. Refresh from the actual assignment, not a list
        # of exact iteration numbers.
        if (phase(self.env.iteration)!=self._provider_phase
            or (self.it<self.providers.stop_iteration)!=self._provider_before_stop
            or (self.it<self.providers.stop_iteration and self.it//1000!=self._provider_refresh_bucket)
            or self.env.task_assignment_version!=self._provider_task_version):
            self.refresh_providers(before)
    def learning_rates(self):
        rates=self.schedule.rates(self.it)
        if self.it<self.actor_frozen_until: rates['actor']=rates['teacher']=0.
        for opt,name in ((self.opt_wm,'wm'),(self.opt_critic,'critic'),(self.opt_cts,'cts')):
            for group in opt.param_groups: group['lr']=rates[name]
        for group in self.opt_actor.param_groups: group['lr']=rates[group['name']]
        return rates
    @torch.no_grad()
    def camera_inputs(self,before):
        frame=flat_ids(before['frame_index']); episode=flat_ids(before['episode'])
        episode_changed=episode!=self.camera_episode
        # Curriculum phase changes can reset inside snapshot(), before step()
        # returns a done mask. Recurrent memory must follow episode identity.
        self.real_hidden[episode_changed]=0
        if self.driver is not None and self.it<self.providers.stop_iteration:self.driver.reset(episode_changed)
        refresh=(frame%self.cfg.depth_period==0)|episode_changed
        self.depth_pose[refresh]=before['pose'][refresh]
        self.depth_scene_ids[refresh]=flat_ids(before['scene_ids'])[refresh]
        self.depth_frame_index[refresh]=frame[refresh]
        self.camera_episode.copy_(episode)
        ids=(refresh&(self.source==1)).nonzero().flatten()
        if len(ids):
            self.real_depth[ids]=self.depth.captured(self.depth_pose[ids],self.depth_scene_ids[ids],self.depth_frame_index[ids],episode[ids],self.index[ids]).float()
        return self.real_depth
    @torch.no_grad()
    def collect(self):
        cfg=self.cfg; before=self.env.snapshot()
        self.sync_providers(before)
        # Official walking data are not represented as a validated5m/s expert.
        # Eligibility is decided from this exact frame's command, and Replay
        # and curriculum statistics record the actually executed provider.
        self.source=self.provider_requested.clone()
        if self.it<self.providers.stop_iteration:
            unsupported=((before['command'][:,:2].norm(dim=-1)>4.)
                         |(flat_ids(before['task_ids'])>=5))&(self.source==2)
            self.source[unsupported]=0
        self.env.own_mask=self.source<2
        source_counts=torch.bincount(self.source.long(),minlength=3)
        priv=st.privileged(before['s'],before['a_prev'],before['command'],before['xi'],before['relief'])
        depth=self.camera_inputs(before); a=torch.empty((cfg.num_envs,12),device=self.device)
        for branch in (0,1):
            ids=(self.source==branch).nonzero().flatten()
            if not len(ids): continue
            height=geometry_crop(before['s'][ids],before['pose'][ids],self.env.field,kind='teacher') if branch==0 else before['s'].new_zeros((len(ids),2,1,1))
            action,hidden,_,_=self.policy.step(before['obs'][ids],priv[ids],branch,depth=depth[ids],hidden=self.real_hidden[ids],
                height=height,task_ids=flat_ids(before['task_ids'])[ids])
            a[ids]=action; self.real_hidden[ids]=hidden
        if self.driver is not None and self.it<self.providers.stop_iteration and bool((self.source==2).any()):
            external=self.driver.act(before['command']); a[self.source==2]=external[self.source==2]
        sigma=cfg.explore_noise*math.sqrt(2*cfg.explore_theta)
        self.ou=self.ou*(1-cfg.explore_theta*cfg.dt)+sigma*math.sqrt(cfg.dt)*torch.randn_like(a)
        noise=torch.where((self.index<int(cfg.num_envs*cfg.explore_frac))[:,None],self.ou,cfg.action_noise*torch.randn_like(a))
        a=a+noise
        if not scalar_finite(a): raise FloatingPointError('Nonfinite real action')
        after,done,metrics=self.env.step(a,defer_metrics=True)
        # `after` is pre-reset; snapshot() is the next episode when done is true.
        values={k:before[k] for k in self.buf.data if k in before}
        values.update(s_next=after['s'],pose_next=after['pose'],a=a,source=self.source,done=done,
            terminated=after['terminated'],timeout=after['timeout'],success=after['success'],
            task_context_next=after['task_context'],next_contact_prob=after['contact_prob'],
            next_applied_torque=after['applied_torque'],next_foot_velocity=after['foot_velocity'],
            depth_pose=self.depth_pose,depth_scene_ids=self.depth_scene_ids,depth_frame_index=self.depth_frame_index)
        self.buf.add(**values)
        training=~self.buf.validation_episodes(before['episode'])
        telemetry={k:after[k][training] for k in ('contact_prob','applied_torque','foot_velocity')}
        self.wm.update_stats(before['s'][training],a[training],before['a_prev'][training],before['xi'][training],None,after['s'][training],
            st.pose_delta(before['pose'][training],after['pose'][training]),telemetry,pose=before['pose'][training])
        self.policy.update_stats(before['obs'][training],priv[training])
        self.ou[done]=0; self.real_hidden[done]=0; self.camera_episode[done]=-1
        if self.driver is not None and self.it<self.providers.stop_iteration: self.driver.reset(done)
        self.counts['collect']+=1; self.counts['transitions']+=cfg.num_envs
        metrics['source_counts']=source_counts; self.metrics.append(metrics)
    def fit_world_model(self):
        losses=[]
        for _ in range(self.cfg.wm_updates_for(self.it)):
            w=mixed_windows(self.buf,self.archive,self.cfg,self.env.xi)
            loss=self._wm_loss_fn(self.wm,w,self.env.field,self.env.consts.default_q)
            self.opt_wm.zero_grad(set_to_none=True); loss.backward(); gn=clip_grad(self.wm.parameters(),self.cfg.grad_clip)
            if scalar_finite(gn) and scalar_finite(loss): self.opt_wm.step(); self.counts['wm']+=1
            else: self.nonfinite+=1
            losses.append(loss.detach())
        return sum(torch.stack(losses).tolist())/len(losses)
    def _actor_ready(self,seed):
        tasks=flat_ids(seed['task_ids'])
        activation=tasks.new_tensor([1000,5000,5000,5000,5000,8000,8000,8000,8000,8000])
        return (self.it>=activation[tasks]+self.cfg.critic_warmup_iterations).to(seed['s'].dtype)
    def improve_policy(self,*,detailed_timing=True):
        cfg=self.cfg; nt=int(cfg.imag_batch*cfg.teacher_fraction); sizes=(nt,cfg.imag_batch-nt)
        objective_means=[]
        use_cuda=str(self.device).startswith('cuda'); spans=[]
        def stamp():
            if not detailed_timing:return None
            if not use_cuda:return time.monotonic()
            mark=torch.cuda.Event(enable_timing=True);mark.record();return mark
        def span(name,start):
            if detailed_timing:spans.append((name,start,stamp()))
        actor_active=self.it>=cfg.policy_start and self.it>=self.actor_frozen_until
        if actor_active:self.feedback.before_actor()
        self.opt_actor.zero_grad(set_to_none=True)
        with frozen(self.wm,self.critic,self.policy.student_encoder):
            sample_start=stamp()
            teacher_seed=self.buf.seeds(sizes[0],burnin=0)
            student_seed=self.buf.seeds(sizes[1],burnin=cfg.cts_burnin)
            seed={k:torch.cat((value,student_seed[k]),0) for k,value in teacher_seed.items()}
            seed['student_prefix']=student_seed['prefix']
            span('seed_sampling',sample_start)
            if detailed_timing:
                source_counts=torch.bincount(flat_ids(seed['source']),minlength=3)
                imagined_tasks=torch.bincount(flat_ids(seed['task_ids']),minlength=10).tolist()
            # One fused mixed cohort, not separate teacher/student rollouts or
            # renamed microbatching. Both branch means retain their own divisor.
            with torch.enable_grad() if actor_active else torch.no_grad():
                forward_start=stamp()
                obj,data=self.imag.rollout(self.wm,self.policy,self.critic,seed,sizes,iteration=self.it)
                span('mixed_imagination_forward',forward_start)
                if actor_active:
                    backward_start=stamp();branch_loss(obj*self._actor_ready(seed),sizes,sizes).backward()
                    span('mixed_imagination_backward',backward_start)
            objective_means=torch.stack((obj[:nt].float().mean().detach(),obj[nt:].float().mean().detach())).tolist()
            del teacher_seed,student_seed,seed
            actor_step_start=stamp()
            gn=clip_grad(self.actor_params,cfg.grad_clip,cfg.grad_clip_elem) if actor_active else torch.tensor(0.)
            finite=scalar_finite(gn) and all(math.isfinite(v) for v in objective_means)
            if actor_active and finite: self.opt_actor.step(); self.counts['actor']+=1
            elif not finite: self.nonfinite+=1
            span('actor_clip_and_step',actor_step_start)
        # The detached TD records are sufficient below. Drop the completed
        # BPTT graph and consumed Actor gradients before allocating map blocks.
        del obj
        self.opt_actor.zero_grad(set_to_none=True)
        if actor_active and finite:self.feedback.after_actor()
        else:self.feedback.actor_pending=None
        self.feedback.before_critic()
        geometry_start=stamp()
        geometry=self.critic.prepare_geometry(data['critic_in'])
        span('critic_geometry_cache',geometry_start)
        target_start=stamp();target=self.imag.targets(self.critic,data,geometry=geometry);span('critic_targets',target_start)
        x=data['critic_in'][:,:-1].reshape(-1,self.critic.feature_dim); y=target.reshape(-1)
        valid=data['critic_valid'].reshape(-1).bool(); x,y=x[valid],y[valid]
        # Index the originalH+1 cache; do not duplicate its multi-GB pixel arrays.
        geometry_indices=torch.arange(data['critic_in'].shape[0]*data['critic_in'].shape[1],device=self.device)
        geometry_indices=geometry_indices.reshape(data['critic_in'].shape[:2])[:,:-1].reshape(-1)[valid]
        if not len(x): raise RuntimeError('No valid imagined critic samples')
        popart_start=stamp()
        self.critic.norm.update(x[:,:self.critic.physical_dim])
        self.critic.update_popart(x,y);span('critic_normalization_and_popart',popart_start)
        fit_start=stamp();losses=[];critic_samples_processed=0;critic_batch_sizes=[]
        for _ in range(cfg.critic_epochs):
            order=torch.randperm(len(x),device=self.device)
            for ids in torch.tensor_split(order,cfg.critic_minibatches):
                critic_samples_processed+=len(ids);critic_batch_sizes.append(len(ids))
                # Group once per original minibatch, then run larger single-task
                # blocks. Its members, denominator, clipping and Adam step stay
                # unchanged; the random permutation is still drawn once/epoch.
                self.opt_critic.zero_grad(set_to_none=True); total=x.new_zeros((),dtype=torch.float64)
                task_ids=x[ids,-1].long(); grouped=ids[torch.argsort(task_ids,stable=True)]
                task_counts=torch.bincount(task_ids,minlength=len(self.critic.tasks)).tolist()
                offset=0
                for task,count in enumerate(task_counts):
                    block=grouped[offset:offset+count]; offset+=count
                    for sub in block.split(getattr(cfg,'critic_microbatch',4096)):
                        if not len(sub): continue
                        loss=self.critic.loss_for_task(x[sub],y[sub],task,geometry=geometry,geometry_indices=geometry_indices[sub])
                        (loss*len(sub)/len(ids)).backward()
                        total.add_(loss.detach().double()*len(sub)/len(ids))
                cgn=clip_grad(self.critic.parameters(),cfg.grad_clip)
                total=float(total)
                if scalar_finite(cgn) and math.isfinite(total): self.opt_critic.step(); self.counts['critic']+=1
                else: self.nonfinite+=1
                losses.append(total)
        span('critic_fit',fit_start)
        # No terrain pixels or encoded features survive into the CTS update.
        del geometry,geometry_indices
        self.feedback.after_critic()
        if not detailed_timing:return {}
        if use_cuda and detailed_timing:torch.cuda.synchronize(self.device)
        update_times={}
        for name,start,end in spans:
            update_times[name]=update_times.get(name,0.)+(start.elapsed_time(end)/1000. if use_cuda else end-start)
        return {'branch_objectives':objective_means,'objective':sum(objective_means),
            'actor_grad_norm':float(gn),'critic_loss':sum(losses)/len(losses),'imagined_sources':source_counts.tolist(),'update_times':update_times,'critic_geometry_cache':self.critic.geometry_cache_metadata,
            'imagined_branch_samples':list(sizes),'imagined_tasks':imagined_tasks,'fused_imagination':True,
            'critic_epochs':cfg.critic_epochs,'critic_minibatches':cfg.critic_minibatches,
            'critic_valid_samples':len(x),'critic_samples_processed':critic_samples_processed,'critic_batch_sizes':critic_batch_sizes,
            'immediate':float(data['immediate'].mean()),'bootstrap':float(data['bootstrap'].mean())}
    def sequence_loss(self,student,sequence,training=True):
        cfg=self.cfg; total=sequence['obs'].shape[1]; b=len(sequence['obs'])
        hidden=sequence['obs'].new_zeros((b,256),dtype=torch.float32)
        burn=cfg.cts_burnin;steps=total-burn
        # Replay inputs are already known: batch all nonrecurrent operations,
        # while preserving ordered GRU resets and the configured16-step burn-in.
        with torch.no_grad():
            single=self.policy.single_norm(st.single_from_history(sequence['obs'].float().reshape(b*total,-1))).reshape(b,total,-1)
            burn_features=depth_sequence_features(student,self.depth,sequence,0,burn)
            for j in range(burn):
                hidden=student.advance_memory(single[:,j],burn_features[:,j],hidden,sequence['reset'][:,j].reshape(-1).bool())
            history=self.policy.history_norm(sequence['obs'][:,burn:].float().reshape(b*steps,-1))
            fields={k:sequence[k][:,burn:].reshape(b*steps,*sequence[k].shape[2:])
                for k in ('s','a_prev','command','xi','relief','pose','task_ids')}
            labels=[]
            for lo in range(0,b*steps,ENCODING_CHUNK):
                item={k:v[lo:lo+ENCODING_CHUNK] for k,v in fields.items()}
                priv=st.privileged(item['s'],item['a_prev'],item['command'],item['xi'],item['relief'])
                height=geometry_crop(item['s'],item['pose'],self.env.field,kind='teacher')
                labels.append(self.policy.teacher_latent(priv,height,flat_ids(item['task_ids'])))
            label=torch.cat(labels).reshape(b,steps,-1)
        del burn_features
        # Separate invocation deliberately rebuilds a differentiable feature at
        # the burn-in/loss boundary even if the captured camera frame is held.
        # No encoded feature survives this call or either CTS optimizer step.
        depth_features=depth_sequence_features(student,self.depth,sequence,burn,total,training=training)
        proprio=batched_apply(lambda hist:student.encode_proprio(hist)[0],history,training=training).reshape(b,steps,-1)
        losses=[]
        for j in range(steps):
            reset=sequence['reset'][:,burn+j].reshape(-1).bool()
            args=(proprio[:,j],single[:,burn+j],depth_features[:,j],hidden,reset)
            z,hidden=checkpoint(student.fuse_features,*args,use_reentrant=False) if training else student.fuse_features(*args)
            losses.append((z.float()-label[:,j].float()).square().mean(-1))
        loss=torch.stack(losses,1);mask=sequence['valid'][:,burn:].reshape(b,steps).float()
        return (loss*mask).sum()/mask.sum().clamp_min(1)
    def update_cts(self):
        if self.it<self.cfg.policy_start:return None
        losses=[]
        with frozen(self.wm,self.critic,self.policy.actor,self.policy.teacher_encoder):
            for _ in range(self.cfg.cts_updates):
                seq=self.buf.sequences(self.cfg.cts_batch,self.cfg.cts_sequence_steps,self.cfg.cts_burnin,source=1)
                loss=self.sequence_loss(self.policy.student_encoder,seq)
                self.opt_cts.zero_grad(set_to_none=True); loss.backward(); gn=clip_grad(self.policy.student_encoder.parameters(),self.cfg.grad_clip)
                if scalar_finite(gn) and scalar_finite(loss): self.opt_cts.step(); self.counts['cts']+=1
                else: self.nonfinite+=1
                losses.append(loss.detach())
        return sum(torch.stack(losses).tolist())/len(losses)
    def save(self,path=None):
        path=Path(path) if path else self.output/f'model_{self.it:08d}.pt'
        contract_path=self.output/'physics_contract.json'
        project_commit=json.loads(contract_path.read_text()).get('code_commit') if contract_path.exists() else None
        value={'schema':'physx-depth-multitask-v2','upstream_commit':UPSTREAM_COMMIT,'cfg':asdict(self.cfg),
            'project_code_commit':project_commit,
            'policy':self.policy.state_dict(),'critic':self.critic.state_dict(),'world_model':self.wm.state_dict(),
            'optimizers':{k:getattr(self,'opt_'+k).state_dict() for k in ('wm','actor','critic','cts')},
            'iteration':self.it,'counts':self.counts,'training_schedule_version':LearningRateController.version,
            'feedback':self.feedback.state_dict(),
            'task_allocator':self.task_allocator.state_dict(),
            'schedule':self.schedule.state_dict(),'teacher_stable':self.teacher_stable,
            'model_validation_history':getattr(self,'_model_validation_history',{}),
            'providers':self.providers.state_dict(),
            'curriculum':self.env.curriculum_state_dict(),'actor_frozen_until':self.actor_frozen_until,
            'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'resume_mode':'fresh simulator episodes/replay; models/optimizers/normalizers/task curriculum retained',
            'physics':'isaacsim_physx','depth':self.depth.metadata,'consts':{k:v.detach().cpu() for k,v in vars(self.env.consts).items()}}
        from .task import reward_budget
        value['reward_budget']=reward_budget()
        partial=path.with_suffix('.partial'); torch.save(value,partial); os.replace(partial,path); return path
    def load(self,path,continue_run=False):
        ck=torch.load(path,map_location=self.device,weights_only=False)
        if ck.get('training_schedule_version')!=LearningRateController.version:
            raise ValueError('Checkpoint predates observed-update LR schedule; explicit new initialization required')
        if ck['schema']!='physx-depth-multitask-v2' or ck['physics']!='isaacsim_physx': raise ValueError('Architecture mismatch; a matching expert-count/shared-Critic-geometry checkpoint is required')
        if ck['cfg']['actor_experts']!=self.cfg.actor_experts:raise ValueError('Actor expert count differs from checkpoint')
        for name,expected in ck['consts'].items():
            if not torch.allclose(getattr(self.env.consts,name),expected.to(getattr(self.env.consts,name)),atol=1e-6,rtol=1e-6): raise ValueError('Control constant mismatch: '+name)
        from .task import reward_budget
        if ck.get('reward_budget')!=reward_budget(): raise ValueError('Reward calibration differs from checkpoint')
        self.policy.load_state_dict(ck['policy']); self.wm.load_state_dict(ck['world_model']); self.critic.load_state_dict(ck['critic'])
        for k in ('wm','actor','critic','cts'): getattr(self,'opt_'+k).load_state_dict(ck['optimizers'][k])
        self.schedule.load_state_dict(ck['schedule']); self.teacher_stable=ck['teacher_stable']
        self.collapse_history=[]; self.collapse_strikes=0
        self._model_validation_history=ck.get('model_validation_history',{})
        self.it=ck['iteration']; self.counts=ck['counts']
        self.feedback.load_state_dict(ck['feedback'])
        if 'task_allocator' in ck:self.task_allocator.load_state_dict(ck['task_allocator'])
        self.actor_frozen_until=ck.get('actor_frozen_until',0)
        self.providers.load_state_dict(ck['providers'])
        self.env.load_curriculum_state_dict(ck['curriculum']); self.env.iteration=self.it
        torch.set_rng_state(ck['rng'].cpu())
        if ck['cuda_rng']: torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
        # Restoring into an existing Learner has the same fresh-data semantics
        # as a CLI resume. Allocated storage is reused without exposing old slots.
        self.buf.ptr=0; self.buf.size=0; self.buf.total_collections=0
        self.buf.event_history.zero_();self.buf.heldout.zero_()
        self.archive.ptr=0; self.archive.size=0; self.archive.total_admitted=0
        self.archive.admission_tick.zero_(); self.metrics.clear()
        self.env.reset(self.env.ids); self.env._snapshot=self.env._observe(); self.refresh_providers()
        self.reset_rollout_state(); self.last_good=Path(path)
        # Checkpoints intentionally omit Replay. Refill a complete student
        # sequence before any resumed optimization, without advancing curriculum
        # or LR clocks. Otherwise the first Actor update uses one reset frame
        # while its student burn-in and the dynamics window are still missing.
        self._resume_refill_steps=max(self.cfg.wm_window,self.cfg.cts_burnin+self.cfg.cts_sequence_steps)
        identity_path=self.output/'run_identity.json'
        identity=json.loads(identity_path.read_text())
        identity.update(init='resume',checkpoint=str(Path(path).resolve()),resume_iteration=self.it,
                        replay_restored=False,refill_collections=self._resume_refill_steps)
        identity_path.write_text(json.dumps(identity,indent=2))
    def rollback(self,path=None):
        path=path or self.last_good
        if path is None: return False
        ck=torch.load(path,map_location=self.device,weights_only=False)
        self.policy.load_state_dict(ck['policy']); self.critic.load_state_dict(ck['critic'])
        self.schedule.rollback(self.it)
        self.feedback.reset()
        self.collapse_strikes=0; self.collapse_history=[]
        # Preserve optimizer moments and the original ramp/freeze clocks.
        self.learning_rates()
        # Recurrent state encoded by replaced student weights is no longer valid.
        self.real_hidden.zero_(); self.camera_episode.fill_(-1)
        return True
    def log(self,stats):
        from .physx_env import materialize_metrics
        self.metrics=materialize_metrics(self.metrics)
        if self.metrics and 'source_counts' in self.metrics[-1]:self.provider_counts=self.metrics[-1]['source_counts']
        metrics=self.metrics; count=max(1,len(metrics)); own=max(1,self.provider_counts[0]+self.provider_counts[1])
        falls=sum(m.get('falls',0) for m in metrics)/count
        result={'iteration':self.it,'elapsed_seconds':time.monotonic()-self.start_time,'counts':self.counts.copy(),
            'metrics':{k:sum(m.get(k,0.) for m in metrics)/count for k in ('height','xy_rmse','yaw_rmse','reward','falls')},
            'falls_per_env_min':falls/own*60/self.cfg.dt,'archive_size':self.archive.size,'nonfinite_skipped':self.nonfinite,**stats}
        result['terrain_levels']=self.env.terrain_level_statistics
        result['archive_retention']=self.archive.statistics(self.buf.total_collections)
        from .task import reward_weights
        result['reward_weights']={k:float(v) for k,v in reward_weights(self.it).items()}
        # Retain per-task diagnostics; only the approved global joint trend triggers rollback.
        task_rows={}
        for metric in metrics:
            for task,row in metric.get('task_metrics',{}).items():
                task_rows.setdefault(str(task),[]).append(row)
        result['task_metrics']=task_rows
        falls_all=sum(m['falls_all'] for m in metrics)/count
        result['falls_per_vector_step_all_providers']=falls_all
        from .schedules import collapse_trend
        active=self.it>max(self.cfg.warmup_iterations+self.cfg.critic_warmup_iterations,self.actor_frozen_until)
        if active and len(metrics)==self.cfg.log_interval and self.it%self.cfg.log_interval==0:
            row={'iteration':self.it,'falls':falls_all,'xy_rmse':result['metrics']['xy_rmse'],'reward':result['metrics']['reward']}
            if self.collapse_history and self.it-self.collapse_history[-1]['iteration']!=self.cfg.log_interval:self.collapse_history=[]
            self.collapse_history=(self.collapse_history+[row])[-3:]
        else:self.collapse_history=[]
        self.metrics=[]
        trigger=collapse_trend(self.collapse_history,self.cfg.collapse_falls)
        self.collapse_strikes=int(active and falls_all>self.cfg.collapse_falls)
        result['rollback_window_history']=list(self.collapse_history)
        result['rollback_triggered']=trigger
        result['rollback_checkpoint']=str(self.last_good) if trigger and self.last_good else None
        result['rolled_back']=self.rollback() if trigger else False
        result['lr_after_rollback']=self.schedule.rates(self.it) if result['rolled_back'] else None
        with (self.output/'training.jsonl').open('a') as f: f.write(json.dumps(result,allow_nan=False)+'\n')
        (self.output/'progress.json').write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result),flush=True)
    def run(self,iterations=None):
        end=self.cfg.iterations if iterations is None else self.it+iterations
        refill=getattr(self,'_resume_refill_steps',0)
        if refill:
            print(json.dumps({'resume_replay_refill':refill,'iteration_held':self.it}),flush=True)
            for _ in range(refill):
                if getattr(self,'stop_requested',False):return self.last_good
                self.collect()
            self.metrics.clear()
            self._resume_refill_steps=0
        while self.it<end:
            if getattr(self,'stop_requested',False): self.save(); break
            self.it+=1; self.env.iteration=self.it; stats={'lr':self.learning_rates()}; timings={}
            tick=time.monotonic(); self.collect(); timings['collect_seconds']=time.monotonic()-tick
            if self.buf.size>=self.cfg.wm_window:
                tick=time.monotonic(); stats['wm_loss']=self.fit_world_model(); timings['wm_seconds']=time.monotonic()-tick
            if self.it>=self.cfg.warmup_iterations:
                tick=time.monotonic(); stats.update(self.improve_policy(detailed_timing=self.it%self.cfg.log_interval==0 or self.it==end)); timings['actor_critic_seconds']=time.monotonic()-tick
            if self.it>=self.cfg.policy_start and self.buf.size>=self.cfg.cts_burnin+self.cfg.cts_sequence_steps:
                tick=time.monotonic(); stats['cts_loss']=self.update_cts(); timings['cts_seconds']=time.monotonic()-tick
            stats['phase_times']=timings
            if self.it%self.cfg.log_interval==0:stats['task_allocation']=self.env.allocation_status()
            feedback=self.feedback.finish()
            if feedback is not None:
                stats['lr_feedback']=feedback
                with (self.output/'lr_feedback.jsonl').open('a') as f:f.write(json.dumps(feedback,allow_nan=False)+'\n')
            if self.it==1 or self.it%self.cfg.log_interval==0 or self.it==end: self.log(stats)
            path=None
            if self.it==1 or self.it%self.cfg.save_interval==0 or self.it==end:
                path=self.save()
                if self.it>=self.cfg.policy_start and not self.collapse_strikes: self.last_good=path
            if self.cfg.evaluation_python and self.it%self.cfg.evaluation_interval==0:
                self.evaluate_feedback(path if path is not None else self.save())
            previous_start=self.schedule.student_start
            self.schedule.check_student(self.it)
            self.teacher_stable=self.schedule.teacher_stable
            if self.schedule.student_start!=previous_start:
                self.save()  # Persist the gate decided from this iteration's new reports.
        return self.last_good
    def evaluate_feedback(self,path):
        from .robogauge_feedback import evaluate_checkpoint
        from .diagnostics import native_diagnostics,model_diagnostics
        out=self.output/f'robogauge_{self.it:08d}'; out.mkdir(parents=True,exist_ok=True)
        results={}
        for name,job in {'native':lambda:native_diagnostics(self),'model':lambda:model_diagnostics(self),
                'transfer':lambda:evaluate_checkpoint(path,out,self.it,self.cfg.evaluation_python)}.items():
            try:
                results[name]=job()
                if name!='transfer': (out/(name+'.json')).write_text(json.dumps(results[name],indent=2))
            except Exception as error:
                record={'error':str(error),'iteration':self.it,'training_continues':True}
                (out/(name+'_error.json')).write_text(json.dumps(record,indent=2)); print(json.dumps({'diagnostic_error':name,**record}),flush=True)
        # Stability and provider retirement require fixed real tests plus heldout
        # model quality; never infer competence from imagined return alone.
        for name,kind in (('native','teacher_fixed_eval'),('transfer','teacher_transfer_fixed_eval')):
            record=results.get(name,{}).get('teacher_fixed_eval')
            self.schedule.observe(dict(record,kind=kind) if record else
                                  {'kind':kind,'valid':False,'iteration':self.it})
        if 'native' in results and 'model' in results:
            self.consume_validation(results['native'],results['model'],results.get('transfer'))
        if self.it>=20000:
            report=results.get('transfer',{}).get('task_allocation',{'iteration':self.it,'tasks':{}})
            status=self.env.allocation_status()
            decision=self.task_allocator.observe(report,status['actual'],pending=bool(status['pending']))
            if decision is not None:
                if decision['changed']:self.env.request_task_allocation(decision['target'])
                decision['native_tasks']=results.get('native',{}).get('tasks',{})
                (out/'task_allocation.json').write_text(json.dumps(decision,indent=2,allow_nan=False))
                with (self.output/'task_allocation.jsonl').open('a') as f:f.write(json.dumps(decision,allow_nan=False)+'\n')
                self.save()  # Persist the decision AND its deferred reset transfers.
        (self.output/'latest_diagnostics.json').write_text(json.dumps({'iteration':self.it,'results':results,'automatic_transfer_rollback':False},indent=2))
    def consume_validation(self,native,model,transfer=None):
        fixed=native.get('teacher_fixed_eval')
        if fixed is not None: self.schedule.observe(fixed)
        if transfer is not None and transfer.get('teacher_fixed_eval') is not None:
            transfer_fixed=dict(transfer['teacher_fixed_eval'],kind='teacher_transfer_fixed_eval')
            self.schedule.observe(transfer_fixed)
        self.teacher_stable=self.schedule.teacher_stable
        for key,row in native.get('tasks',{}).items():
            task=int(key); model_row=model.get('task_validation',{}).get(str(task),{})
            self.providers.observe(task,self.it,row.get('own_success_rate',0.),
                model_row.get('valid',False),model_row.get('stable',False))
    @torch.no_grad()
    def reset_rollout_state(self):
        self.real_hidden.zero_(); self.real_depth.zero_(); self.ou.zero_(); self.camera_episode.fill_(-1)
        if self.driver is not None and self.it<self.providers.stop_iteration: self.driver.reset(torch.ones(self.cfg.num_envs,dtype=torch.bool,device=self.device))
    @torch.no_grad()
    def policy_action(self,snapshot,branch,hidden,depth=None,policy=None):
        policy=self.policy if policy is None else policy
        if depth is None and branch==0:
            depth=snapshot['s'].new_zeros((len(snapshot['s']),1,48,64))
        if depth is None:
            depth=self.depth.render(snapshot['pose'],flat_ids(snapshot['scene_ids']),flat_ids(snapshot['frame_index']),
                flat_ids(snapshot['episode']),torch.arange(len(snapshot['s']),device=self.device),mode='inference')
        priv=st.privileged(snapshot['s'],snapshot['a_prev'],snapshot['command'],snapshot['xi'],snapshot['relief'])
        height=geometry_crop(snapshot['s'],snapshot['pose'],self.env.field,kind='teacher')
        action,hidden,_,_=policy.step(snapshot['obs'],priv,branch,depth=depth,hidden=hidden,height=height,task_ids=flat_ids(snapshot['task_ids']))
        return action,hidden
    def export(self,path):
        module=copy.deepcopy(StudentExport(self.policy)).eval().cpu()
        inputs=(torch.zeros(1,450),torch.zeros(1,1,48,64),torch.zeros(1,256),torch.zeros(1,dtype=torch.bool))
        traced=torch.jit.trace(module,inputs,check_trace=True); traced.save(str(path)); return traced
