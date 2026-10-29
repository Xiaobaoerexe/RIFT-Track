"""Three-phase source loop plus explicit CTS update; no old SMCTS imports."""
import copy,json,math,os,time,subprocess,sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import torch
from . import UPSTREAM_COMMIT,state as st
from .networks import Policy,Critic,StudentExport
from .world_model import WorldModel
from .buffer import Replay,Archive,mixed_windows
from .imagine import Imagination
from .terrain import relief

@contextmanager
def frozen(*modules):
    params=[p for m in modules for p in m.parameters()];flags=[p.requires_grad for p in params]
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

class Learner:
    def __init__(self,cfg,env,output,driver=None,expert=None):
        self.cfg,self.env,self.output=cfg,env,Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.device=cfg.device;torch.manual_seed(cfg.seed)
        self.policy=Policy(cfg.actor_hidden).to(cfg.device);self.critic=Critic(cfg.critic_hidden).to(cfg.device)
        self.wm=WorldModel(cfg.wm_hidden).to(cfg.device)
        self.actor_params=list(self.policy.teacher_encoder.parameters())+list(self.policy.actor.parameters())
        self.opt_wm=torch.optim.Adam(self.wm.parameters(),lr=cfg.lr)
        self.opt_actor=torch.optim.Adam(self.actor_params,lr=cfg.lr,betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=cfg.lr)
        self.opt_cts=torch.optim.Adam(self.policy.student_encoder.parameters(),lr=cfg.cts_lr)
        self.buf=Replay(cfg.num_envs,cfg.buffer_len,cfg.device)
        self.archive=Archive(cfg.archive_size,cfg.wm_window,cfg.device)
        self.imag=Imagination(cfg,env.consts,env.field)
        self.driver,self.expert=driver,expert
        if cfg.driver_frac and driver is None:raise ValueError('External driver not loaded')
        if cfg.expert_frac and expert is None:raise ValueError('Second expert not loaded')
        self.index=torch.arange(cfg.num_envs,device=cfg.device)
        self.source=torch.zeros(cfg.num_envs,device=cfg.device,dtype=torch.uint8)
        n_driver=int(cfg.num_envs*cfg.driver_frac);n_expert=int(cfg.num_envs*cfg.expert_frac)
        own=cfg.num_envs-n_driver-n_expert
        self.source[int(own*cfg.teacher_fraction):own]=1
        if n_expert:self.source[own:own+n_expert]=3
        if n_driver:self.source[-n_driver:]=2
        self.ou=torch.zeros((cfg.num_envs,12),device=cfg.device)
        self.it=0;self.lr_scale=1.;self.ramp_start=cfg.warmup_iterations;self.actor_frozen_until=0
        self.last_good=None;self.collapse_strikes=0;self.nonfinite=0
        self.counts=dict(collect=0,transitions=0,wm=0,actor=0,critic=0,cts=0)
        self.metrics=[];self.start_time=time.monotonic()
        self.provider_counts=[int((self.source==i).sum()) for i in range(4)]
        self.env.own_mask=self.source<2
        (self.output/'resolved_config.json').write_text(json.dumps(asdict(cfg),indent=2))
        (self.output/'run_identity.json').write_text(json.dumps({'upstream_commit':UPSTREAM_COMMIT,'physics':'isaacsim_physx','providers':self.provider_counts,'imaginary_branch_samples':[int(cfg.imag_batch*cfg.teacher_fraction),cfg.imag_batch-int(cfg.imag_batch*cfg.teacher_fraction)],'branch_objective_weights':[cfg.teacher_fraction,1-cfg.teacher_fraction],'replay_tensor_bytes':self.buf.storage_bytes,'expert_missing':not bool(expert),'init':'cold','old_checkpoint_compatible':False},indent=2))

    def learning_rates(self):
        cfg=self.cfg;t=min(1.,self.it/cfg.lr_decay_iterations)
        lr=cfg.lr*(cfg.lr_final_frac+(1-cfg.lr_final_frac)*.5*(1+math.cos(math.pi*t)))*self.lr_scale
        ramp=min(1.,max(0.,self.it-max(self.ramp_start,self.actor_frozen_until,cfg.warmup_iterations))/max(1,cfg.actor_ramp_iterations))
        for opt,scale in [(self.opt_wm,1),(self.opt_actor,ramp*cfg.actor_lr_scale),(self.opt_critic,1)]:
            for g in opt.param_groups:g['lr']=lr*scale
        return {'wm':lr,'actor':lr*ramp*cfg.actor_lr_scale,'critic':lr,'cts':self.cfg.cts_lr,'actor_ramp':ramp}

    @torch.no_grad()
    def collect(self):
        cfg=self.cfg;before=self.env.snapshot()
        priv=st.privileged(before['s'],before['a_prev'],before['command'],self.env.xi,before['relief'])
        a=torch.empty((cfg.num_envs,12),device=self.device)
        for branch in (0,1):
            ids=self.source==branch
            if ids.any():a[ids]=self.policy(before['obs'][ids],priv[ids],branch)
        if self.driver is not None:
            external=self.driver.act(before['command']);a[self.source==2]=external[self.source==2]
        if self.expert is not None:
            external=self.expert(before['obs']);a[self.source==3]=external[self.source==3]
        sigma=cfg.explore_noise*math.sqrt(2*cfg.explore_theta)
        self.ou=self.ou*(1-cfg.explore_theta*cfg.dt)+sigma*math.sqrt(cfg.dt)*torch.randn_like(a)
        noise=torch.where((self.index<int(cfg.num_envs*cfg.explore_frac))[:,None],self.ou,cfg.action_noise*torch.randn_like(a))
        # Source overwrites the noisy learner actions with clean expert actions.
        a=a+noise*(self.source<2)[:,None]
        if not scalar_finite(a):raise FloatingPointError('Nonfinite real action')
        xi=self.env.xi.clone()
        after,done,metrics=self.env.step(a)
        self.buf.add(s=before['s'],pose=before['pose'],a=a,a_prev=before['a_prev'],obs=before['obs'],
                     command=before['command'],timer=before['timer'],done=done,relief=before['relief'],
                     source=self.source,episode=before['episode'],xi=xi)
        keep=~done
        if keep.any():self.wm.update_stats(before['s'][keep],a[keep],before['a_prev'][keep],xi[keep],before['relief'][keep],after['s'][keep],st.pose_delta(before['pose'][keep],after['pose'][keep]))
        self.policy.update_stats(before['obs'],priv)
        self.ou[done]=0
        if self.driver is not None:self.driver.reset(done)
        self.counts['collect']+=1;self.counts['transitions']+=cfg.num_envs
        metrics['source_counts']=self.provider_counts;self.metrics.append(metrics)

    def fit_world_model(self):
        losses=[]
        for _ in range(self.cfg.wm_updates):
            w=mixed_windows(self.buf,self.archive,self.cfg,self.env.xi)
            dp=torch.stack([st.pose_delta(w['pose'][:,j],w['pose'][:,j+1]) for j in range(self.cfg.wm_window)],1)
            loss=self.wm.loss(w['s'][:,0],w['a'],w['a_prev'],w['xi'],w['relief'].float(),w['s'][:,1:],dp,w['valid'])
            self.opt_wm.zero_grad(set_to_none=True);loss.backward();gn=clip_grad(self.wm.parameters(),self.cfg.grad_clip)
            if scalar_finite(gn) and scalar_finite(loss):self.opt_wm.step();self.counts['wm']+=1
            else:self.nonfinite+=1
            losses.append(float(loss.detach()))
        return sum(losses)/len(losses)

    def improve_policy(self):
        cfg=self.cfg;nt=int(cfg.imag_batch*cfg.teacher_fraction);sizes=(nt,cfg.imag_batch-nt)
        all_data=[];objective_means=[];source_counts=torch.zeros(4,device=self.device)
        self.opt_actor.zero_grad(set_to_none=True)
        with frozen(self.wm,self.critic,self.policy.student_encoder):
            groups=[((nt,cfg.imag_batch-nt),cfg.imag_batch)] if cfg.actor_microbatch>=cfg.imag_batch else list(enumerate(sizes))
            for branch,batch in groups:
                seed=self.buf.seeds(batch,self.env.xi)
                source_counts+=torch.bincount(seed['source'][:,0].long(),minlength=4)
                totals=[]
                for lo in range(0,batch,cfg.actor_microbatch):
                    sub={k:v[lo:lo+cfg.actor_microbatch] for k,v in seed.items()}
                    obj,data=self.imag.rollout(self.wm,self.policy,self.critic,sub,branch)
                    (-obj.sum()/cfg.imag_batch).backward();all_data.append(data);totals.append(obj.detach().sum())
                if isinstance(branch,tuple):objective_means=[float(obj[:nt].mean().detach()),float(obj[nt:].mean().detach())]
                else:objective_means.append(float(torch.stack(totals).sum()/batch))
            gn=clip_grad(self.actor_params,cfg.grad_clip,cfg.grad_clip_elem)
            finite=scalar_finite(gn) and all(math.isfinite(v) for v in objective_means)
            if self.it>self.actor_frozen_until and finite:self.opt_actor.step();self.counts['actor']+=1
            elif not finite:self.nonfinite+=1
        data={k:torch.cat([d[k] for d in all_data]) for k in all_data[0]}
        target=self.imag.targets(self.critic,data)
        x=data['critic_in'][:,:-1].reshape(-1,st.PRIV_DIM+2);y=target.reshape(-1)
        self.critic.norm.update(x)
        losses=[]
        for _ in range(cfg.critic_epochs):
            order=torch.randperm(len(x),device=self.device)
            for ids in torch.tensor_split(order,cfg.critic_minibatches):
                loss=(self.critic(x[ids])-y[ids]).square().mean()
                self.opt_critic.zero_grad(set_to_none=True);loss.backward();cgn=clip_grad(self.critic.parameters(),cfg.grad_clip)
                if scalar_finite(cgn) and scalar_finite(loss):self.opt_critic.step();self.counts['critic']+=1
                else:self.nonfinite+=1
                losses.append(float(loss.detach()))
        return {'branch_objectives':objective_means,'objective':sum(v*n/cfg.imag_batch for v,n in zip(objective_means,sizes)),
                'actor_grad_norm':float(gn),'critic_loss':sum(losses)/len(losses),'imagined_sources':source_counts.tolist(),
                'immediate':float(data['immediate'].mean()),'bootstrap':float(data['bootstrap'].mean())}

    def update_cts(self):
        cfg=self.cfg;losses=[]
        with frozen(self.wm,self.critic,self.policy.actor,self.policy.teacher_encoder):
            for _ in range(cfg.cts_updates):
                seed=self.buf.seeds(cfg.cts_batch,self.env.xi,source=1)
                priv=st.privileged(seed['s'],seed['a_prev'],seed['command'],seed['xi'],seed['relief'])
                with torch.no_grad():label=self.policy.teacher_encoder(self.policy.teacher_norm(priv))
                z,gate=self.policy.student_encoder(self.policy.history_norm(seed['obs']))
                latent=(z-label).square().mean();balance=(gate.mean(0)-1/8).square().mean()
                loss=latent+cfg.cts_balance*balance
                self.opt_cts.zero_grad(set_to_none=True);loss.backward();gn=clip_grad(self.policy.student_encoder.parameters(),cfg.grad_clip)
                if scalar_finite(gn) and scalar_finite(loss):self.opt_cts.step();self.counts['cts']+=1
                else:self.nonfinite+=1
                losses.append(float(loss.detach()))
        return sum(losses)/len(losses)

    def save(self,path=None):
        path=Path(path) if path else self.output/f'model_{self.it:08d}.pt'
        value={'schema':'physx-cts-source-port-v1','upstream_commit':UPSTREAM_COMMIT,'cfg':asdict(self.cfg),
               'policy':self.policy.state_dict(),'critic':self.critic.state_dict(),'world_model':self.wm.state_dict(),
               'optimizers':{k:getattr(self,'opt_'+k).state_dict() for k in ('wm','actor','critic','cts')},
               'iteration':self.it,'counts':self.counts,'lr_scale':self.lr_scale,'ramp_start':self.ramp_start,
               'actor_frozen_until':self.actor_frozen_until,'rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'resume_mode':'fresh_simulator_episodes_and_replay; model/optimizer/normalizer_resume',
               'physics':'isaacsim_physx','consts':{k:v.detach().cpu() for k,v in vars(self.env.consts).items()}}
        partial=path.with_suffix('.partial');torch.save(value,partial);os.replace(partial,path)
        return path

    def load(self,path,continue_run=False):
        ck=torch.load(path,map_location=self.device,weights_only=False)
        if ck['schema']!='physx-cts-source-port-v1' or ck['physics']!='isaacsim_physx':raise ValueError('Old/Newton checkpoint forbidden')
        self.policy.load_state_dict(ck['policy']);self.critic.load_state_dict(ck['critic']);self.wm.load_state_dict(ck['world_model'])
        for k in ('wm','actor','critic','cts'):getattr(self,'opt_'+k).load_state_dict(ck['optimizers'][k])
        if continue_run:
            self.it=ck['iteration'];self.counts=ck['counts'];self.lr_scale=ck['lr_scale'];self.ramp_start=self.it
            self.actor_frozen_until=self.it+self.cfg.warmup_iterations
            # Retired evaluator metadata in immutable checkpoints is ignored.
            torch.set_rng_state(ck['rng'].cpu())
            if ck['cuda_rng']:torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
        self.last_good=Path(path)

    def rollback(self,path=None):
        path=path or self.last_good
        if path is None:return False
        ck=torch.load(path,map_location=self.device,weights_only=False)
        self.policy.load_state_dict(ck['policy']);self.critic.load_state_dict(ck['critic'])
        self.lr_scale*=.5;self.ramp_start=self.it;self.collapse_strikes=0
        # Keep the newly trained WM/replay/archive, discard diverging policy moments.
        self.opt_actor=torch.optim.Adam(self.actor_params,lr=self.cfg.lr,betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=self.cfg.lr)
        self.opt_cts=torch.optim.Adam(self.policy.student_encoder.parameters(),lr=self.cfg.cts_lr)
        return True

    def log(self,stats):
        falls=sum(m['falls'] for m in self.metrics)/max(1,len(self.metrics))
        own=max(1,self.provider_counts[0]+self.provider_counts[1])
        falls_per_min=falls/own*60/self.cfg.dt
        result={'iteration':self.it,'elapsed_seconds':time.monotonic()-self.start_time,'counts':self.counts.copy(),
                'metrics':{k:sum(m[k] for m in self.metrics)/len(self.metrics) for k in ('height','xy_rmse','yaw_rmse','reward','falls')},
                'falls_per_env_min':falls_per_min,'archive_size':self.archive.size,'nonfinite_skipped':self.nonfinite,**stats}
        falls_all=sum(m['falls_all'] for m in self.metrics)/max(1,len(self.metrics)) if self.metrics else falls
        self.metrics=[]
        result['falls_per_vector_step_all_providers']=falls_all
        if self.it>max(self.cfg.warmup_iterations,self.actor_frozen_until) and falls_all>self.cfg.collapse_falls:self.collapse_strikes+=1
        else:self.collapse_strikes=0
        result['rolled_back']=self.rollback() if self.collapse_strikes>=2 else False
        with (self.output/'training.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
        (self.output/'progress.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps(result),flush=True)

    def run(self,iterations=None):
        while self.buf.size<=self.cfg.wm_window+1:self.collect()
        end=self.cfg.iterations if iterations is None else self.it+iterations
        while self.it<end:
            if getattr(self,'stop_requested',False):
                self.save();break
            self.it+=1;stats={'lr':self.learning_rates()}
            self.env.iteration=self.it
            self.collect();stats['wm_loss']=self.fit_world_model()
            if self.it>self.cfg.warmup_iterations:stats.update(self.improve_policy())
            stats['cts_loss']=self.update_cts()
            if self.it==1 or self.it%self.cfg.log_interval==0 or self.it==end:self.log(stats)
            if self.it==1 or self.it%self.cfg.save_interval==0 or self.it==end:
                path=self.save()
                if self.it>self.cfg.warmup_iterations and not self.collapse_strikes:self.last_good=path
            if self.cfg.evaluation_python and self.it%self.cfg.evaluation_interval==0:
                path=self.save();self.evaluate_feedback(path)
                if getattr(self,'stop_requested',False):break
            if self.it==end and iterations is None and self.cfg.adaptive_iterations:
                (self.output/'needs_review.json').write_text(json.dumps({'iteration':self.it,'reason':'Iteration review reached; native and official transfer evidence required before extending budget','B0':False}))
        return self.last_good

    def evaluate_feedback(self,path):
        from .robogauge_feedback import evaluate_checkpoint
        out=self.output/f'robogauge_{self.it:08d}'
        try:
            evaluate_checkpoint(path,out,self.it,self.cfg.evaluation_python,self.cfg.evaluation_profile)
        except Exception:
            self.stop_requested=True
            raise
        self.stop_requested=True
        (self.output/'needs_review.json').write_text(json.dumps({
            'iteration':self.it,'evaluation':str(out/'result.json'),
            'reason':'Official transfer evidence requires native/transfer review before curriculum or rollback decisions',
            'B0':False},indent=2))

    def export(self,path):
        module=copy.deepcopy(StudentExport(self.policy)).eval().cpu()
        traced=torch.jit.trace(module,torch.zeros(1,450),check_trace=True);traced.save(str(path))
        return traced
