"""Bounded, no-gradient diagnostics of ordinary updates on held-out episodes.

The simulator, optimizers and training RNG are never advanced by this module.
Small immutable panels live on CPU; geometry/features are rebuilt when used.
"""
from contextlib import contextmanager
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from . import state as st
from .terrain import geometry_crop
from .sequence_execution import depth_sequence_features,batched_apply
from .imagine import time_limit_lambda_targets,flat_ids

def finite_mean(value):
    value=float(np.mean(value))
    return value if math.isfinite(value) else None

def move(value,device):
    if torch.is_tensor(value):return value.detach().to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    return value

@contextmanager
def diagnostic_context(learner,fp32=True):
    modules=[learner.wm,learner.policy,learner.critic]
    flags=[]
    for root in modules:
        for m in root.modules():
            if fp32 and hasattr(m,'mixed_precision'):
                flags.append((m,m.mixed_precision));m.mixed_precision=False
    devices=[torch.device(learner.device).index or 0] if str(learner.device).startswith('cuda') else []
    try:
        with torch.no_grad(),torch.random.fork_rng(devices=devices):
            yield
    finally:
        for m,value in flags:m.mixed_precision=value

def episode_keys(env,episode):
    return np.stack((env.detach().cpu().numpy().reshape(-1),
                     episode.detach().cpu().numpy().reshape(-1)),axis=1)

def paired(before,after,keys,groups,min_episodes=8):
    """Task-balanced paired episode-cluster CI; time steps are not iid trials."""
    a=np.asarray(before,dtype=np.float64).reshape(-1)
    b=np.asarray(after,dtype=np.float64).reshape(-1)
    groups=np.asarray(groups).reshape(-1);keys=np.asarray(keys).reshape(-1,2)
    if len(a)!=len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'valid':False,'reason':'nonfinite_or_unpaired'}
    cells={}
    for av,bv,key,group in zip(a,b,keys,groups):
        cell=cells.setdefault(int(group),{})
        cell.setdefault(tuple(map(int,key)),[]).append((float(av),float(bv)))
    episodes={tuple(map(int,k)) for k in keys}
    if len(episodes)<min_episodes or not cells:
        return {'valid':False,'reason':'insufficient_episodes','episodes':len(episodes)}
    means=[];variances=[];detail={}
    for group,records in cells.items():
        rows=np.asarray([np.mean(v,axis=0) for v in records.values()])
        if len(rows)<2:return {'valid':False,'reason':'insufficient_group_episodes','group':group}
        denominator=max(float(np.mean(rows[:,0])),1e-8)
        changes=(rows[:,1]-rows[:,0])/denominator
        mean=float(changes.mean());var=float(changes.var(ddof=1)/len(changes))
        means.append(mean);variances.append(var)
        detail[str(group)]={'episodes':len(rows),'before':float(rows[:,0].mean()),
                            'after':float(rows[:,1].mean()),'relative_change':mean}
    mean=float(np.mean(means))
    # Conservative aggregation: teacher/student groups can share episodes, so
    # do not assume independence between their estimated means.
    se=sum(math.sqrt(v) for v in variances)/len(means)
    df=min(len(v) for v in cells.values())-1
    critical=(12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,
              2.201,2.179,2.160,2.145,2.131,2.120,2.110,2.101,2.093,2.086,
              2.080,2.074,2.069,2.064,2.060,2.056,2.052,2.048,2.045,2.042)
    multiplier=critical[min(df,30)-1]  # Conservative >=30df bound, approximate CI.
    return {'valid':True,'mean':mean,'low':mean-multiplier*se,'high':mean+multiplier*se,
            'episodes':len(episodes),'groups':detail}

def classify(summary,threshold):
    if not summary.get('valid'):return 'unknown'
    if summary['low']>threshold:return 'bad'
    if summary['high']<-threshold:return 'good'
    return 'neutral'

@torch.no_grad()
def sequence_latents(learner,sequence):
    policy=learner.policy;student=policy.student_encoder;c=learner.cfg
    b,total=sequence['obs'].shape[:2]
    single=policy.single_norm(st.single_from_history(sequence['obs'].float().reshape(b*total,-1))).reshape(b,total,-1)
    hist=policy.history_norm(sequence['obs'].float().reshape(b*total,-1))
    proprio=batched_apply(lambda v:student.encode_proprio(v)[0],hist).reshape(b,total,-1)
    depth=depth_sequence_features(student,learner.depth,sequence,0,total)
    hidden=torch.zeros((b,256),device=learner.device);zs=[]
    for j in range(total):
        z,hidden=student.fuse_features(proprio[:,j],single[:,j],depth[:,j],hidden,sequence['reset'][:,j])
        if j>=c.cts_burnin:zs.append(z)
    zs=torch.stack(zs,1);steps=total-c.cts_burnin
    fields={k:sequence[k][:,c.cts_burnin:].reshape(b*steps,*sequence[k].shape[2:])
            for k in ('s','a_prev','command','xi','relief','pose','task_ids')}
    labels=[]
    for lo in range(0,b*steps,256):
        item={k:v[lo:lo+256] for k,v in fields.items()}
        priv=st.privileged(item['s'],item['a_prev'],item['command'],item['xi'],item['relief'])
        labels.append(policy.teacher_latent(priv,geometry_crop(item['s'],item['pose'],learner.env.field,kind='teacher'),flat_ids(item['task_ids'])))
    teacher=torch.cat(labels).reshape(b,steps,-1)
    return teacher,zs,single[:,-1]

@torch.no_grad()
def critic_values(critic,x):
    flat,physical,pose,branches,tasks=critic.decode(x)
    output=torch.empty(len(flat),device=x.device)
    for task in tasks.unique().tolist():
        ids=(tasks==task).nonzero().flatten()
        for part in ids.split(256):
            height=geometry_crop(None,pose[part],critic.field,kind='teacher')
            z=critic.geometry(height)
            v=critic.tasks[task](critic.norm(physical[part]),z).gather(1,branches[part,None]).squeeze(1)
            output[part]=v*critic.pop_scale[task,branches[part]]+critic.pop_mean[task,branches[part]]
    return output.reshape(x.shape[:-1])

class UpdateFeedback:
    STATE_FIELDS=('start','fixed','recent','sequence','wm_before','scales','latents_before',
                  'actor_rows','critic_rows','observed','errors','last_report')
    def __init__(self,learner):
        self.l=learner;self.start=None;self.fixed=None;self.recent=None;self.sequence=None
        self.wm_before={};self.scales=None;self.latents_before=None
        self.actor_rows=[];self.critic_rows=[];self.observed=[];self.errors=[]
        self.actor_pending=None;self.critic_pending=None
        self.last_report={}

    def reset(self):
        self.__init__(self.l)

    def due(self):
        return self.start is not None and self.l.schedule.observe_due(self.l.it)

    def begin(self):
        l=self.l;c=l.cfg
        self.start=l.it;self.actor_rows=[];self.critic_rows=[];self.observed=[];self.errors=[]
        with diagnostic_context(l):
            try:
                if self.fixed is None:
                    self.fixed=move(l.buf.windows(c.lr_probe_windows,c.wm_window,validation=True),'cpu')
                self.recent=move(l.buf.windows(c.lr_probe_windows,c.wm_window,validation=True),'cpu')
                self.scales=(l.wm.out_norm.var.sqrt().clamp_min(1e-4).cpu().clone(),
                             l.wm.telemetry_norm.var.sqrt().clamp_min(1e-3).cpu().clone())
                self.wm_before={name:self._wm(panel) for name,panel in (('fixed',self.fixed),('recent',self.recent))}
                self.sequence=move(l.buf.sequences(c.lr_probe_sequences,c.cts_sequence_steps,c.cts_burnin,source=1,validation=True),'cpu')
                t,s,_=sequence_latents(l,move(self.sequence,l.device))
                self.latents_before=(t.cpu(),s.cpu())
            except ValueError as error:
                self.errors.append(str(error));self.sequence=None;self.latents_before=None
        # Missing samples cause HOLD, not a replacement with training examples.

    def _wm(self,panel):
        l=self.l
        rows=l.wm.validation_errors(move(panel,l.device),l.env.field,l.env.consts.default_q,move(self.scales,l.device))
        return {k:v.cpu().numpy().tolist() for k,v in rows.items()}

    def before_actor(self):
        self.actor_pending=None
        if not self.due() or self.sequence is None:return
        l=self.l
        with diagnostic_context(l):
            seq=move(self.sequence,l.device);t,s,single=sequence_latents(l,seq)
            priv=st.privileged(seq['s'][:,-1],seq['a_prev'][:,-1],seq['command'][:,-1],seq['xi'][:,-1],seq['relief'][:,-1])
            height=geometry_crop(seq['s'][:,-1],seq['pose'][:,-1],l.env.field,kind='teacher')
            tasks=flat_ids(seq['task_ids'][:,-1])
            actions=torch.stack((l.policy.actor(t[:,-1],single),l.policy.actor(s[:,-1],single)),1)
            self.actor_pending=(priv,height,tasks,s[:,-1],single,actions)

    def after_actor(self):
        if self.actor_pending is None:return
        l=self.l
        with diagnostic_context(l):
            priv,height,tasks,zs,single,old=self.actor_pending
            zt=l.policy.teacher_latent(priv,height,tasks)
            new=torch.stack((l.policy.actor(zt,single),l.policy.actor(zs,single)),1)
            change=((new-old)*l.env.consts.action_scale).square().mean(-1).sqrt()
            self.actor_rows.append({'task':tasks.cpu().tolist(),'change':change.cpu().tolist()})
        self.actor_pending=None

    def before_critic(self):
        self.critic_pending=None
        if not self.due():return
        l=self.l;c=l.cfg;n=c.lr_probe_imaginations;split=(3*n//4,n//4)
        try:
            # Diagnostic-only eager imagination avoids compiling a second BPTT
            # cohort or retaining training graphs. It uses the same dynamics.
            from .imagine import Imagination
            cfg=copy.copy(c);cfg.compile=False
            with diagnostic_context(l,fp32=False):
                imag=Imagination(cfg,l.env.consts,l.env.field,l.env.command_sampler,l.depth)
                a=l.buf.seeds(split[0],burnin=0,validation=True)
                b=l.buf.seeds(split[1],burnin=c.cts_burnin,validation=True)
                seed={k:torch.cat((v,b[k])) for k,v in a.items()};seed['student_prefix']=b['prefix']
                _,data=imag.rollout(l.wm,l.policy,l.critic,seed,split,l.it)
            with diagnostic_context(l):
                x=data['critic_in'];values=critic_values(l.critic,x)
                y=time_limit_lambda_targets(data['rewards'],data['soft_done'],data['timeout'],values,c.gamma,c.lam)
                flat=x[:,:-1].reshape(-1,l.critic.feature_dim);target=y.reshape(-1)
                _,_,_,branch,task=l.critic.decode(flat)
                scale=l.critic.pop_scale[task,branch].clone()
                before=F.smooth_l1_loss(critic_values(l.critic,flat)/scale,target/scale,reduction='none')
                keys=episode_keys(seed['trajectory_ids'],seed['episode'])
                groups=(flat_ids(seed['task_ids']).cpu().numpy()[:,None]*2+np.r_[np.zeros(split[0],int),np.ones(split[1],int)][:,None]).reshape(-1)
                valid=data['critic_valid'].bool().reshape(n,-1)
                self.critic_pending=(flat,target,scale,before.reshape(n,-1),valid,keys,groups)
        except ValueError as error:
            self.errors.append('critic: '+str(error))

    def after_critic(self):
        if self.critic_pending is None:return
        l=self.l
        with diagnostic_context(l):
            x,y,scale,before,valid,keys,groups=self.critic_pending
            after=F.smooth_l1_loss(critic_values(l.critic,x)/scale,y/scale,reduction='none').reshape_as(before)
            den=valid.sum(-1).clamp_min(1);eligible=valid.any(-1)
            self.critic_rows.append({'before':((before*valid).sum(-1)/den)[eligible].cpu().tolist(),
                'after':((after*valid).sum(-1)/den)[eligible].cpu().tolist(),
                'keys':keys[eligible.cpu().numpy()].tolist(),'groups':groups[eligible.cpu().numpy()].tolist()})
        self.critic_pending=None

    def _actor_summary(self):
        l=self.l;cells={}
        if len(self.actor_rows)!=l.cfg.lr_feedback_interval//l.cfg.lr_observe_interval:
            return {'status':'unknown','reason':'incomplete_actual_updates'}
        for row in self.actor_rows:
            for task,values in zip(row['task'],row['change']):
                for branch,v in enumerate(values):
                    cells.setdefault(str(task*2+branch),[]).append(v)
        ratios={};new=[]
        for key,values in cells.items():
            if not np.isfinite(values).all():return {'status':'unknown','reason':'nonfinite'}
            value=float(np.quantile(values,.95))
            if key not in l.schedule.actor_reference:
                if l.schedule.current['actor']==l.cfg.lr and not l.collapse_strikes:
                    l.schedule.actor_reference[key]=max(value,1e-8);new.append(key)
                continue
            ratios[key]=value/l.schedule.actor_reference[key]
        if not ratios or new:return {'status':'unknown','reason':'reference_initialization','new_groups':new}
        high=max(ratios.values())
        return {'status':'bad' if high>l.cfg.lr_actor_high else 'good' if high<=l.cfg.lr_actor_low else 'neutral',
                'max_ratio':high,'ratios':ratios}

    def finish(self):
        l=self.l;c=l.cfg
        if l.it<c.lr_feedback_start or l.it>=c.lr_cosine_start:return None
        if self.start is None:
            self.begin();return None
        if not l.schedule.observe_due(l.it):return None
        self.observed.append(l.it)
        if not l.schedule.window_due(l.it):return None
        evidence={}
        with diagnostic_context(l):
            wm={}
            for name,panel in (('fixed',self.fixed),('recent',self.recent)):
                if panel is None or name not in self.wm_before:
                    wm[name]={'valid':False};continue
                after=self._wm(panel)
                summary=paired(self.wm_before[name]['total'],after['total'],
                    episode_keys(panel['trajectory_ids'],panel['episode_ids']),panel['task_ids'][:,0].numpy(),c.lr_min_episodes)
                summary['physical_metrics']={k:finite_mean(v) for k,v in after.items()}
                wm[name]=summary
            states=[classify(v,c.lr_wm_threshold) for v in wm.values()]
            evidence['wm']={'status':states[0] if len(states)==2 and states[0]==states[1] else 'unknown','panels':wm}
            if len(self.critic_rows)==c.lr_feedback_interval//c.lr_observe_interval:
                joined={k:sum((r[k] for r in self.critic_rows),[]) for k in ('before','after','keys','groups')}
                summary=paired(**joined,min_episodes=c.lr_min_episodes)
                evidence['critic']={'status':classify(summary,c.lr_critic_threshold),'measurement':summary}
            else:evidence['critic']={'status':'unknown','reason':'no_independent_fit_samples'}
            evidence['actor']=self._actor_summary()
            evidence['tracking']={'status':'unknown'}
            if self.sequence is not None and self.latents_before is not None:
                seq=move(self.sequence,l.device)
                t,s,single=sequence_latents(l,seq);ot,os=move(self.latents_before,l.device)
                e00=(os-ot).square().mean((1,2));e01=(os-t).square().mean((1,2));e11=(s-t).square().mean((1,2))
                oldgap=(l.policy.actor(os[:,-1],single)-l.policy.actor(ot[:,-1],single)).square().mean(-1)
                newgap=(l.policy.actor(s[:,-1],single)-l.policy.actor(t[:,-1],single)).square().mean(-1)
                keys=episode_keys(seq['trajectory_ids'],seq['episode'][:,-1]);groups=seq['task_ids'][:,-1].cpu().numpy()
                latent=paired(e00.cpu(),e11.cpu(),keys,groups,c.lr_min_episodes)
                gap=paired(oldgap.cpu(),newgap.cpu(),keys,groups,c.lr_min_episodes)
                valid=latent.get('valid') and gap.get('valid')
                bad=valid and latent['low']>c.lr_tracking_threshold and gap['low']>0.
                good=valid and latent['high']<=c.lr_tracking_threshold and gap['high']<=c.lr_tracking_threshold
                evidence['tracking']={'status':'bad' if bad else 'good' if good else 'unknown',
                    'latent':latent,'action_gap':gap,'target_drift':finite_mean((e01-e00).cpu().numpy()),
                    'student_catchup':finite_mean((e01-e11).cpu().numpy())}
        expected=list(range(self.start+c.lr_observe_interval,l.it+1,c.lr_observe_interval))
        if self.observed!=expected:
            for name in ('wm','critic','actor'):evidence[name]['status']='unknown'
        result=l.schedule.feedback(l.it,evidence)
        self.last_report={'iteration':l.it,'diagnostic_points':list(self.observed),'errors':list(self.errors),'decision':result}
        result=self.last_report
        self.begin()
        return result

    def state_dict(self):
        return {k:copy.deepcopy(getattr(self,k)) for k in self.STATE_FIELDS}

    def load_state_dict(self,state):
        for key,value in state.items():
            if key not in self.STATE_FIELDS:raise ValueError('Unknown feedback field: '+key)
            setattr(self,key,move(value,'cpu'))
