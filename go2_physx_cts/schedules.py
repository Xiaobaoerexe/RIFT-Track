"""Validation-driven bounded learning-rate trials for model-based BPTT.

No PPO likelihood/KL rule is applied to this deterministic return optimizer.
Candidate updates start from identical model, optimizer, data and RNG state.
The selected rate is used by subsequent ordinary updates; probes are rolled back.
"""
import copy
import math
import torch

def collapse_trend(rows,threshold=5.):
    if len(rows)!=3:return False
    if not all(math.isfinite(r[k]) for r in rows for k in ('falls','xy_rmse','reward')):return False
    return (all(r['falls']>threshold for r in rows)
            and all(a['falls']<b['falls'] and a['xy_rmse']<b['xy_rmse'] and a['reward']>b['reward']
                    for a,b in zip(rows,rows[1:])))

class LearningRateController:
    def __init__(self,cfg):
        self.cfg=cfg
        self.current={'wm':cfg.wm_lr,'actor':cfg.lr,'critic':cfg.lr,'cts':cfg.cts_lr}
        self.bounds={'wm':cfg.wm_lr_bounds,'actor':cfg.actor_lr_bounds,
                     'critic':cfg.critic_lr_bounds,'cts':cfg.cts_lr_bounds}
        self.last_trial={name:-1 for name in self.current}
        self.history=[]
        self.teacher_stable=False
        self.teacher_evaluations=[]
        self.transfer_evaluations=[]
        self.ramp_start=cfg.warmup_iterations+cfg.critic_warmup_iterations
        self.rollback_recovery={}

    def rates(self,iteration,teacher_stable=False):
        self.teacher_stable=self.teacher_stable or teacher_stable
        ramp=min(1.,max(0.,(iteration-self.ramp_start)/self.cfg.actor_ramp_iterations))
        for name,record in list(self.rollback_recovery.items()):
            fraction=min(1.,max(0.,(iteration-record['start'])/(record['end']-record['start'])))
            self.current[name]=record['initial']+(record['target']-record['initial'])*fraction
            if fraction>=1.:del self.rollback_recovery[name]
        rates=dict(self.current)
        rates['actor']*=ramp
        rates['teacher']=rates['actor']*self.cfg.teacher_lr/self.cfg.lr
        rates['actor_ramp']=ramp
        return rates

    def due(self,name,iteration,teacher_stable=False):
        if iteration<=self.cfg.lr_fixed_until:return False
        if name=='cts' and not(self.teacher_stable or teacher_stable):return False
        return iteration%self.cfg.lr_trial_interval==0 and self.last_trial[name]!=iteration

    def _candidates(self,name):
        rate=self.current[name];factor=self.cfg.lr_trial_factor
        lo,hi=self.bounds[name]
        return sorted(set(max(lo,min(hi,rate*f)) for f in (1/factor,1.,factor)))

    @staticmethod
    def _set_rate(optimizer,rate,original_rates,base_rate):
        for group,old in zip(optimizer.param_groups,original_rates):
            group['lr']=rate*(old/base_rate if base_rate else 1.)

    def supervised_trial(self,name,module,optimizer,train_loss_fn,validation_fn,iteration,teacher_stable=False):
        if name not in ('wm','critic','cts'):raise ValueError('Supervised trials require an actual held-out supervised objective')
        if not self.due(name,iteration,teacher_stable):return None
        base=copy.deepcopy(module.state_dict());moments=copy.deepcopy(optimizer.state_dict())
        rates=[g['lr'] for g in optimizer.param_groups]
        base_rate=self.current[name];records=[]
        flags=module.training
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for candidate in self._candidates(name):
                module.load_state_dict(base);optimizer.load_state_dict(copy.deepcopy(moments))
                torch.set_rng_state(cpu_rng)
                if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
                self._set_rate(optimizer,candidate,rates,base_rate)
                module.train();optimizer.zero_grad(set_to_none=True)
                loss=train_loss_fn(module)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],self.cfg.grad_clip)
                if torch.isfinite(loss).all() and torch.isfinite(norm).all():
                    optimizer.step();module.eval()
                    with torch.no_grad():score=float(validation_fn(module))
                else:score=float('inf')
                records.append({'lr':candidate,'validation_error':score})
        finally:
            module.load_state_dict(base);optimizer.load_state_dict(copy.deepcopy(moments));optimizer.zero_grad(set_to_none=True);module.train(flags)
            torch.set_rng_state(cpu_rng)
            if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
        finite=[r for r in records if math.isfinite(r['validation_error'])]
        winner=min(finite,key=lambda r:(r['validation_error'],abs(math.log(r['lr']/base_rate)))) if finite else None
        if winner:self.current[name]=winner['lr']
        return self._record(name,iteration,base_rate,records)

    def actor_trial(self,policy,optimizer,train_loss_fn,probe_fn,iteration):
        if not self.due('actor',iteration):return None
        if probe_fn is None:raise ValueError('Actor LR trials require a real fixed-PhysX probe')
        base=copy.deepcopy(policy.state_dict());moments=copy.deepcopy(optimizer.state_dict())
        rates=[g['lr'] for g in optimizer.param_groups];base_rate=self.current['actor']
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        records=[];training=policy.training
        try:
            for candidate in self._candidates('actor'):
                policy.load_state_dict(base);optimizer.load_state_dict(copy.deepcopy(moments))
                torch.set_rng_state(cpu_rng)
                if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
                self._set_rate(optimizer,candidate,rates,base_rate)
                optimizer.zero_grad(set_to_none=True)
                loss=train_loss_fn(policy)
                loss.backward()
                params=[p for g in optimizer.param_groups for p in g['params']]
                norm=torch.nn.utils.clip_grad_norm_(params,self.cfg.grad_clip)
                for p in params:
                    if p.grad is not None:p.grad.clamp_(-self.cfg.grad_clip_elem,self.cfg.grad_clip_elem)
                if not torch.isfinite(loss).all() or not torch.isfinite(norm).all():
                    records.append({'lr':candidate,'accepted':False,'reason':'nonfinite_gradient'});continue
                optimizer.step();policy.eval()
                result=probe_fn(policy)
                required=('heldout_return','real_return','falls_per_env_min','tracking_rmse')
                if any(k not in result or not math.isfinite(float(result[k])) for k in required):
                    raise ValueError('Incomplete/nonfinite fixed-task real Actor validation')
                records.append({'lr':candidate,**{k:float(result[k]) for k in required},'accepted':True})
        finally:
            policy.load_state_dict(base);optimizer.load_state_dict(copy.deepcopy(moments));optimizer.zero_grad(set_to_none=True);policy.train(training)
            torch.set_rng_state(cpu_rng)
            if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
        baseline=min((r for r in records if r['accepted']),key=lambda r:abs(r['lr']-base_rate),default=None)
        eligible=[]
        if baseline:
            for r in records:
                if not r['accepted']:continue
                # Candidate must not trade real falls/tracking for hallucinated
                # gain. Small cohort tolerances are logged, never called proof.
                safe=r['falls_per_env_min']<=baseline['falls_per_env_min']+.05 and r['tracking_rmse']<=baseline['tracking_rmse']*1.05+.01
                supported=r['heldout_return']>=baseline['heldout_return']-.01*max(1.,abs(baseline['heldout_return']))
                r['accepted']=safe and supported
                if r['accepted']:eligible.append(r)
        if eligible:
            winner=max(eligible,key=lambda r:(r['real_return'],r['heldout_return'],-abs(math.log(r['lr']/base_rate))))
            self.current['actor']=winner['lr']
        return self._record('actor',iteration,base_rate,records)

    def _record(self,name,iteration,previous,candidates):
        self.last_trial[name]=iteration
        candidates=copy.deepcopy(candidates)
        for candidate in candidates:
            for key,value in list(candidate.items()):
                if isinstance(value,float) and not math.isfinite(value):
                    candidate[key]=None
                    candidate['reason']='nonfinite_candidate_rejected'
        record={'component':name,'iteration':iteration,'previous':previous,'selected':self.current[name],'factor':self.cfg.lr_trial_factor,'candidates':candidates}
        self.history.append(record)
        self.history=self.history[-1000:]
        return record

    def observe(self,record):
        """Only matching fixed-task evaluations may release student LR adaptation."""
        if record.get('kind') not in ('teacher_fixed_eval','teacher_transfer_fixed_eval') or not record.get('valid'):return
        name='teacher_evaluations' if record['kind']=='teacher_fixed_eval' else 'transfer_evaluations'
        rows=getattr(self,name)+[dict(record)];setattr(self,name,rows[-3:])
        def stable(records):
            if len(records)<3:return False
            first=records[0]
            if any(r.get('cohort_id')!=first.get('cohort_id') for r in records):return False
            if any(float(r['success_rate'])<.8 for r in records):return False
            values=[float(r['return']) for r in records]
            return (max(values)-min(values))/max(1.,abs(sum(values)/len(values)))<.05
        self.teacher_stable=stable(self.teacher_evaluations) and stable(self.transfer_evaluations)

    def rollback(self,iteration):
        self.rates(iteration)
        end=self.ramp_start+self.cfg.actor_ramp_iterations
        for name in ('actor','critic'):
            target=self.rollback_recovery.get(name,{}).get('target',self.current[name])
            self.current[name]=max(self.bounds[name][0],self.current[name]*.5)
            if iteration<end:
                self.rollback_recovery[name]={'start':iteration,'end':end,'initial':self.current[name],'target':target}
            else:self.rollback_recovery.pop(name,None)

    def state_dict(self):
        return {'current':self.current,'last_trial':self.last_trial,'history':self.history,
                'teacher_stable':self.teacher_stable,'teacher_evaluations':self.teacher_evaluations,
                'transfer_evaluations':self.transfer_evaluations,'ramp_start':self.ramp_start,'rollback_recovery':self.rollback_recovery}

    def load_state_dict(self,state):
        for key,value in state.items():setattr(self,key,value)

