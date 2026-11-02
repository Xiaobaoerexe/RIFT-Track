"""Observed-update feedback and explicit curriculum-clock cosine schedules.

No candidate optimizer steps, PPO KL or trial-model copies. The controller owns
all effective rates; diagnostics only submit evidence from ordinary updates.
"""
import copy
import math

def collapse_trend(rows,threshold=5.):
    if len(rows)!=3:return False
    if not all(math.isfinite(r[k]) for r in rows for k in ('falls','xy_rmse','reward')):return False
    return (all(r['falls']>threshold for r in rows)
            and all(a['falls']<b['falls'] and a['xy_rmse']<b['xy_rmse'] and a['reward']>b['reward']
                    for a,b in zip(rows,rows[1:])))


class LearningRateController:
    version = 3

    def __init__(self, cfg):
        self.cfg = cfg
        self.current = {'wm': cfg.wm_lr, 'actor': cfg.lr, 'critic': cfg.lr}
        self.bounds = {'wm': cfg.wm_lr_bounds, 'actor': cfg.actor_lr_bounds,
                       'critic': cfg.critic_lr_bounds}
        self.streaks = {k: {'bad': 0, 'good': 0, 'reductions': 0, 'blocked': False}
                        for k in self.current}
        self.last_window = cfg.lr_feedback_start
        self.history = []
        self.teacher_stable = False
        self.teacher_evaluations = []
        self.transfer_evaluations = []
        self.student_start = None
        self.student_end = None
        self.last_stability_check = -1
        self.cosine_base = None
        self.cosine_safety = {k: 1. for k in self.current}
        self.ramp_start = cfg.policy_start
        self.rollback_recovery = {}
        self.actor_reference = {}
        self.native_regression = False

    @staticmethod
    def cosine(initial, final, start, end, iteration):
        if end <= start:
            raise ValueError('Cosine must have a positive duration')
        progress = min(1., max(0., (iteration - start) / (end - start)))
        return final + (initial - final) * .5 * (1. + math.cos(math.pi * progress))

    def _recover_initial_ramp(self, iteration):
        # Preserve the latest server rollback contract: original ramp clock and
        # original endpoint, no new freeze and no optimizer-moment reset.
        for name, record in list(self.rollback_recovery.items()):
            p = min(1., max(0., (iteration-record['start'])/(record['end']-record['start'])))
            self.current[name] = record['initial'] + p*(record['target']-record['initial'])
            if p >= 1.:
                del self.rollback_recovery[name]

    def _stable(self, records, iteration):
        if len(records) != 3:
            return False
        if len({r['iteration'] for r in records}) != 3:
            return False
        if iteration-records[-1]['iteration'] > 2*self.cfg.evaluation_interval:
            return False
        if any(r['cohort_id'] != records[0]['cohort_id'] for r in records):
            return False
        if any(r['success_rate'] < .8 for r in records):
            return False
        values = [r['return'] for r in records]
        return (max(values)-min(values))/max(1., abs(sum(values)/3)) <= .05

    def observe(self, record):
        kind = record.get('kind')
        if kind not in ('teacher_fixed_eval', 'teacher_transfer_fixed_eval'):
            return
        name = 'teacher_evaluations' if kind == 'teacher_fixed_eval' else 'transfer_evaluations'
        required = ('iteration', 'cohort_id', 'success_rate', 'return')
        if not record.get('valid') or any(k not in record for k in required):
            setattr(self, name, [])
            self.teacher_stable = False
            return
        if not all(math.isfinite(float(record[k])) for k in ('iteration','success_rate','return')):
            setattr(self, name, [])
            self.teacher_stable = False
            return
        rows = getattr(self, name)
        if rows and record['iteration'] <= rows[-1]['iteration']:
            return  # One report is never counted as three evaluations.
        if rows and record['cohort_id'] != rows[-1]['cohort_id']:
            rows = []
        if kind == 'teacher_fixed_eval':
            self.native_regression = bool(rows and record['cohort_id']==rows[-1]['cohort_id']
                and record['return'] < rows[-1]['return']-.05*max(1.,abs(rows[-1]['return']))
                and record['success_rate'] < rows[-1]['success_rate'])
        setattr(self, name, (rows+[copy.deepcopy(record)])[-3:])
        it = int(record['iteration'])
        self.teacher_stable = self._stable(self.teacher_evaluations,it) and self._stable(self.transfer_evaluations,it)

    def check_student(self, iteration):
        c = self.cfg
        if iteration < c.cts_stability_start or iteration % c.cts_stability_interval:
            return
        if self.last_stability_check == iteration:
            return
        self.last_stability_check = iteration
        self.teacher_stable = (self._stable(self.teacher_evaluations, iteration)
                               and self._stable(self.transfer_evaluations, iteration))
        if self.student_start is None and self.teacher_stable and iteration < c.cts_late_end:
            self.student_start = iteration
            self.student_end = c.cts_early_end if iteration < c.cts_early_end else c.cts_late_end
            self.history.append({'iteration':iteration,'component':'cts','action':'start_cosine',
                                 'end':self.student_end})
        self.history = self.history[-1000:]

    def rates(self, iteration):
        c = self.cfg
        self._recover_initial_ramp(iteration)
        if iteration >= c.lr_cosine_start and self.cosine_base is None:
            self.cosine_base = {k:max(c.lr_final,v) for k,v in self.current.items()}
        if self.cosine_base is None:
            rates = dict(self.current)
        else:
            rates = {k:c.lr_final+self.cosine_safety[k]*(self.cosine(
                v,c.lr_final,c.lr_cosine_start,c.lr_cosine_end,iteration)-c.lr_final)
                     for k,v in self.cosine_base.items()}
        ramp = min(1.,max(0.,(iteration-self.ramp_start+1)/c.actor_ramp_iterations))
        rates['actor'] *= ramp
        rates['teacher'] = rates['actor']
        rates['actor_ramp'] = ramp
        rates['cts'] = 0. if iteration < c.policy_start else c.cts_lr
        if self.student_start is not None:
            rates['cts'] = self.cosine(c.cts_lr,c.cts_final_lr,self.student_start,self.student_end,iteration)
        return rates

    def observe_due(self, iteration):
        c = self.cfg
        return c.lr_feedback_start < iteration < c.lr_cosine_start and (
            iteration-c.lr_feedback_start) % c.lr_observe_interval == 0

    def window_due(self, iteration):
        c = self.cfg
        return c.lr_feedback_start < iteration < c.lr_cosine_start and (
            iteration-c.lr_feedback_start) % c.lr_feedback_interval == 0

    def feedback(self, iteration, evidence):
        if not self.window_due(iteration) or iteration <= self.last_window:
            return {}
        complete = iteration-self.last_window == self.cfg.lr_feedback_interval
        self.last_window = iteration
        if not complete:
            self.clear_streaks()
        result = {}
        actor_can_recover = (all(evidence.get(k,{}).get('status')=='good' for k in ('wm','critic'))
                             and evidence.get('tracking',{}).get('status')=='good'
                             and not self.native_regression)
        for name in ('wm','critic','actor'):
            row = evidence.get(name, {})
            status = row.get('status','unknown') if complete else 'unknown'
            s = self.streaks[name]
            old = self.current[name]
            action = 'hold'
            if name=='actor' and status=='good' and not actor_can_recover:
                status='unknown'
            if status=='bad':
                s['bad']+=1; s['good']=0
                if s['bad']>=2 and not s['blocked']:
                    if s['reductions']>=2:
                        s['blocked']=True; action='ineffective_reductions_stop'
                    else:
                        # A protection rollback may already lie below the ordinary
                        # feedback floor: a "decrease" must never increase it.
                        floor=self.bounds[name][0]
                        self.current[name]=min(old,max(floor,old/self.cfg.lr_adjust_factor))
                        action='decrease' if self.current[name]<old else 'at_feedback_floor'
                        if self.current[name]<old:s['reductions']+=1
                    s['bad']=s['good']=0
            elif status=='good':
                s['good']+=1; s['bad']=0
                if s['good']>=3:
                    self.current[name]=min(self.bounds[name][1],old*self.cfg.lr_adjust_factor)
                    action='recover' if self.current[name]>old else 'at_initial_ceiling'
                    s.update(bad=0,good=0,reductions=0,blocked=False)
            else:
                s['bad']=s['good']=0
            result[name]={'previous':old,'selected':self.current[name],'action':action,
                          'evidence':copy.deepcopy(row),'streak':dict(s)}
        record={'iteration':iteration,'complete_window':complete,'components':result,
                'tracking':evidence.get('tracking',{}),'actor_recovery_allowed':actor_can_recover}
        self.history=(self.history+[record])[-1000:]
        return record

    def clear_streaks(self):
        for s in self.streaks.values():
            s['bad']=s['good']=0

    def rollback(self, iteration):
        self.rates(iteration)
        end = self.ramp_start+self.cfg.actor_ramp_iterations-1
        for name in ('actor','critic'):
            if iteration >= self.cfg.lr_cosine_start:
                self.cosine_safety[name] *= .5
            else:
                target=self.rollback_recovery.get(name,{}).get('target',self.current[name])
                self.current[name]*=.5
                if iteration<end:
                    self.rollback_recovery[name]={'start':iteration,'end':end,
                        'initial':self.current[name],'target':target}
                else:
                    self.rollback_recovery.pop(name,None)
        self.clear_streaks()
        self.teacher_evaluations=[];self.transfer_evaluations=[];self.teacher_stable=False
        # Student's already-latched cosine clock is intentionally preserved.

    def state_dict(self):
        return {'version':self.version, **{k:copy.deepcopy(v) for k,v in vars(self).items() if k!='cfg'}}

    def load_state_dict(self, state):
        if state.get('version') != self.version:
            raise ValueError('Legacy candidate LR state cannot resume under observed-update scheduling')
        for key,value in state.items():
            if key!='version':
                if key not in vars(self):raise ValueError('Unknown schedule field: '+key)
                setattr(self,key,copy.deepcopy(value))

class ProviderSchedule:
    """Supported official collection fades with competence and ends before phase3."""
    supported=(0,1,2,3,4)
    stop_iteration=8000
    def __init__(self,cfg):
        self.initial=cfg.driver_frac;self.duration=cfg.provider_fade_iterations
        self.started={};self.confirmations={};self.evidence=[]
    def observe(self,task,iteration,own_success,model_valid,stable):
        if task not in self.supported:return
        ready=own_success>=.8 and bool(model_valid) and bool(stable)
        self.confirmations[task]=self.confirmations.get(task,0)+1 if ready else 0
        if self.confirmations[task]>=2 and task not in self.started:self.started[task]=iteration
        self.evidence.append({'task':task,'iteration':iteration,'own_success':own_success,'model_valid':bool(model_valid),'stable':bool(stable),'fade_start':self.started.get(task)})
        self.evidence=self.evidence[-100:]
    def fraction(self,task,iteration):
        if iteration>=self.stop_iteration or task not in self.supported:return 0.
        start=self.started.get(task)
        return self.initial if start is None else self.initial*max(0.,1.-(iteration-start)/self.duration)
    def state_dict(self):return {'started':self.started,'confirmations':self.confirmations,'evidence':self.evidence}
    def load_state_dict(self,state):
        self.started={int(k):v for k,v in state['started'].items()}
        self.confirmations={int(k):v for k,v in state['confirmations'].items()};self.evidence=state['evidence']
