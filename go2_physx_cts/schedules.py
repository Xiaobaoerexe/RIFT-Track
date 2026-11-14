"""Window-level feedback controller. No candidate updates or weight rollback."""
import copy, math, statistics
class LearningRateController:
    def __init__(self,cfg):
        self.cfg=cfg
        self.current={'wm':cfg.wm_lr,'actor':cfg.lr,'critic':cfg.lr,'cts':cfg.cts_lr}
        self.bounds={'wm':cfg.wm_lr_bounds,'actor':cfg.actor_lr_bounds,'critic':cfg.critic_lr_bounds,'cts':cfg.cts_lr_bounds}
        self.ramp_start=cfg.warmup_iterations+cfg.critic_warmup_iterations
        self.rows=[];self.previous=None;self.streak={k:0 for k in ('actor_bad','actor_good','wm_bad','wm_good','critic_bad','critic_good','cts_bad','cts_good')}
        self.last_change={k:-10000 for k in self.current};self.history=[];self.teacher_stable=False
    def rates(self,it):
        out=dict(self.current);ramp=min(1.,max(0.,(it-self.ramp_start)/self.cfg.actor_ramp_iterations))
        out['actor']*=ramp;out['teacher']=out['actor']*self.cfg.teacher_lr/self.cfg.lr;out['actor_ramp']=ramp
        return out
    def observe(self,row):
        self.rows.append(copy.deepcopy(row));self.rows=self.rows[-20:]
    @staticmethod
    def _mean(rows,k):
        vals=[r.get(k) for r in rows]
        return statistics.mean(vals) if vals and all(v is not None and math.isfinite(v) for v in vals) else None
    def adjust(self,it):
        if it%1000:return None
        rr=self.rows
        complete=len(rr)==20 and [r['iteration'] for r in rr]==list(range(it-950,it+1,50))
        contexts={r['context'] for r in rr}
        agg={k:self._mean(rr,k) for k in ('reward','xy_rmse','falls','wm_error','critic_error','cts_error','teacher_drift')}
        agg.update(iteration=it,context=rr[-1]['context'] if rr else '',complete=complete and len(contexts)==1,
                   cts_stable=all(r.get('cts_stable',False) for r in rr))
        p=self.previous;matched=bool(p and p['complete'] and agg['complete'] and p['iteration']==it-1000 and p['context']==agg['context'])
        # Fixed composition: compare exactly the same branch/terrain/command bins.
        if matched:
            def composition(block):
                keys=set(block[0].get('groups',{}))
                for r in block:keys &= set(r.get('groups',{}))
                return {k:{m:statistics.mean(r['groups'][k][m] for r in block) for m in ('reward','xy_rmse','falls')} for k in keys}
            a,b=composition(self.previous_rows),composition(rr);keys=sorted(set(a)&set(b))
            # At least 70% of the previously represented strata, no missing branch.
            if len(keys)<max(2,math.ceil(.7*len(a))) or not all(any(k.startswith(str(br)+'/') for k in keys) for br in (0,1)):
                matched=False
            else:
                p=dict(p)
                for m in ('reward','xy_rmse','falls'):
                    p[m]=statistics.mean(a[k][m] for k in keys);agg[m]=statistics.mean(b[k][m] for k in keys)
                agg['matched_strata']=len(keys)
        before=dict(self.current);reasons={k:'hold' for k in self.current}
        if matched and it>self.cfg.lr_fixed_until:
            finite=lambda *keys:all(p[k] is not None and agg[k] is not None for k in keys)
            bad=finite('reward','xy_rmse','falls') and agg['reward']<p['reward']-.1*max(abs(p['reward']),.005) and agg['xy_rmse']>1.1*p['xy_rmse'] and agg['falls']>max(1.2*p['falls'],p['falls']+.1)
            good=finite('reward','xy_rmse','falls','critic_error','cts_error') and agg['reward']>=p['reward']-.02*max(abs(p['reward']),.005) and agg['xy_rmse']<=1.02*p['xy_rmse'] and agg['falls']<=p['falls']+.05 and agg['critic_error']<=1.1*p['critic_error'] and agg['cts_error']<=1.1*p['cts_error']
            conditions={'actor_bad':bad,'actor_good':good}
            for name,key,threshold,start in [('wm','wm_error',1.10,self.cfg.wm_fixed_until),('critic','critic_error',1.25,self.cfg.lr_fixed_until),('cts','cts_error',1.25,self.cfg.cts_fixed_until)]:
                eligible=it>=start and finite(key)
                conditions[name+'_bad']=eligible and agg[key]>threshold*max(p[key],1e-8)
                conditions[name+'_good']=eligible and agg[key]<.95*max(p[key],1e-8)
                if name=='cts':
                    conditions[name+'_bad'] &= agg['teacher_drift'] is not None and agg['teacher_drift']<.01
            for k,v in conditions.items():self.streak[k]=self.streak[k]+1 if v else 0
            for name in self.current:
                if it-self.last_change[name]<2000:continue
                direction=-1 if self.streak[name+'_bad']>=2 else 1 if self.streak[name+'_good']>=2 else 0
                if name=='cts' and not direction and it>=self.cfg.cts_fixed_until and it-self.last_change[name]>=5000 and agg['cts_stable'] and agg['teacher_drift'] is not None and agg['teacher_drift']<.01:
                    direction=-1;reasons[name]='multi_terrain_target_stable'
                if direction:
                    lo,hi=self.bounds[name];v=self.current[name]*(self.cfg.lr_adjust_factor**direction)
                    self.current[name]=min(hi,max(lo,v))
                    if self.current[name]!=before[name]:
                        self.last_change[name]=it
                        if reasons[name]=='hold':reasons[name]='two_full_windows_degraded' if direction<0 else 'two_full_windows_recovered'
                        self.streak[name+'_bad']=self.streak[name+'_good']=0
        else:
            self.streak={k:0 for k in self.streak}
        record={'iteration':it,'complete':agg['complete'],'comparable':matched,'window':agg,'previous_window':p,
                'before':before,'after':dict(self.current),'reasons':reasons,'algorithm':'actual_1000_iteration_windows_v2'}
        self.previous=agg;self.previous_rows=copy.deepcopy(rr)
        self.history=(self.history+[record])[-128:]
        return record
    def state_dict(self):
        return {k:copy.deepcopy(v) for k,v in vars(self).items() if k!='cfg'}
    def load_state_dict(self,s):
        if 'rows' not in s:raise ValueError('Retired single-step LR state cannot resume this experiment')
        for k,v in s.items():setattr(self,k,copy.deepcopy(v))
