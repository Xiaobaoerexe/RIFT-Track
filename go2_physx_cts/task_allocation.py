"""Task-resource feedback, independent of rewards and optimizer schedules.

Only matched, complete RoboGauge cohorts may move resources. Counts are targets;
the environment applies transfers at natural episode resets.
"""
import copy
import hashlib
import json
import math

BASE_COUNTS=(820,365,365,365,365,365,356,365,365,365)
SEEDS=(17,29,43,61,83)
PAYLOADS=(0.,1.2)
TERRAIN_WEIGHTS=({7:.4,1:.2,2:.2,3:.2},{7:1.},{0:.25,6:.25,8:.25,9:.25},
                 {4:1.},{5:1.},{10:1.},{7:.5,**{t:.5/9 for t in (0,1,2,3,4,5,6,8,9)}},
                 {11:1.},{12:1.},{13:1.})

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def apportion(total,weights):
    """Largest remainder, deterministic ties; exact integer conservation."""
    if total==0:return [0]*len(weights)
    den=sum(weights)
    if den<=0:raise ValueError('Positive allocation weights required')
    raw=[total*w/den for w in weights];out=[int(math.floor(x)) for x in raw]
    for i in sorted(range(len(out)),key=lambda i:(-(raw[i]-out[i]),i))[:total-sum(out)]:out[i]+=1
    return out

def redistribute(counts,poor,good,minimum=200):
    if len(counts)!=10 or any(type(n) is not int for n in counts) or sum(counts)!=4096 or min(counts)<minimum or any(n>2*b for n,b in zip(counts,BASE_COUNTS)):
        raise ValueError('Allocation requires ten tasks, 4096 environments and floor200')
    if set(poor)&set(good):raise ValueError('A recipient cannot also donate')
    demand={i:max(0,min(int(counts[i]*.1+.5),2*BASE_COUNTS[i]-counts[i])) for i in poor}
    capacity={i:max(0,min(int(counts[i]*.1),counts[i]-minimum)) for i in good}
    budget=min(sum(demand.values()),sum(capacity.values()))
    out=list(counts)
    if not budget:return out
    keys=sorted(demand)
    for i,n in zip(keys,apportion(budget,[demand[i] for i in keys])):out[i]+=n
    # Equal absolute donation, saturating donors at their own cap/floor.
    remaining=budget
    while remaining:
        donors=[i for i in sorted(capacity) if capacity[i]]
        take=apportion(remaining,[1]*len(donors))
        for i,n in zip(donors,take):
            n=min(n,capacity[i]);out[i]-=n;capacity[i]-=n;remaining-=n
    assert sum(out)==4096 and min(out)>=minimum
    return out

def summarize_cohort(iteration,rows,expected):
    """All requested cases must exist once, with finite metrics and explicit pass.

    Expected cases are emitted by the bridge before executing them; missing maps,
    crashes, duplicate cases and stale schemas cannot masquerade as low ability.
    """
    tasks={}
    for task,weights in enumerate(TERRAIN_WEIGHTS):
        exp=[r for r in expected if r['task_id']==task]
        got=[r for r in rows if r.get('iteration')==iteration and r.get('cohort_role')=='task_allocation' and r['task_id']==task]
        reason=None;by_id={r.get('allocation_case_id'):r for r in got}
        if not exp or len(by_id)!=len(got) or set(by_id)!={r['case_id'] for r in exp}:reason='missing_or_duplicate_cases'
        scores={}
        for branch in ('teacher','student'):
            scores[branch]={}
            for terrain in weights:
                cases=[r for r in exp if r['branch']==branch and r['terrain_id']==terrain]
                pairs={(r['seed'],r['payload_kg']) for r in cases}
                if not set((s,p) for s in SEEDS for p in PAYLOADS)<=pairs:reason='incomplete_terrain_seed_payload_coverage'
                values=[]
                for case in cases:
                    row=by_id.get(case['case_id'],{})
                    # A genuine fall before the one-second metric warmup is a
                    # failed trial, not a missing evaluation. Never turn a
                    # simulator error or nonfinite observation into a low score.
                    early_failure=(row.get('pass') is False and row.get('survived') is False and
                        row.get('samples')==0 and isinstance(row.get('survival_seconds'),(int,float)) and
                        math.isfinite(row['survival_seconds']) and 0<=row['survival_seconds']<=1.)
                    metrics_bad=any((row.get(k) is None and not early_failure) or
                        (row.get(k) is not None and not math.isfinite(float(row[k]))) for k in ('xy_rmse','yaw_rmse','height'))
                    if row.get('error') or type(row.get('pass')) is not bool or metrics_bad:
                        reason='invalid_case'
                    else:values.append(int(row['pass']))
                scores[branch][str(terrain)]=sum(values)/len(values) if values else None
        rates={b:sum(weights[t]*scores[b][str(t)] for t in weights) for b in scores} if reason is None else {}
        tasks[str(task)]={'valid':reason is None,'reason':reason,'cohort_id':fingerprint(sorted(exp,key=lambda r:r['case_id'])),
                          'branch_rates':rates,'terrain_rates':scores,'rate':min(rates.values()) if rates else None,
                          'expected_cases':len(exp),'actual_cases':len(got)}
    return {'version':1,'iteration':int(iteration),'tasks':tasks}

class TaskAllocator:
    def __init__(self):
        self.previous={};self.last_iteration=-1;self.history=[]

    def observe(self,report,counts,pending=False):
        it=int(report['iteration'])
        if it<20000 or it%5000 or it<=self.last_iteration:return None
        self.last_iteration=it;poor=[];good=[];states={}
        for i in range(10):
            row=report.get('tasks',{}).get(str(i),{});old=self.previous.get(str(i));state='hold'
            value=row.get('rate')
            valid=row.get('valid') and value is not None and math.isfinite(value) and 0<=value<=1
            matched=valid and old and old['iteration']==it-5000 and old['cohort_id']==row.get('cohort_id')
            if matched and it>=25000:
                if value<.5 and old['rate']<.5:poor.append(i);state='recipient'
                elif value>=.8 and old['rate']>=.8 and old['rate']-value<.1-1e-12:good.append(i);state='donor'
            states[str(i)]=state
            if valid:self.previous[str(i)]={'iteration':it,'cohort_id':row['cohort_id'],'rate':value}
            else:self.previous.pop(str(i),None)
        eligible=it>=25000 and it%5000==0 and not pending
        target=redistribute(counts,poor,good) if eligible else list(counts)
        record={'iteration':it,'before':list(counts),'target':target,'states':states,'pending_previous_transfer':pending,
                'changed':target!=list(counts),'minimum':200,'maximum':[2*x for x in BASE_COUNTS]}
        self.history=(self.history+[record])[-32:]
        return record

    def state_dict(self):return copy.deepcopy(vars(self))
    def load_state_dict(self,value):
        for key in ('previous','last_iteration','history'):setattr(self,key,copy.deepcopy(value[key]))
