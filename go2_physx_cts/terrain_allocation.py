"""Port of depth task_allocation.py at b160cfeb7dee395bc2b54726c2853cfc4aa551dc.
Same 10%/two-cohort/floor/cap rules; blind terrain IDs and fixed PhysX evidence.
"""
import copy,math
def apportion(total,weights):
    if not total:return [0]*len(weights)
    raw=[total*w/sum(weights) for w in weights];out=[int(x) for x in raw]
    for i in sorted(range(len(out)),key=lambda i:(-(raw[i]-out[i]),i))[:total-sum(out)]:out[i]+=1
    return out
def redistribute(counts,poor,good,base,minimum=200):
    floor=[min(minimum,b) for b in base]
    assert sum(counts)==sum(base) and not(set(poor)&set(good))
    demand={i:max(0,min(int(counts[i]*.1+.5),2*base[i]-counts[i])) for i in poor}
    capacity={i:max(0,min(int(counts[i]*.1),counts[i]-floor[i])) for i in good}
    budget=min(sum(demand.values()),sum(capacity.values()));out=list(counts)
    if not budget:return out
    keys=sorted(demand)
    for i,n in zip(keys,apportion(budget,[demand[i] for i in keys])):out[i]+=n
    remaining=budget
    while remaining:
        donors=[i for i in sorted(capacity) if capacity[i]]
        for i,n in zip(donors,apportion(remaining,[1]*len(donors))):
            n=min(n,capacity[i]);out[i]-=n;capacity[i]-=n;remaining-=n
    assert sum(out)==sum(counts) and all(f<=n<=2*b for f,n,b in zip(floor,out,base))
    return out
class TerrainAllocator:
    def __init__(self,base):
        self.base=list(base);self.previous={};self.history=[];self.last_iteration=-1;self.version=0
    def observe(self,it,report,counts,pending=False):
        if it<20000 or it%5000 or it<=self.last_iteration:return None
        self.last_iteration=it;poor=[];good=[];states={}
        for i in range(len(counts)):
            r=report.get(str(i),{});old=self.previous.get(str(i));state='hold'
            valid=r.get('valid',False) and r.get('rate') is not None and math.isfinite(r['rate'])
            matched=valid and old and old['iteration']==it-5000 and old['cohort_id']==r['cohort_id']
            if matched and it>=25000:
                if r['rate']<.5 and old['rate']<.5:poor.append(i);state='recipient'
                elif r['rate']>=.8 and old['rate']>=.8 and old['rate']-r['rate']<.1-1e-12:good.append(i);state='donor'
            states[str(i)]=state
            if valid:self.previous[str(i)]={**r,'iteration':it}
            else:self.previous.pop(str(i),None)
        target=redistribute(counts,poor,good,self.base) if not pending else list(counts)
        if target!=list(counts):self.version+=1
        record={'iteration':it,'before':list(counts),'target':target,'states':states,'pending':pending,'version':self.version,'changed':target!=list(counts)}
        self.history=(self.history+[record])[-64:]
        return record
    def state_dict(self):return copy.deepcopy(vars(self))
    def load_state_dict(self,s):self.__dict__.update(copy.deepcopy(s))
