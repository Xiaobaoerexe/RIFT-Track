"""Paired fixed PhysX terrain cohorts; no optimizer/replay updates."""
import hashlib,json,random
import torch
from . import state as st
from .commands import RANGE_ITERATIONS,RANGE_STAGES,TERRAIN_RANGES

@torch.no_grad()
def fixed_native_diagnostics(learner,steps=500,levels=(2,5),seeds=(17,29,43,61,83)):
    env=learner.env;p=learner.policy;n=env.num_envs;dev=env.device
    stage=sum(learner.it>=v for v in RANGE_ITERATIONS);unlocked=RANGE_STAGES[stage][0][1]
    names=env.terrain_names;cases=[]
    for t,name in enumerate(names):
        speeds=[0.,.5,min(1.,TERRAIN_RANGES[name][0][1])]
        if name=='flat':speeds += [1.5,2.,3.,4.,4.2]
        speeds=sorted(set(speeds))
        for level in levels:
            for load in (0.,env.cfg.payload_max_kg):
                for seed in seeds:
                    for speed in speeds:
                        # Keep all high speed diagnostics; only unlocked cases drive allocation.
                        for br in (0,1):
                            cases.append(dict(terrain_id=t,terrain=name,level=level,payload_kg=load,seed=seed,speed=speed,branch=br,
                                              allocation=speed<=min(unlocked,TERRAIN_RANGES[name][0][1])+.0001))
    if len(cases)>n:raise ValueError('Fixed cohort exceeds num_envs; use a full-scale validation')
    count=len(cases);expanded=cases+[cases[i%count] for i in range(n-count)]
    ti=torch.tensor([c['terrain_id'] for c in expanded],device=dev);br=torch.tensor([c['branch'] for c in expanded],device=dev)
    lv=torch.tensor([c['level'] for c in expanded],device=dev);load=torch.tensor([c['payload_kg'] for c in expanded],device=dev)
    cmd=torch.zeros(n,3,device=dev);cmd[:,0]=torch.tensor([c['speed'] for c in expanded],device=dev)
    jitter=[];delay=[]
    for c in expanded:
        rng=random.Random(c['seed']+c['terrain_id']*1009+c['level']*103)
        jitter.append([rng.uniform(-.15,.15),rng.uniform(-.15,.15)]);delay.append(rng.randrange(3))
    fixed={'jitter':torch.tensor(jitter,device=dev),'yaw':torch.zeros(n,device=dev),'delay':torch.tensor(delay,device=dev)}
    saved=env.curriculum_state_dict();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    env.evaluation_mode=True;env.fixed_reset=fixed
    try:
        torch.manual_seed(91317)
        if env.field is not None:
            terrain=env.scene.terrain
            for t in range(len(names)):
                ids=(ti==t).nonzero().flatten();cols=(env.terrain_lookup.column_types==t).nonzero().flatten()
                terrain.terrain_types[ids]=cols[(ids//2)%len(cols)]
            terrain.terrain_levels.copy_(lv)
            terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
        env.command_override=cmd;env.reset(env.ids);env.set_payload(env.ids,load);env._snapshot=env._observe()
        alive=torch.ones(n,device=dev,dtype=torch.bool);sums=torch.zeros(n,6,device=dev);samples=torch.zeros(n,device=dev)
        for i in range(steps):
            b=env.snapshot();priv=st.privileged(b['s'],b['a_prev'],cmd,env.xi,b['relief'])
            a=torch.empty(n,12,device=dev)
            for branch in (0,1):
                mask=br==branch;a[mask]=p(b['obs'][mask],priv[mask],branch)
            after,done,_=env.step(a);alive &= ~done
            if i>=50:
                s=after['s'];v=st.velocity_body(s)
                vals=torch.stack(((v[:,:2]-cmd[:,:2]).square().mean(-1),(s[:,9]-cmd[:,2]).square(),s[:,0],v[:,0],v[:,1],s[:,9]),-1)
                sums+=vals*alive[:,None];samples+=alive
        avg=sums/samples.clamp_min(1)[:,None];xy=avg[:,0].sqrt();yaw=avg[:,1].sqrt()
        passed=alive&(samples>0)&(xy<torch.maximum(torch.full_like(xy,.2),cmd[:,0].abs()*.2))&(yaw<.3)&(avg[:,2]>.22)
        data=torch.cat((avg,xy[:,None],yaw[:,None],alive[:,None],passed[:,None],samples[:,None]),1)[:count].cpu().tolist()
        rows=[]
        for c,v in zip(cases,data):
            rows.append({**c,'iteration':learner.it,'branch':('teacher','student')[c['branch']],'vx':v[3],'vy':v[4],'wz':v[5],
                         'height':v[2],'xy_rmse':v[6],'yaw_rmse':v[7],'survived':bool(v[8]),'pass':bool(v[9]),'samples':int(v[10])})
        report={}
        for t in range(len(names)):
            selected=[r for r in rows if r['terrain_id']==t and r['allocation']]
            rates={b:sum(r['pass'] for r in selected if r['branch']==b)/sum(r['branch']==b for r in selected) for b in ('teacher','student')}
            identity=[{k:r[k] for k in ('terrain_id','level','seed','payload_kg','speed','branch')} for r in selected]
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            valid=all(all(torch.isfinite(torch.tensor(r[k])).item() for k in ('xy_rmse','yaw_rmse','height')) for r in selected)
            report[str(t)]={'valid':valid,'rate':min(rates.values()),'branch_rates':rates,'cohort_id':key,'cases':len(selected)}
        return {'iteration':learner.it,'duration_seconds':steps*env.cfg.dt,'rows':rows,'terrains':report,
                'protocol':'fixed_levels_seeded_reset_shared_teacher_student_cases_PhysX','noise':True,'delay':'fixed per seed 0..2 physics steps',
                'note':'Pass is 10s tracking/stability, not full terrain traversal. Only unlocked speed cases drive allocation.'}
    finally:
        env.fixed_reset=None;env.command_override=None;env.load_curriculum_state_dict(saved)
        env.steps.zero_();env.reset(env.ids)
        env.set_payload(env.ids,torch.rand(n,device=dev)*env.cfg.payload_max_kg)
        env.evaluation_mode=False;env._snapshot=env._observe()
        learner.ou.zero_()
        if learner.driver is not None:learner.driver.reset(torch.ones(n,device=dev,dtype=torch.bool))
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
