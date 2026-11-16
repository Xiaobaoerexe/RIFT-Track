"""Fixed PhysX cohorts for AS2 perception; separate from training and sim2sim.

Tracking thresholds, levels, payload endpoints and seeds are retained from
as2_baseline_2061be6/fixed_evaluation.py. Event completion and the fast-turn task
are explicit project extensions. No old Go2 transfer evaluator is invoked.
"""
import copy,hashlib,json,math,random
import torch
from . import state as st
from .commands import TERRAIN_NAMES,TERRAIN_RANGES,speed_cap,special_speed_contract
from .terrain import geometry_crop
from .task import COMPLETED,EPISODE_LIMIT
from .latent import prepare_latent_targets
from . import buffer as replay_buffer


def _ids(s,key):return s[key].reshape(-1).long()


def fixed_cases(cfg,iteration,levels=(2,5),seeds=(17,29,43,61,83)):
    """Commands and maps are fixed; unlocked flags only select the gate cohort."""
    unlocked=float(speed_cap(0,iteration));cases=[]
    tasks=range(4)
    for task in tasks:
        terrains=range(10) if task==0 else ((10,11,12) if task==3 else (7,))
        for terrain in terrains:
            name=TERRAIN_NAMES[terrain];cap=TERRAIN_RANGES[name][0][1]
            if task==1:speeds=(1.,2.,3.,4.,6.,8.)
            elif task==2:speeds=(.5,1.)
            elif task==3:speeds=(special_speed_contract()['minimum_speed_mps'],1.)
            elif terrain>=10:speeds=(.5,1.)
            else:
                speeds=sorted(set((0.,.5,min(1.,cap))+((1.5,2.,3.,4.,4.2,6.,8.) if terrain==7 else ())))
            for level in levels:
                for payload in (0.,cfg.payload_max_kg):
                    for seed in seeds:
                        for speed in speeds:
                            # New tasks are invisible to learning until 5k; fixed
                            # reports may probe them early but never gate on them.
                            active=(cfg.variant=='simple' or task==0 or iteration>=cfg.all_terrain_iteration)
                            cases.append({'task_id':task,'terrain_id':terrain,'terrain':name,'level':level,
                                'payload_kg':payload,'seed':seed,'speed':speed,
                                'allocation':active and speed<=min(unlocked,cap)+1e-4,
                                'command_protocol':'alternating_turn_1s' if task==2 else 'constant_forward'})
    return cases


def _command(case_rows,step,device):
    command=torch.zeros((len(case_rows),3),device=device)
    command[:,0]=torch.tensor([c['speed'] for c in case_rows],device=device)
    turns=torch.tensor([c['task_id']==2 for c in case_rows],device=device)
    # Repeatable one-second command discontinuities, with visible body-frame
    # velocity intent. No hidden task label is given to either encoder.
    sign=1. if (step//50)%2==0 else -1.
    command[turns,1]=.3*sign;command[turns,2]=.8*sign
    return command


def _fixed_reset(rows,device):
    jitter=[];delays=[]
    for c in rows:
        rng=random.Random(c['seed']+c['terrain_id']*1009+c['level']*103)
        jitter.append([rng.uniform(-.15,.15),rng.uniform(-.15,.15)])
        delays.append(rng.randrange(3))
    return {'jitter':torch.tensor(jitter,device=device),'yaw':torch.zeros(len(rows),device=device),
            'delay':torch.tensor(delays,device=device)}


@torch.no_grad()
def native_diagnostics(learner,steps=500,levels=(2,5),seeds=(17,29,43,61,83)):
    if steps<=50:raise ValueError('Fixed diagnostics need more than the 50-step startup exclusion')
    env=learner.env;cfg=learner.cfg;device=env.device;n=env.num_envs
    cases=fixed_cases(cfg,learner.it,levels,seeds);records=[]
    geometry_fingerprint=hashlib.sha256(json.dumps(env.terrain_metadata,sort_keys=True,default=str).encode()).hexdigest()
    saved=copy.deepcopy(env.curriculum_state_dict())
    flags={name:getattr(env,name) for name in ('evaluation_mode','calibration_flat','command_override','fixed_reset','iteration')}
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    initial_layouts={}
    try:
        env.evaluation_mode=True;env.calibration_flat=False
        for start in range(0,len(cases),n):
            selected=cases[start:start+n];count=len(selected)
            expanded=selected+[selected[i%count] for i in range(n-count)]
            tasks=torch.tensor([c['task_id'] for c in expanded],device=device)
            terrains=torch.tensor([c['terrain_id'] for c in expanded],device=device)
            for branch in (0,1):
                torch.manual_seed(91317+start)
                env.task_ids.copy_(tasks);env.terrain_ids.copy_(terrains)
                env.terrain_levels.copy_(torch.tensor([c['level'] for c in expanded],device=device))
                env.task_assignment_version+=1;env.fixed_reset=_fixed_reset(expanded,device)
                command=_command(expanded,0,device);env.command_override=command
                env.reset(env.ids,advance_curriculum=False)
                env.delay.copy_(env.fixed_reset['delay']);env.xi[:,3]=env.delay*cfg.physics_dt
                env.set_payload(env.ids,torch.tensor([c['payload_kg'] for c in expanded],device=device))
                env._command_views();env._snapshot=env._observe()
                initial=env.snapshot()
                signature=torch.cat((initial['pose'][:count],initial['s'][:count,st.Q],env.xi[:count]),-1).cpu()
                if branch==0:initial_layouts[start]=signature
                elif not torch.allclose(signature,initial_layouts[start],atol=1e-5,rtol=0):
                    raise RuntimeError('Teacher/student diagnostic initial conditions differ')
                hidden=torch.zeros(n,256,device=device);image=torch.zeros(n,1,48,64,device=device)
                alive=torch.ones(n,device=device,dtype=torch.bool);failed=torch.zeros_like(alive)
                completed=torch.zeros_like(alive);exposure=torch.zeros(n,device=device)
                sums=torch.zeros(n,7,device=device);samples=torch.zeros(n,device=device)
                post_sums=torch.zeros(n,3,device=device);post_samples=torch.zeros(n,device=device)
                returns=torch.zeros(n,device=device);event_returns=torch.zeros(n,device=device)
                expected=torch.minimum(initial['task_context'][:,EPISODE_LIMIT]/cfg.dt,torch.full((n,),steps,device=device)).round()
                noise_episode=torch.tensor([c['seed'] for c in expanded],device=device)
                trajectory=torch.arange(start,start+n,device=device)
                for j in range(steps):
                    command=_command(expanded,j,device);env.command_override=command
                    env._command_views();env._snapshot=env._observe();before=env.snapshot()
                    if branch and j%cfg.depth_period==0:
                        image=learner.depth.captured(before['pose'],_ids(before,'scene_ids'),j,noise_episode,trajectory)
                    priv=st.privileged(before['s'],before['a_prev'],command,before['xi'],before['relief'])
                    height=geometry_crop(before['s'],before['pose'],env.field,kind='teacher') if branch==0 else before['s'].new_zeros((n,2,1,1))
                    action,hidden,_,_=learner.policy.step(before['obs'],priv,branch,depth=image,hidden=hidden,height=height)
                    after,done,_=env.step(action)
                    state=after['s'];velocity=st.velocity_body(state,after['pose'])
                    failure=after['terminated'].reshape(-1).bool();good=alive&~failure
                    was_complete=completed.clone();completed|=alive&(after['task_context'][:,COMPLETED]>=.5)
                    failed|=alive&failure;exposure+=alive
                    returns+=after['reward'].reshape(-1)*alive
                    event_returns+=after['event_reward'].reshape(-1)*alive
                    feet=after['foot_velocity'].reshape(n,4,3);support=after['contact_prob'][:,:4]
                    slip=(feet[...,:2].norm(dim=-1)*support).sum(-1)/support.sum(-1).clamp_min(1)
                    xy=(velocity[:,:2]-command[:,:2]).square().mean(-1)
                    yaw=(state[:,9]-command[:,2]).square()
                    value=torch.stack((xy,yaw,state[:,0],velocity[:,0],velocity[:,1],state[:,9],slip),-1)
                    if j>=50:sums+=value*good[:,None];samples+=good
                    post=good&was_complete;post_sums+=value[:,:3]*post[:,None];post_samples+=post
                    alive&=~done.reshape(-1)
                mean=sums/samples.clamp_min(1)[:,None];xy=mean[:,0].sqrt();yaw=mean[:,1].sqrt()
                post_mean=post_sums/post_samples.clamp_min(1)[:,None]
                full=(exposure>=expected)&~failed
                tolerance=torch.maximum(torch.full_like(xy,.2),.2*torch.tensor([c['speed'] for c in expanded],device=device))
                passed=full&(samples>0)&(xy<tolerance)&(yaw<.3)&(mean[:,2]>.28)
                event=(tasks==3) if cfg.variant!='simple' else torch.zeros_like(full)
                post_ok=(post_samples==0)|((post_mean[:,0].sqrt()<tolerance)&(post_mean[:,1].sqrt()<.3)&(post_mean[:,2]>.28))
                passed=torch.where(event,full&completed&post_ok,passed)
                data=torch.cat((mean,xy[:,None],yaw[:,None],full[:,None],passed[:,None],completed[:,None],samples[:,None],
                    exposure[:,None],returns[:,None],event_returns[:,None],post_mean,post_samples[:,None]),1)[:count].cpu().tolist()
                for case,row in zip(selected,data):
                    records.append({**case,'branch':('teacher','student')[branch],'iteration':learner.it,
                        'height':row[2],'vx':row[3],'vy':row[4],'wz':row[5],'foot_slip_mps':row[6],
                        'xy_rmse':row[7],'yaw_rmse':row[8],'survived':bool(row[9]),'pass':bool(row[10]),
                        'hard_success':bool(row[11]),'samples':int(row[12]),'exposure_steps':int(row[13]),
                        'return':row[14],'event_return':row[15],'post_success_xy_rmse':math.sqrt(max(0,row[16])),
                        'post_success_yaw_rmse':math.sqrt(max(0,row[17])),'post_success_height':row[18],'post_success_samples':int(row[19])})
    finally:
        env.command_override=None;env.fixed_reset=None;env.evaluation_mode=True
        env.load_curriculum_state_dict(saved);env.reset(env.ids,advance_curriculum=False)
        env.set_payload(env.ids,torch.rand(n,device=device)*cfg.payload_max_kg)
        for name,value in flags.items():setattr(env,name,value)
        env._snapshot=env._observe();learner.reset_rollout_state();learner.refresh_providers()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    terrains={};task_reports={}
    for task in range(4):
        selected=[r for r in records if r['task_id']==task and r['allocation']]
        if selected:
            task_reports[str(task)]={'cases':len(selected),'branch_rates':{
                b:sum(r['pass'] for r in selected if r['branch']==b)/max(1,sum(r['branch']==b for r in selected)) for b in ('teacher','student')}}
    for terrain in range(13 if cfg.variant=='simple' else 10):
        # Terrain allocation in event variants acts only inside task 0. Never
        # let crossing/high-speed/turning results alter that group's quota.
        selected=[r for r in records if r['terrain_id']==terrain and r['allocation'] and (cfg.variant=='simple' or r['task_id']==0)]
        if not selected:continue
        rates={branch:sum(r['pass'] for r in selected if r['branch']==branch)/max(1,sum(r['branch']==branch for r in selected)) for branch in ('teacher','student')}
        identity=[{k:r[k] for k in ('task_id','terrain_id','level','seed','payload_kg','speed','command_protocol','branch')} for r in selected]
        protocol={'version':'as2_visual_fixed_h036_50hz_v2','steps':steps,'cases':identity,'depth_period':cfg.depth_period,'variant':cfg.variant,
            'geometry_fingerprint':geometry_fingerprint,'depth_weights':learner.depth.metadata.get('direct_model',{}).get('checkpoint_sha256')}
        cohort=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        valid=all(all(math.isfinite(r[k]) for k in ('xy_rmse','yaw_rmse','height','return')) for r in selected)
        terrains[str(terrain)]={'valid':valid,'rate':min(rates.values()),'branch_rates':rates,'cohort_id':cohort,'cases':len(selected)}
    return {'iteration':learner.it,'duration_seconds':steps*cfg.dt,'rows':records,'terrains':terrains,'tasks':task_reports,
        'protocol':'as2_visual_fixed_h036_50hz_v2','source':'AS2 2061be6 fixed_evaluation.py tracking thresholds; project event/vision extensions',
        'thresholds':{'xy_rmse':'< max(0.2, 0.2*forward_command)','yaw_rmse':.3,'mean_height_min_m':.28,'startup_excluded_steps':50},
        'levels':list(levels),'seeds':list(seeds),'payload_kg':[0.,cfg.payload_max_kg],
        'notes':['Finite task timeout is completion of the observation interval, not a physical fall.',
                 'Event tasks require hard completion and retained post-success tracking/stability.',
                 'Unopened speed/task cases are reported but excluded from training gates.',
                 'This is native PhysX evidence; no AS2 visual MuJoCo validation is implied.'],
        'training_episodes_reset_after_probe':True}


@torch.no_grad()
def model_diagnostics(learner,batch=256,horizon=None):
    horizon=learner.cfg.wm_window if horizon is None else horizon
    try:
        w=learner.buf.windows(batch,horizon,validation=True)
    except ValueError as exc:
        # Existing125/156 processes can import this file lazily while their
        # already-loaded Replay is the old version. Never require a new symbol
        # at import time or swallow that legacy sampler's unrelated ValueErrors.
        if not isinstance(exc,getattr(replay_buffer,'ReplayPartitionUnavailable',())):raise
        return {'iteration':learner.it,'horizon':horizon,'available':False,'reason':str(exc)}
    target=prepare_latent_targets(w,learner.depth,learner.policy.student_encoder.depth_encoder,learner.cfg.depth_period)
    scales=(learner.wm.out_norm.var.sqrt().clamp_min(1e-4),learner.wm.telemetry_norm.var.sqrt().clamp_min(1e-3))
    errors=learner.wm.validation_errors(w,learner.env.field,learner.env.consts.default_q,scales,latent_targets=target)
    tasks=w['task_ids'][:,0].reshape(-1);rows=[]
    for task in tasks.unique().tolist():
        mask=tasks==task
        rows.append({'task_id':int(task),'windows':int(mask.sum()),**{name:float(value[mask].mean()) for name,value in errors.items()}})
    return {'iteration':learner.it,'horizon':horizon,'available':True,'rows':rows,'mean_errors':{name:float(value.mean()) for name,value in errors.items()},
        'sampling':'held-out real windows with re-encoded real 50Hz visual targets',
        'note':'Read-only normalized prediction errors; no running-statistics, optimizer, curriculum or Replay updates.'}


@torch.no_grad()
def model_horizon_diagnostics(learner):
    """Full-label 8/16/32-step audit; never a stop rule or LR feedback input."""
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    try:return {str(h):model_diagnostics(learner,horizon=h) for h in (8,16,32)}
    finally:
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
