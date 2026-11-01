"""Matched fixed-difficulty PhysX probes, separate from official transfer results."""
import copy,hashlib,json
import torch
from . import state as st
from .terrain import geometry_crop

def _ids(snapshot,key):return snapshot[key].reshape(-1).long()

@torch.no_grad()
def _native_probe(learner,policy,steps,per_task=24,high_speed=None):
    env=learner.env;dev=env.device;n=env.num_envs
    course=copy.deepcopy(env.curriculum_state_dict());was_eval=env.evaluation_mode
    command_override=env.command_override;rows=[]
    devices=[torch.device(dev).index or 0] if str(dev).startswith('cuda') else []
    try:
        env.evaluation_mode=True
        env.set_diagnostic_levels(0)
        for branch in (0,1):
            # Identical terrain, payload, command and initialization across
            # candidates; this clock does not advance the training curriculum.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(76119)
                if devices:torch.cuda.manual_seed_all(76119)
                env.command_override=None;env.reset(env.ids)
                snap=env._observe();tasks=_ids(snap,'task_ids')
                chosen=[]
                for task in range(10):chosen.append((tasks==task).nonzero().flatten()[:per_task])
                index=torch.cat(chosen);nt=len(index)
                payload=torch.zeros(n,device=dev);payload[index]=(torch.arange(nt,device=dev)%3)*.6
                env.set_payload(env.ids,payload)
                commands=torch.tensor([[.5,0,0],[0,0,0],[.5,.2,.3]],device=dev)
                cmd=commands[torch.arange(n,device=dev)%3].clone()
                cmd[tasks==1]=torch.tensor([1.5 if high_speed is None else high_speed,0,0],device=dev)
                cmd[(tasks==5)|(tasks==8)|(tasks==9)]=torch.tensor([.8,0,0],device=dev)
                env.command_override=cmd;env.command=cmd.clone();env._snapshot=env._observe()
                hidden=torch.zeros(nt,256,device=dev);image=torch.zeros(nt,1,48,64,device=dev)
                live=torch.ones(nt,device=dev,dtype=torch.bool);returns=torch.zeros(nt,device=dev)
                error=torch.zeros(nt,device=dev);yaw_error=error.clone();height=torch.zeros(nt,device=dev);slip=torch.zeros(nt,device=dev)
                counts=torch.zeros(nt,device=dev);failures=torch.zeros(nt,device=dev,dtype=torch.bool)
                success=torch.zeros(nt,device=dev,dtype=torch.bool)
                for j in range(steps):
                    full=env.snapshot();b={k:v[index] for k,v in full.items() if torch.is_tensor(v) and v.ndim and len(v)==n}
                    # Depth at25Hz, recurrent controller at50Hz. No stale image
                    # from another pose or replay trajectory is substituted.
                    if branch==1 and j%2==0:
                        image=learner.depth.render(b['pose'],b['scene_ids'],j,0,index,mode='inference')
                    priv=st.privileged(b['s'],b['a_prev'],b['command'],b['xi'],b['relief'])
                    act,hidden,_,_=policy.step(b['obs'].float(),priv,branch,depth=image,hidden=hidden,
                        height=geometry_crop(b['s'],b['pose'],env.field,kind='teacher'),task_ids=tasks[index])
                    action=torch.zeros(n,12,device=dev);action[index]=act
                    after,done,_=env.step(action)
                    s=after['s'][index];v=st.velocity_body(s)
                    returns+=after['reward'][index]*live
                    error+=(v[:,:2]-cmd[index,:2]).square().sum(-1)*live
                    yaw_error+=(s[:,9]-cmd[index,2]).square()*live
                    height+=s[:,0]*live;counts+=live
                    tele=after['telemetry'];feet=tele['foot_velocity'][index]
                    support=tele['contact_prob'][index,:4]
                    slip+=(feet[...,:2].norm(dim=-1)*support).sum(-1)/support.sum(-1).clamp_min(1)*live
                    failures|=live&after['terminated'][index].bool()
                    success|=live&(after['task_context'][index,3]>=.99)
                    live&=~done[index]
                for task in tasks[index].unique().tolist():
                    mask=tasks[index]==task;den=counts[mask].clamp_min(1)
                    trial_xy=(error[mask]/den).sqrt();trial_yaw=(yaw_error[mask]/den).sqrt()
                    tolerance=torch.maximum(torch.full_like(trial_xy,.2),.2*cmd[index][mask,:2].norm(dim=-1))
                    mastered=(~failures[mask])&(counts[mask]>=steps*.95)&(trial_xy<tolerance)&(trial_yaw<.3)
                    if task in (5,6,8,9):mastered=(~failures[mask])&success[mask]
                    rows.append({'task_id':task,'branch':('teacher','student')[branch],'trials':int(mask.sum()),
                        'mean_return':float(returns[mask].mean()),'survival_fraction':float((~failures[mask]).float().mean()),
                        'success_fraction':float(success[mask].float().mean()),
                        'own_success_rate':float(mastered.float().mean()),
                        'tracking_rmse':float((error[mask]/den).sqrt().mean()),
                        'height':float((height[mask]/den).mean()),'foot_slip_mps':float((slip[mask]/den).mean()),
                        'falls_per_env_min':float(failures[mask].sum()/mask.sum()/(steps*env.cfg.dt)*60),
                        'difficulty':'fixed level0; recovery fixedlevel5','seconds':steps*env.cfg.dt})
    finally:
        env.load_curriculum_state_dict(course);env.command_override=command_override
        env.evaluation_mode=True;env.reset(env.ids,advance_curriculum=False)
        env.evaluation_mode=was_eval;env._snapshot=env._observe()
        learner.reset_rollout_state()
    count=sum(r['trials'] for r in rows)
    def mean(key):return sum(r[key]*r['trials'] for r in rows)/max(count,1)
    teacher=[r for r in rows if r['branch']=='teacher']
    cohort=hashlib.sha256(json.dumps([(r['task_id'],r['trials'],r['seconds']) for r in teacher]).encode()).hexdigest()
    task_records={str(t):{'own_success_rate':sum(r['own_success_rate'] for r in rows if r['task_id']==t)/2}
                  for t in sorted(set(r['task_id'] for r in rows))}
    fixed={'kind':'teacher_fixed_eval','valid':bool(teacher),'cohort_id':cohort,
           'success_rate':sum(r['own_success_rate'] for r in teacher)/max(1,len(teacher)),
           'return':sum(r['mean_return'] for r in teacher)/max(1,len(teacher)),'iteration':learner.it}
    return {'iteration':learner.it,'rows':rows,'real_return':mean('mean_return'),'tasks':task_records,'teacher_fixed_eval':fixed,
            'falls_per_env_min':mean('falls_per_env_min'),'tracking_rmse':mean('tracking_rmse'),
            'protocol':'fixed seed76119, matched task/payload/commands, no replay or optimizer writes',
            'training_episode_reset_after_probe':True}

def native_candidate_probe(learner,policy):return _native_probe(learner,policy,steps=150,per_task=24)
def native_diagnostics(learner,steps=500):
    result=_native_probe(learner,learner.policy,steps,per_task=24)
    if learner.it>=5000:
        from .commands import speed_cap
        speed=float(speed_cap(1,learner.it))
        probe=_native_probe(learner,learner.policy,steps,per_task=24,high_speed=speed)
        result['high_speed_probe']={'command_vx':speed,'rows':[r for r in probe['rows'] if r['task_id']==1],
                                   'used_for_student_stability':False}
    return result

@torch.no_grad()
def model_diagnostics(learner,batch=2048):
    windows=learner.buf.windows(batch,learner.cfg.wm_window)
    loss,terms=learner.wm.loss(windows,learner.env.field,learner.env.consts.default_q,return_terms=True)
    state=windows['s'][:,0];pose=windows['pose'][:,0];initial=state.clone()
    error=torch.zeros(len(state),device=state.device);baseline=error.clone();den=error.clone()
    for j in range(learner.cfg.wm_window):
        xi=windows['xi'][:,j] if windows['xi'].ndim==3 else windows['xi']
        raw,dp,_=learner.wm.predict(state,windows['a'][:,j],windows['a_prev'][:,j],xi,
            geometry_crop(state,pose,learner.env.field,kind='world_model'),pose)
        pose=st.apply_pose_delta(pose,dp);state=st.reconcile_state(raw,pose,learner.env.field,learner.env.consts.default_q)
        target=windows['s_next'][:,j];valid=windows['valid'][:,j]
        error+=(state[:,4:10]-target[:,4:10]).square().mean(-1)*valid
        baseline+=(initial[:,4:10]-target[:,4:10]).square().mean(-1)*valid;den+=valid
    tasks=windows['task_ids'][:,0].reshape(-1).long()
    previous=getattr(learner,'_model_validation_history',{});current={};records={}
    for task in tasks.unique().tolist():
        mask=(tasks==task)&(den>0)
        if not mask.any():continue
        mse=float((error[mask]/den[mask]).mean());naive=float((baseline[mask]/den[mask]).mean())
        old=previous.get(str(task));valid=bool(torch.isfinite(error[mask]).all()) and mse<naive
        # Source-data improvement over persistence is an interpretable model
        # check. Two non-degrading diagnostics, together with real capability,
        # are required before supported external providers may begin fading.
        stable=old is not None and mse<=old*1.1
        records[str(task)]={'valid':valid,'stable':stable,'velocity_mse':mse,'persistence_velocity_mse':naive,
                            'windows':int(mask.sum()),'comparison':'8-step velocity prediction better than frozen initial state; <=10% regression from previous check'}
        current[str(task)]=mse
    learner._model_validation_history={**previous,**current}
    return {'iteration':learner.it,'loss':float(loss),'rows':[{'component':k,'normalized_loss':float(v)} for k,v in terms.items()],
            'task_validation':records,
            'sampling':'held-out sampled windows; no optimizer or curriculum updates',
            'note':'Normalized supervised errors are model checks, not locomotion capability.'}
