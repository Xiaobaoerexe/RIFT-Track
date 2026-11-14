"""Real PhysX hard-event/MC calibration with a frozen mature blind AS2 policy.

This is a bounded diagnostic, not policy training. A temporary event Critic is
fitted on complete real episodes and tested on separate complete episodes.
Artificial goal/fall/timeout starts are explicitly labelled boundary tests; they
do not demonstrate crossing ability or improvement of the new depth policy.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def discounted_event_returns(reward,done,valid,gamma=.99):
    import torch
    tail=torch.zeros(reward.shape[0],device=reward.device,dtype=reward.dtype)
    rows=[]
    for t in reversed(range(reward.shape[1])):
        tail=torch.where(valid[:,t],reward[:,t]+gamma*(~done[:,t]).to(reward.dtype)*tail,0.)
        rows.append(tail)
    return torch.stack(rows[::-1],1)


def _reference(project,checkpoint,env):
    import torch
    from go2_physx_cts import state as st
    source=Path(project).resolve()/'go2_physx_cts'
    spec=importlib.util.spec_from_file_location('_event_blind',source/'__init__.py',submodule_search_locations=[str(source)])
    if spec is None or spec.loader is None:raise ImportError('Cannot load pinned blind AS2 model')
    package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
    networks=importlib.import_module('_event_blind.networks')
    ck=torch.load(checkpoint,map_location=env.device,weights_only=False)
    if ck.get('cfg',{}).get('robot')!='as2':raise ValueError('Reference checkpoint must be AS2')
    policy=networks.Policy(tuple(ck['cfg']['actor_hidden'])).to(env.device).eval().requires_grad_(False)
    policy.load_state_dict(ck['policy'],strict=True)
    for key,value in ck.get('consts',{}).items():
        if not torch.allclose(getattr(env.consts,key),value.to(env.device),rtol=1e-5,atol=1e-6):
            raise ValueError('Reference physical constants disagree: '+key)
    @torch.no_grad()
    def action(snapshot,branch):
        priv=st.privileged(snapshot['s'],snapshot['a_prev'],snapshot['command'],snapshot['xi'],snapshot['relief'])
        result=torch.empty((env.num_envs,12),device=env.device)
        for b in (0,1):
            mask=branch==b
            if mask.any():result[mask]=policy(snapshot['obs'][mask].float(),priv[mask],b)
        return result
    return action


def _prepare_cases(env,include_boundaries):
    import torch
    from go2_physx_cts import state as st
    from go2_physx_cts.task import EPISODE_LIMIT
    cases=[('natural_gap',3,10,False),('natural_up',3,11,False),
           ('natural_down',3,12,False),('natural_recovery',4,7,False)]
    if include_boundaries:
        cases += [('artificial_goal_gap',3,10,True),('artificial_goal_up',3,11,True),
            ('artificial_goal_down',3,12,True),('artificial_timeout',3,10,True),
            ('artificial_fallen',3,10,True)]
    rows=torch.arange(env.num_envs,device=env.device);kind=rows%len(cases)
    repeat=rows//len(cases);branch=repeat%2
    if env.num_envs<len(cases)*6:
        raise ValueError('Need at least 6 replicas per case for both reference branches and heldout episodes')
    env.evaluation_mode=True;env.calibration_flat=False;env.iteration=5000
    env.task_ids=torch.tensor([cases[int(i)][1] for i in kind],device=env.device)
    env.terrain_ids=torch.tensor([cases[int(i)][2] for i in kind],device=env.device)
    env.terrain_levels.zero_();env.task_assignment_version+=1
    env.reset(env.ids,advance_curriculum=False)
    env.set_payload(env.ids,torch.zeros(env.num_envs,device=env.device))
    command=torch.zeros((env.num_envs,3),device=env.device)
    command[:,0]=torch.where(env.task_ids==3,.8,.5)
    env.command_override=command;env._command_views()
    snapshot=env._observe();pose=snapshot['pose'].clone()
    for i,(name,_,_,artificial) in enumerate(cases):
        ids=(kind==i).nonzero().flatten()
        if name.startswith('artificial_goal'):
            pose[ids,:3]=env.goals[ids]+pose.new_tensor([0.,0.,.30])
            pose[ids,3:]=pose.new_tensor([0.,0.,0.,1.])
        elif name=='artificial_timeout':env.task_context[ids,EPISODE_LIMIT]=.8
        elif name=='artificial_fallen':
            pose[ids,2]+=.12
            pose[ids,3:]=st.rotvec_quat(pose.new_tensor([[1.9,0.,0.]]).expand(len(ids),-1))
        else:continue
        if name!='artificial_timeout':
            env.robot.write_root_pose_to_sim_index(root_pose=pose[ids],env_ids=ids)
            env.robot.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros((len(ids),6)),env_ids=ids)
            env.robot.write_joint_position_to_sim_index(position=env.consts.default_q.expand(len(ids),12),env_ids=ids)
            env.robot.write_joint_velocity_to_sim_index(velocity=pose.new_zeros((len(ids),12)),env_ids=ids)
            env.robot.reset(ids);env.sensor.reset(ids)
    snapshot=env._observe()
    env.hist.zero_();env.hist.copy_(st.push_history(env.hist,st.frame(snapshot['s'],env.last,command,noise=True)))
    env._snapshot=env._observe()
    # Entire episodes, not neighboring states, are held out. Alternating repeats
    # provides separate heldout teacher and student reference trajectories.
    heldout=repeat%6>=4
    return cases,kind,branch,heldout


def save_reset_evidence(env,cases,case_id,branch,heldout,output):
    """Capture actual articulation readback before the first physics step."""
    import torch
    from go2_physx_cts.physx_env import tensor
    data=env.robot.data
    loaded={'root_pose_w':tensor(data.root_link_pose_w).detach().cpu().clone(),
        'joint_position':tensor(data.joint_pos).detach().cpu().clone(),
        'joint_velocity':tensor(data.joint_vel).detach().cpu().clone(),
        'body_link_pose_w':tensor(data.body_link_pose_w).detach().cpu().clone(),
        'case_id':case_id.cpu(),'branch':branch.cpu(),'heldout':heldout.cpu(),
        'recovery_kind':env.recovery_kind.cpu().clone(),'task_ids':env.task_ids.cpu().clone(),
        'terrain_ids':env.terrain_ids.cpu().clone(),'origins':env.origins.cpu().clone(),
        'task_context':env.task_context.cpu().clone(),'command':env.command.cpu().clone(),
        'joint_limits':env.consts.soft_limits.cpu().clone()}
    q=loaded['joint_position'];limits=loaded['joint_limits']
    finite=all(bool(torch.isfinite(loaded[key]).all()) for key in
        ('root_pose_w','joint_position','joint_velocity','body_link_pose_w'))
    joint_ok=(q>=limits[:,0]-1e-5)&(q<=limits[:,1]+1e-5)
    quaternion_norm=loaded['root_pose_w'][:,3:7].norm(dim=-1)
    if not finite or not joint_ok.all() or not torch.allclose(quaternion_norm,torch.ones_like(quaternion_norm),atol=1e-4,rtol=0):
        raise RuntimeError('Invalid actual reset pose/joint readback')
    names=['left_side','right_side','back','belly','front_tilt','rear_tilt']
    selected=(loaded['task_ids']==4).nonzero().flatten().tolist()
    record={'schema':'as2_recovery_reset_readback_v1','phase':'after reset/write, before first physics step',
        'quaternion_order':'x,y,z,w','root_pose_order':'world x,y,z,qx,qy,qz,qw',
        'joint_names':list(env.robot.joint_names),'body_names':list(env.robot.body_names),
        'finite':finite,'joint_limits_respected':bool(joint_ok.all()),
        'quaternion_max_norm_error':float((quaternion_norm-1).abs().max()),
        'collision_overlap_tested':False,
        'visualization':'Load the same AS2 USD/URDF, apply root_pose_w and named joint_position. This readback is not a collision-penetration proof.',
        'recovery_samples':[{'env':i,'case':cases[int(loaded['case_id'][i])][0],
            'branch':'teacher' if int(loaded['branch'][i])==0 else 'student',
            'recovery_kind':names[int(loaded['recovery_kind'][i])],
            'root_pose_w':loaded['root_pose_w'][i].tolist(),
            'joint_position':q[i].tolist(),'joint_velocity':loaded['joint_velocity'][i].tolist(),
            'body_link_pose_w':loaded['body_link_pose_w'][i].tolist()} for i in selected]}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.save({**loaded,'metadata':record},output/'recovery_reset_states.pt')
    (output/'recovery_reset_states.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return {'samples':len(selected),'finite':finite,'joint_limits_respected':bool(joint_ok.all()),
        'path':'recovery_reset_states.json','collision_overlap_tested':False}


def run(env,reference,output,*,fit_steps=300,batch=1024,include_boundaries=True,seed=1729):
    import torch
    from go2_physx_cts import state as st
    from go2_physx_cts.task import COMPLETED,reward_budget
    from go2_physx_cts.value import Critic
    torch.manual_seed(seed)
    cases,case_id,branch,heldout=_prepare_cases(env,include_boundaries)
    reset_readback=save_reset_evidence(env,cases,case_id,branch,heldout,output)
    critic=Critic(variant='events',mixed_precision=env.cfg.mixed_precision).to(env.device)
    live=torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
    records={key:[] for key in ('features','event','done','valid','success','failure','timeout',
        'completed','pose','priv','context','contacts','task_ids','joint_position','joint_velocity')}
    start=time.monotonic();max_steps=round(max(row['episode_seconds'] for row in reward_budget()['tasks'][3:])/env.cfg.dt)
    for _ in range(max_steps):
        before=env.snapshot()
        with torch.no_grad():
            action=reference(before,branch)
            after,done,_=env.step(action,defer_metrics=True)
            priv=st.privileged(after['s'],action,before['command'],before['xi'],after['relief'])
            context=before['task_context'];contacts=after['contact_prob'];pose=after['pose']
            x=critic.event_features(priv,branch,before['task_ids'],context,pose,contacts)
            values={'features':x,'event':after['event_reward'],'done':done,'valid':live.clone(),
                'success':after['success'],'failure':after['terminated'],'timeout':after['timeout'],
                'completed':context[:,COMPLETED],'pose':pose,'priv':priv,'context':context,'contacts':contacts,
                'task_ids':before['task_ids'].clone(),'joint_position':after['s'][:,st.Q]+env.consts.default_q,
                'joint_velocity':after['s'][:,st.QD]}
            for key in records:records[key].append(values[key].detach())
            live=live&~done
        if not live.any():break
    rows={key:torch.stack(value,1) for key,value in records.items()}
    if live.any():raise RuntimeError('Unterminated episodes would give truncated MC labels; increase collection horizon')
    mask=rows['valid'].bool();returns=discounted_event_returns(rows['event'],rows['done'].bool(),mask,env.cfg.gamma)
    train_mask=mask&~heldout[:,None];test_mask=mask&heldout[:,None]
    x,y=rows['features'][train_mask],returns[train_mask]
    if not len(x) or not test_mask.any():raise RuntimeError('No independent real train/heldout episodes')
    critic.update_stats(x)
    optimizer=torch.optim.Adam(critic.parameters(),lr=2e-4)
    losses=[]
    for _ in range(fit_steps):
        ids=torch.randint(len(x),(min(batch,len(x)),),device=env.device)
        loss=critic.loss(x[ids],y[ids],kind='event');optimizer.zero_grad(set_to_none=True)
        loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);optimizer.step()
        losses.append(float(loss.detach()))
    critic.eval().requires_grad_(False)
    with torch.no_grad():
        pred=torch.zeros_like(returns)
        pred[mask]=torch.cat([critic.forward_event(part) for part in rows['features'][mask].split(4096)])
    def metrics(selection):
        target=returns[selection];prediction=pred[selection]
        if not len(target):return {'count':0}
        std=target.std(unbiased=False)
        correlation=None
        if len(target)>1 and std>1e-8 and prediction.std(unbiased=False)>1e-8:
            correlation=float(torch.corrcoef(torch.stack((target,prediction)))[0,1])
        mse=(prediction-target).square().mean();constant_mse=(target-y.mean()).square().mean()
        return {'count':len(target),'target_mean':float(target.mean()),'prediction_mean':float(prediction.mean()),
            'rmse':float(mse.sqrt()),'mae':float((prediction-target).abs().mean()),
            'correlation':correlation,'target_std':float(std),'zero_predictor_rmse':float(target.square().mean().sqrt()),
            'training_mean_predictor_rmse':float(constant_mse.sqrt()),
            'mse_improvement_over_training_mean':float(1-mse/constant_mse) if constant_mse>1e-8 else None}
    t=torch.arange(mask.shape[1],device=env.device)[None,:]
    has_success=(rows['success'].bool()&mask).any(1)
    success_time=torch.where(has_success,rows['success'].argmax(1),mask.shape[1]+100)
    has_failure=(rows['failure'].bool()&mask).any(1)
    failure_time=torch.where(has_failure,rows['failure'].float().argmax(1),mask.shape[1]+100)
    to_success=success_time[:,None]-t;to_failure=failure_time[:,None]-t
    categories={'before_success_32':(to_success>0)&(to_success<=32)&has_success[:,None],
        'on_success':rows['success'].bool(),'after_success':rows['completed']>=.5,
        'before_failure_32':(to_failure>0)&(to_failure<=32)&has_failure[:,None],
        'on_failure':rows['failure'].bool(),'timeout':rows['timeout'].bool()}
    pair_rows=test_mask.nonzero()
    chosen=pair_rows[torch.linspace(0,len(pair_rows)-1,min(256,len(pair_rows)),device=env.device).long()]
    er,tt=chosen.unbind(-1)
    with torch.enable_grad():
        pose=rows['pose'][er,tt].detach().clone().requires_grad_(True)
        inp=critic.event_features(rows['priv'][er,tt],branch[er],rows['task_ids'][er,tt],rows['context'][er,tt],pose,rows['contacts'][er,tt])
        value=critic.forward_event(inp)
        grad=torch.autograd.grad(value.sum(),pose)[0]
    if not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite event-value pose gradient')
    report={'schema':'as2_hard_event_mc_validation_v1','status':'measured_not_policy_validated',
        'scope':'Frozen mature blind AS2 PhysX trajectories; fresh same-architecture event Critic MC fit only',
        'policy_updated':False,'world_model_used_for_mc_labels':False,'soft_success_used':False,
        'physx_steps':mask.shape[1],'episodes':env.num_envs,'heldout_episodes':int(heldout.sum()),
        'seed':seed,'fit_steps':fit_steps,'fit_batch':batch,'critic_lr':2e-4,
        'reset_readback':reset_readback,
        'train':metrics(train_mask),'heldout':metrics(test_mask),
        'state_groups':{name:metrics(selection&test_mask) for name,selection in categories.items()},
        'heldout_by_reference_branch':{str(b):{'all':metrics(test_mask&(branch==b)[:,None]),
            'state_groups':{name:metrics(selection&test_mask&(branch==b)[:,None]) for name,selection in categories.items()}}
            for b in (0,1)},
        'success_bonus_once':bool(((rows['success']*mask).sum(1)<=1).all()),
        'pose_gradient':{'l2':float(grad.norm()),'finite':True,'nonzero':bool(grad.abs().sum()>0),
            'meaning':'Input derivative of fitted value with respect to actual afterstate pose; direction accuracy is not established'},
        'fit_first_loss':losses[0] if losses else None,'fit_last_loss':losses[-1] if losses else None,
        'seconds':time.monotonic()-start,'cases':[],
        'limitations':['No Actor update and no proof of improved hard success.',
            'Artificial goal starts test settlement, not crossing/exploration ability.',
            'MC fit under blind reference is not calibration of a future depth/GRU policy.',
            'A finite nonzero value gradient alone does not establish its physical correctness.',
            'Root-Q one-step TD and recovery hybrid-mode approximation need separate ongoing validation.']}
    for i,(name,task_id,terrain,artificial) in enumerate(cases):
        episodes=case_id==i;selection=test_mask&episodes[:,None]
        report['cases'].append({'name':name,'task_id':task_id,'terrain_id':terrain,
            'artificial_boundary_initialization':artificial,'episodes':int(episodes.sum()),
            'hard_successes':int(has_success[episodes].sum()),'failures':int(has_failure[episodes].sum()),
            'teacher_successes':int((has_success&episodes&(branch==0)).sum()),
            'student_successes':int((has_success&episodes&(branch==1)).sum()),'heldout':metrics(selection)})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.save({'rows':{key:value.cpu() for key,value in rows.items()},'mc_returns':returns.cpu(),
        'case_id':case_id.cpu(),'branch':branch.cpu(),'heldout_episodes':heldout.cpu(),
        'critic':critic.state_dict(),'reward_budget':reward_budget()},output/'real_event_evidence.pt')
    (output/'event_value_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    from isaaclab.sim import SimulationContext
    from go2_physx_cts.config import TrainCfg
    from go2_physx_cts.train import physics_configuration
    from go2_physx_cts.task import configure_reward_budget
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('config','output','usd','reference-project','reference-policy'):parser.add_argument('--'+flag,required=True)
    parser.add_argument('--num-envs',type=int,default=108)
    parser.add_argument('--fit-steps',type=int,default=300)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--natural-only',action='store_true')
    parser.add_argument('--reward-calibration',help='Explicit local budget override when checking another host configuration')
    add_launcher_args(parser);args=parser.parse_args()
    cfg=TrainCfg.from_record(json.loads(Path(args.config).read_text()))
    if args.reward_calibration:cfg.reward_calibration=args.reward_calibration
    if cfg.variant=='simple' or not cfg.reward_calibration:
        raise ValueError('Use events/MoE configuration with measured AS2 reward budget')
    cfg.num_envs=args.num_envs;cfg.compile=False;cfg.seed=args.seed;cfg.validate(production=False)
    configure_reward_budget(cfg.reward_calibration)
    sim_cfg=physics_configuration(cfg)
    with launch_simulation(sim_cfg,args):
        from go2_physx_cts.physx_env import PhysXEnv
        sim=SimulationContext(sim_cfg);env=PhysXEnv(cfg,sim,args.usd)
        reference=_reference(args.reference_project,args.reference_policy,env)
        report=run(env,reference,args.output,fit_steps=args.fit_steps,
                   include_boundaries=not args.natural_only,seed=args.seed)
        report['reference_checkpoint_sha256']=hashlib.sha256(Path(args.reference_policy).read_bytes()).hexdigest()
        report['reference_project']=str(Path(args.reference_project).resolve())
        (Path(args.output)/'event_value_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({key:report[key] for key in ('status','heldout','success_bonus_once','pose_gradient')},indent=2))


if __name__=='__main__':main()
