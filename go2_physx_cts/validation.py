"""Real PhysX control, update ownership, restore and export validation."""
import copy,time
from pathlib import Path
import torch
from . import state as st
from .networks import Policy

@torch.no_grad()
def physical_probe(env,driver,steps):
    if driver is None:raise ValueError('Physical positive control requires official driver')
    result={};mass_checks=0
    for name in ('standing','official'):
        env.reset(env.ids);env._snapshot=env._observe();driver.reset(torch.ones(env.num_envs,device=env.device,dtype=torch.bool))
        env.command_override=env.command.new_tensor([.5,0,0]).expand(env.num_envs,3).clone();env._snapshot=env._observe()
        rows=[];fall=0;began=time.monotonic()
        for i in range(steps):
            old_mass=env.xi[:,4].clone()
            a=torch.zeros(env.num_envs,12,device=env.device) if name=='standing' else driver.act(env.command)
            after,done,m=env.step(a);driver.reset(done);fall+=m['falls_all']
            assert torch.equal(old_mass[~done],env.xi[~done,4]),'Mass changed within episode'
            if i%100==0 and env.cfg.payload_max_kg:
                from .physx_env import tensor
                actual=tensor(env.robot.data.body_mass)[:,env._base_id]-env._nominal_masses[:,env._base_id]
                assert torch.allclose(actual,env.xi[:,4],atol=1e-6),'PhysX mass and context differ'
                mass_checks+=1
            if i>=min(100,steps//4):rows.append([float(st.velocity_body(after['s'])[:,0].mean()),m['height'],m['reward'],m['xy_rmse']])
        x=torch.tensor(rows)
        result[name]={'vx':float(x[:,0].mean()),'height':float(x[:,1].mean()),'reward_per_step':float(x[:,2].mean()),'xy_rmse':float(x[:,3].mean()),'fall_count':fall,'env_steps':steps*env.num_envs,'seconds':time.monotonic()-began}
    result['expert_motion_pass']=result['official']['vx']>.2 and result['official']['height']>.2 and result['official']['fall_count']<env.num_envs*.25
    result['reward_order_pass']=result['official']['reward_per_step']>result['standing']['reward_per_step']
    result['fixed_episode_mass_checks']=mass_checks
    env.command_override=None
    if not result['expert_motion_pass'] or not result['reward_order_pass']:raise RuntimeError(f'Official positive control failed: {result}')
    return result

def parameter_copy(m):return [p.detach().clone() for p in m.parameters()]
def changed(m,before):return any(not torch.equal(p,b) for p,b in zip(m.parameters(),before))

def learner_smoke(l,iterations):
    while l.buf.size<=l.cfg.wm_window+1:l.collect()
    torch.cuda.reset_peak_memory_stats();measures=[]
    l.it=l.cfg.warmup_iterations+l.cfg.actor_ramp_iterations
    l.env.iteration=l.it
    l.learning_rates()
    teacher=parameter_copy(l.policy.teacher_encoder);actor=parameter_copy(l.policy.actor);student=parameter_copy(l.policy.student_encoder)
    wm=parameter_copy(l.wm);critic=parameter_copy(l.critic)
    geometry={}
    if l.env.field is not None:
        from .task import crop_height_from_map
        snap=l.env.snapshot()
        error=(snap['height_above_crop']-crop_height_from_map(snap['s'],snap['pose'],l.env.field)).abs()
        assert torch.isfinite(error).all() and snap['height_crop_valid'].all()
        geometry={'crop_error_mean_m':float(error.mean()),'crop_error_p95_m':float(error.quantile(.95)),
                  'crop_error_max_m':float(error.max()),'collision_height_map':l.env.field.metadata,
                  'terrain_allocation':l.env.terrain_lookup.metadata}
    fixed_mass_checks=0
    for i in range(iterations):
        old_episode=l.env.episode.clone();old_payload=l.env.xi[:,4].clone()
        torch.cuda.synchronize();start_unix=time.time();t=time.monotonic();l.collect();tc=time.monotonic();wl=l.fit_world_model();tw=time.monotonic()
        same=(old_episode==l.env.episode).flatten()
        assert torch.equal(old_payload[same],l.env.xi[same,4]),'Payload changed within episode'
        snap=l.env.snapshot();assert torch.equal(snap['cmd_state'][:,3:6],snap['command'])
        fixed_mass_checks+=1
        stats=l.improve_policy();ta=time.monotonic()
        if i==0:
            assert changed(l.policy.teacher_encoder,teacher) and changed(l.policy.actor,actor)
            assert not changed(l.policy.student_encoder,student),'Student parameters changed during actor step'
            assert all(p.grad is None for p in l.wm.parameters()),'Frozen WM accumulated actor gradients'
        cl=l.update_cts();torch.cuda.synchronize();end=time.monotonic()
        measures.append({'start_unix':start_unix,'end_unix':time.time(),'collect':tc-t,'wm':tw-tc,'actor_critic':ta-tw,'cts':end-ta,'total':end-t})
        l.it+=1
    assert changed(l.wm,wm) and changed(l.critic,critic) and changed(l.policy.student_encoder,student)
    assert l.nonfinite==0
    path=l.save(l.output/'restore_probe.pt');probe=Policy(l.cfg.actor_hidden).to(l.device)
    probe.load_state_dict(torch.load(path,map_location=l.device,weights_only=False)['policy'])
    hist=l.env.snapshot()['obs'][:64]
    with torch.no_grad():err=(probe(hist,None,1)-l.policy(hist,None,1)).abs().max().item()
    exported=l.export(l.output/'export_probe.pt').to(l.device)
    with torch.no_grad():export_err=(exported(hist)-l.policy(hist,None,1)).abs().max().item()
    assert err==0 and export_err<1e-5
    expected={key:copy.deepcopy(mod.state_dict()) for key,mod in [('policy',l.policy),('critic',l.critic),('world_model',l.wm)]}
    expected_count=copy.deepcopy(l.counts)
    for mod in (l.policy,l.critic,l.wm):
        with torch.no_grad():next(mod.parameters()).add_(1.)
    l.load(path,continue_run=True)
    for key,mod in [('policy',l.policy),('critic',l.critic),('world_model',l.wm)]:
        assert all(torch.equal(v,expected[key][k]) for k,v in mod.state_dict().items()),f'{key} restore mismatch'
    assert l.counts==expected_count
    assert all(len(getattr(l,'opt_'+k).state)>0 for k in ('wm','actor','critic','cts'))
    l.last_good=path;w_before=parameter_copy(l.wm);archive_size=l.archive.size;replay_size=l.buf.size
    assert l.rollback() and not changed(l.wm,w_before) and archive_size==l.archive.size and replay_size==l.buf.size
    from .diagnostics import native_diagnostics,model_diagnostics
    levels=l.env.curriculum_state_dict();counts=copy.deepcopy(l.counts)
    native=native_diagnostics(l,steps=100);model=model_diagnostics(l,batch=256)
    if levels['terrain_levels'] is not None:
        assert torch.equal(levels['terrain_levels'],l.env.curriculum_state_dict()['terrain_levels'])
    assert counts==l.counts and l.buf.size==replay_size
    assert native['rows'] and model['rows']
    return {'passed':True,'backend':type(l.env.robot).__module__,'num_envs':l.cfg.num_envs,'imag_batch':l.cfg.imag_batch,
            'wm_batch':l.cfg.wm_batch,'buffer_len':l.cfg.buffer_len,'replay_bytes':l.buf.storage_bytes,
            'timings':measures,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'restore_max_error':err,'full_model_optimizer_restore':True,'export_max_error':export_err,'counts':l.counts,'last_stats':stats,
            'gradient_ownership':True,'rollback_retains_wm_and_data':True,'wm_loss':wl,'cts_loss':cl,
            'geometry':geometry,'fixed_episode_mass_checks':fixed_mass_checks,
            'diagnostic_smoke':{'native_rows':len(native['rows']),'model_rows':len(model['rows']),
                'no_optimizer_replay_or_course_updates':True}}
