"""Call audit_environment(env) inside a root-owned fresh PhysX smoke process.

This runs physical steps but no optimizer. It resets all environments on exit;
never call it on a live formal learner with a replay history to preserve.
"""
import json
import torch
from go2_physx_cts.physx_env import tensor
from go2_physx_cts.task import crop_height_from_map
from go2_physx_cts.terrain import ground


@torch.no_grad()
def audit_environment(env,steps=16):
    if env.num_envs<3:raise ValueError('Need three environments for0/1/2kg cases')
    before_curriculum=env.curriculum_state_dict()
    previous_mode,previous_command=env.evaluation_mode,env.command_override
    report={'kind':'actual_physx_parameter_and_collision_geometry_audit_not_capability',
            'terrain':env.terrain_metadata,'physics_steps':steps*round(env.cfg.dt/env.cfg.physics_dt)}
    try:
        env.evaluation_mode=True;env.command_override=torch.zeros(env.num_envs,3,device=env.device)
        env.reset(env.ids)
        selected=env.ids[:3]
        com=tensor(env.robot.data.body_com_pos_b)[selected,env._base_id].clone()
        inertia=tensor(env.robot.data.body_inertia)[selected,env._base_id].clone()
        env.set_payload(selected,torch.tensor([0.,1.,2.],device=env.device))
        initial_mass=tensor(env.robot.data.body_mass).clone()
        initial_xi=env.xi.clone()
        report['body_names']=list(env.robot.body_names)
        report['base_body_index']=env._base_id
        report['initial_first3_body_mass']=initial_mass[selected].tolist()
        report['initial_first3_com']=com.tolist();report['initial_first3_inertia']=inertia.tolist()
        report['initial_first3_xi']=initial_xi[selected].tolist()
        report['mass_dtype']=str(initial_mass.dtype)
        report['step_parameter_trace']=[]
        assert torch.allclose(initial_mass[selected,env._base_id]-env._nominal_masses[selected,env._base_id],
                              initial_mass.new_tensor([0.,1.,2.]),atol=1e-6,rtol=0.)
        assert torch.equal(com,tensor(env.robot.data.body_com_pos_b)[selected,env._base_id])
        assert torch.equal(inertia,tensor(env.robot.data.body_inertia)[selected,env._base_id])
        actual_payload=env.xi[selected,4].clone()
        max_mass_change=0.;resets=0;first_stepped_mass=None;max_post_first_step_mass_change=0.
        for step in range(steps):
            after,done,metrics=env.step(torch.zeros(env.num_envs,12,device=env.device))
            assert torch.isfinite(after['s']).all()
            assert torch.equal(env.command,env.command_override)
            current=tensor(env.robot.data.body_mass)
            mass_delta=current-initial_mass
            com_delta=tensor(env.robot.data.body_com_pos_b)[selected,env._base_id]-com
            inertia_delta=tensor(env.robot.data.body_inertia)[selected,env._base_id]-inertia
            largest=int(mass_delta.abs().flatten().argmax())
            max_mass_change=max(max_mass_change,float(mass_delta.abs().max()))
            if first_stepped_mass is None:first_stepped_mass=current.clone()
            post_first_delta=float((current-first_stepped_mass).abs().max())
            max_post_first_step_mass_change=max(max_post_first_step_mass_change,post_first_delta)
            report['step_parameter_trace'].append({'step':step+1,'automatic_resets':int(done.sum()),
                'mass_max_change_kg':float(mass_delta.abs().max()),
                'mass_change_since_first_solver_readback_kg':post_first_delta,
                'largest_mass_change_env_body':[largest//mass_delta.shape[1],largest%mass_delta.shape[1]],
                'first3_body_mass_delta_kg':mass_delta[selected].tolist(),
                'first3_base_mass_kg':current[selected,env._base_id].tolist(),
                'first3_xi':env.xi[selected].tolist(),'xi_max_change':float((env.xi-initial_xi).abs().max()),
                'first3_base_com_max_change_m':float(com_delta.abs().max()),
                'first3_base_inertia_max_change':float(inertia_delta.abs().max()),
                'recorded_base_mass_vs_live_max_kg':float((env.actual_base_mass_kg-current[:,env._base_id]).abs().max()),
                'payload_context_vs_live_max_kg':float((env.xi[:,4]-(current[:,env._base_id]-env._nominal_masses[:,env._base_id])).abs().max())})
            resets+=int(done.sum())
        report['mass_max_change_kg']=max_mass_change
        report['mass_comparison_atol_kg']=1e-6;report['mass_comparison_rtol']=0.
        report['max_post_first_step_mass_change_kg']=max_post_first_step_mass_change
        # Measured first-solver float32 inverse-mass roundtrip was4.768e-7kg.
        # Permit only that bounded readback quantization, never cumulative drift.
        if max_mass_change>1e-6 or max_post_first_step_mass_change!=0.:
            raise RuntimeError(f'Actual PhysX mass changed during no-update diagnostic: max_abs_delta_kg={max_mass_change!r}; see step_parameter_trace')
        env.reset(env.ids)
        reset_delta=tensor(env.robot.data.body_mass)-initial_mass
        report['explicit_reset_mass_max_change_kg']=float(reset_delta.abs().max())
        assert torch.allclose(tensor(env.robot.data.body_mass),initial_mass,atol=1e-6,rtol=0.),report['explicit_reset_mass_max_change_kg']
        assert torch.equal(tensor(env.robot.data.body_mass),first_stepped_mass),'Mass changed after initial solver readback on explicit reset'
        if env.field is not None:
            assert torch.equal(env.scene.terrain.terrain_levels,before_curriculum['terrain_levels'])
        report.update(payload_kg=actual_payload.tolist(),actual_base_mass_kg=initial_mass[selected,env._base_id].tolist(),
                      com_unchanged=True,inertia_unchanged=True,mass_max_change_kg=max_mass_change,
                      automatic_resets=resets,evaluation_preserves_terrain_levels=True)
        if env.field is not None:
            terrain=env.scene.terrain;lookup=env.terrain_lookup
            points=[];labels=[]
            for category,name in enumerate(lookup.names):
                columns=(lookup.column_types==category).nonzero().flatten()
                if not len(columns):continue
                column=int(columns[0])
                for level in (0,5,9):
                    for offset in (-3.5,-1.5,0.,1.5,3.5):
                        point=terrain.terrain_origins[level,column].clone()
                        point[0]+=offset;point[2]+=1.
                        points.append(point);labels.append((name,level,offset))
            positions=torch.stack(points);yaw=torch.full_like(positions[:,0],.37)
            measured,valid=env.field.raycaster.height_crop(positions,yaw)
            pose=torch.cat((positions[:,:2],yaw[:,None]),-1)
            state=torch.zeros(len(positions),70,device=env.device)
            state[:,0]=positions[:,2]-ground(positions[:,:2],env.field)
            approximated=crop_height_from_map(state,pose,env.field)
            error=(approximated-measured).abs()
            report['heightfield_vs_collision_ray']={'samples':len(points),'all_crop_rays_valid':bool(valid.all()),
                'mean_abs_m':float(error.mean()),'max_abs_m':float(error.max()),
                'per_terrain':{name:{'count':sum(x[0]==name for x in labels),
                    'mean_abs_m':float(error[torch.tensor([x[0]==name for x in labels],device=env.device)].mean()),
                    'max_abs_m':float(error[torch.tensor([x[0]==name for x in labels],device=env.device)].max())} for name in lookup.names},
                'scope':'Deterministic off-center points at levels0/5/9; geometry approximation, not robot traversal'}
        report['status']='PASS_PHYSICAL_PARAMETERS_GEOMETRY_MEASURED'
        return report
    except Exception as error:
        report['status']='FAIL_PHYSICAL_AUDIT'
        report['exception']=type(error).__name__+': '+str(error)
        print('TERRAIN_PHYSICS_AUDIT_FAILURE '+json.dumps(report),flush=True)
        raise
    finally:
        env.load_curriculum_state_dict(before_curriculum)
        env.steps.zero_();env.evaluation_mode=previous_mode;env.command_override=previous_command
        env.reset(env.ids)
