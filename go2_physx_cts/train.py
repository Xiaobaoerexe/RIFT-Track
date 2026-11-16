"""Versioned AS2 perception training or explicitly separated GPU qualification."""
import os
if 'PYTORCH_ALLOC_CONF' not in os.environ and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
import argparse,hashlib,importlib,importlib.util,importlib.metadata,json,signal,subprocess,sys,time
from pathlib import Path
from .config import TrainCfg

BASELINE_COMMIT='2061be6b0d592afe9e81223206cf5f864e45b9f4'


def physics_configuration(cfg):
    from isaaclab.app.sim_launcher import _make_physics_cfg
    from isaaclab.sim import SimulationCfg,RigidBodyMaterialCfg
    physics=_make_physics_cfg('isaacsim_physx')
    physics.gpu_max_rigid_patch_count=2*1024*1024
    physics.gpu_collision_stack_size=512*1024*1024
    # Retain the measured 4096-robot depth cold-start capacity until this new
    # terrain/asset combination is measured. These capacities are explicitly
    # recorded as an engineering allocation, not AS2 motor/solver parameters.
    scale=max(1/16,cfg.num_envs/4096)
    physics.gpu_found_lost_aggregate_pairs_capacity=max(2**21,int(96*1024*1024*scale))
    physics.gpu_found_lost_pairs_capacity=max(2**21,int(2**23*scale))
    physics.gpu_total_aggregate_pairs_capacity=max(2**21,int(2**23*scale))
    physics.enable_external_forces_every_iteration=True
    return SimulationCfg(dt=cfg.physics_dt,device=cfg.device,render_interval=4,physics=physics,
        physics_material=RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.,
                                              friction_combine_mode='average',restitution_combine_mode='average'))


def _torch(value):
    import torch
    if isinstance(value,torch.Tensor):return value
    if hasattr(value,'torch'):return value.torch
    import warp as wp
    return wp.to_torch(value)


def _tensor_record(value):
    """Actual all-env readback: first env, component ranges, shape and checksum."""
    tensor=_torch(value).detach().float().cpu().contiguous()
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),
            'sha256':hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            'first':tensor[0].tolist() if tensor.ndim else tensor.item(),
            'minimum_over_envs':tensor.amin(0).tolist() if tensor.ndim else tensor.item(),
            'maximum_over_envs':tensor.amax(0).tolist() if tensor.ndim else tensor.item()}


def write_contract(env,cfg,usd,out,sim_cfg,code_commit):
    import torch
    from . import UPSTREAM_COMMIT
    versions={'python':sys.version,'python_executable':sys.executable}
    for name in ('isaacsim','isaaclab','torch','numpy','warp-lang','trimesh'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not a registered distribution'
    view=env.robot.root_view;loaded={}
    backend_names=list(view.shared_metatype.dof_names)
    canonical_names=list(env.robot.joint_names)
    if len(backend_names)!=len(canonical_names) or set(backend_names)!=set(canonical_names):
        raise RuntimeError('Backend DOF names do not match the canonical AS2 joints')
    reorder=[backend_names.index(name) for name in canonical_names]
    def joint_read(value):return _torch(value)[:,reorder]
    # Explicit PD resides in the actuator. Solver stiffness may correctly be 0;
    # report both and do not silently equate solver and actuator parameters.
    methods={'solver_stiffness':'get_dof_stiffnesses','solver_damping':'get_dof_dampings',
        'armature':'get_dof_armatures','solver_effort_limits':'get_dof_max_forces',
        'solver_velocity_limits':'get_dof_max_velocities','joint_limits':'get_dof_limits',
        'body_mass':'get_masses','body_inertia':'get_inertias','body_com':'get_coms'}
    for name,method in methods.items():
        if not hasattr(view,method):raise RuntimeError('Required loaded-physics readback missing: '+method)
        raw=getattr(view,method)()
        loaded[name]=_tensor_record(raw if name.startswith('body_') else joint_read(raw))
    actuator=env.robot.actuators['legs']
    if hasattr(actuator,'stiffness') and hasattr(actuator,'damping'):
        live_pd={'stiffness':_torch(actuator.stiffness),'damping':_torch(actuator.damping)}
        actuator_reader='Lab actuator live buffers'
        for name in ('actuator_effort_limit','actuator_velocity_limit'):
            if hasattr(actuator,name):loaded[name]=_tensor_record(getattr(actuator,name))
    else:
        # Isaac Lab 3 can execute its PD controller using the native actuator
        # adapter while the actual physics backend remains PhysX. Read the
        # controller's owned live storage, never reconstruct gains from cfg.
        from isaaclab.actuators.newton import read_group_parameter
        group_names=list(env.robot.actuators._group_joint_names['legs'])
        group_order=[group_names.index(name) for name in canonical_names]
        def controller(component,attr):
            return read_group_parameter(env.robot.actuators,'legs',component,attr)[:,group_order]
        live_pd={'stiffness':controller('controller','kp'),'damping':controller('controller','kd')}
        actuator_reader='isaaclab.actuators.newton.read_group_parameter: live controller storage; physics is PhysX'
        for name in ('max_effort','max_velocity'):
            try:loaded['actuator_'+name]=_tensor_record(controller('clamping',name))
            except ValueError as exc:loaded['actuator_'+name]={'unavailable':str(exc),'fallback_to_configuration':False}
    for name,value in live_pd.items():loaded['actuator_'+name]=_tensor_record(value)
    loaded['data_joint_effort_limits']=_tensor_record(env.robot.data.joint_effort_limits)
    loaded['default_joint_position']=_tensor_record(env.robot.data.default_joint_pos)
    loaded['soft_joint_limits']=_tensor_record(env.robot.data.soft_joint_pos_limits)
    masses=_torch(env.robot.data.body_mass).sum(-1)
    from .robots import as2_motor_values
    expected_armature,expected_kp,expected_kd,_,_=as2_motor_values(cfg.device)
    checks={}
    for label,actual,expected in (
        ('armature',joint_read(getattr(view,methods['armature'])()),expected_armature),
        ('actuator_stiffness',live_pd['stiffness'],expected_kp),
        ('actuator_damping',live_pd['damping'],expected_kd)):
        actual=_torch(actual).float();expected=expected.to(actual.device)
        checks[label]=bool(torch.allclose(actual,expected.expand_as(actual),rtol=1e-5,atol=1e-6))
        if not checks[label]:raise RuntimeError('Loaded AS2 contract mismatch: '+label)
    nominal=env._nominal_masses.sum(-1)
    checks['nominal_mass_17_64kg']=bool(torch.allclose(nominal,torch.full_like(nominal,17.64),rtol=0,atol=.01))
    if not checks['nominal_mass_17_64kg']:raise RuntimeError('AS2 nominal mass differs from fixed-foot 17.64 kg asset')
    cuda={}
    if torch.cuda.is_available():
        props=torch.cuda.get_device_properties(cfg.device)
        cuda={'name':props.name,'total_memory_bytes':props.total_memory,'capability':[props.major,props.minor],
              'allocator':os.environ.get('PYTORCH_ALLOC_CONF',os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))}
    record={'schema':'as2_depth_physics_contract_v1','physics_backend':type(env.robot).__module__,
        'robot':'as2','variant':cfg.variant,'usd_path':str(Path(usd).resolve()),
        'usd_sha256':hashlib.sha256(Path(usd).read_bytes()).hexdigest(),'code_commit':code_commit,
        'as2_baseline_commit':BASELINE_COMMIT,'upstream_supermimic':UPSTREAM_COMMIT,
        'upstream_go2rllab':'6c95d9feef4814c25eb6536e61bb38e1d1d718d4','versions':versions,'cuda':cuda,
        'joint_names':canonical_names,'backend_dof_names':backend_names,'backend_to_canonical_dof_indices':reorder,
        'body_names':list(env.robot.body_names),'loaded_joint_parameter_order':'canonical joint_names; raw view explicitly reordered',
        'actuator_parameter_reader':actuator_reader,
        'total_mass_range_kg':[float(masses.min()),float(masses.max())],'actual_loaded_physics':loaded,'loaded_contract_checks':checks,
        'motor':{k:v.detach().cpu().tolist() for k,v in vars(env.consts).items()},
        'context_dim':5,'wm_state_dim':70,'full_pose_dim':7,'wm_pose_delta_dim':6,
        'reward_telemetry_dim':40,'wm_regression_telemetry_dim':44,'wm_contact_dim':14,
        'teacher_map':[2,64,128],'wm_local_map':[2,41,49],'depth_image':[1,48,64],
        'depth_latent_dim':128,'wm_capture_context_dim':0,'student_history_dim':450,'student_hidden_dim':256,
        'task_count':cfg.task_count,'value_channels':1,
        'actor_experts':cfg.actor_experts,'proprio_experts':cfg.proprio_experts,
        'actor_structure':'plain MLP' if cfg.actor_experts==1 else 'dense 3-expert MoE',
        'policy_hz':1/cfg.dt,'physics_hz':1/cfg.physics_dt,'depth_hz':1/(cfg.dt*cfg.depth_period),
        'network_precision':'bf16 mixed' if cfg.mixed_precision else 'fp32',
        'depth_cnn_optimizer':'world_model only, clean-depth reconstruction',
        'future_imagination':'50Hz WM latent recurrence; 10Hz held policy visual input; no future depth-image rendering',
        'imagination_policy_depth_hz':1/(cfg.dt*cfg.imagination_depth_period),
        'height_target_m':cfg.base_height_target,'latent_actor_stop_gradient':cfg.latent_actor_stop_gradient,
        'payload_model':'rigid mass at existing base COM, reset only; inertia unchanged',
        'payload_max_kg':cfg.payload_max_kg,'control_delay':'per-episode 0/1/2 physics substeps',
        'curriculum_clock':'one real vector step per learner iteration',
        'terrain':env.terrain_metadata,'reward_calibration':cfg.reward_calibration,
        'physx_allocation':{k:getattr(sim_cfg.physics,k) for k in ('gpu_max_rigid_patch_count','gpu_collision_stack_size','gpu_found_lost_aggregate_pairs_capacity','gpu_found_lost_pairs_capacity','gpu_total_aggregate_pairs_capacity','enable_external_forces_every_iteration')},
        'transfer_validation':'Native PhysX only. Old Go2 visual MuJoCo ABI is not reused as AS2 evidence.'}
    if cfg.reward_calibration:record['reward_calibration_sha256']=hashlib.sha256(Path(cfg.reward_calibration).read_bytes()).hexdigest()
    (out/'physics_contract.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    return record


def calibrate_reference(env,reference_project,reference_policy,out):
    """Load the mature blind AS2 model under a separate package namespace."""
    import torch
    from . import state as st
    from .calibration import calibrate_as2_reward
    project=Path(reference_project).resolve();source=project/'go2_physx_cts'
    spec=importlib.util.spec_from_file_location('_blind',source/'__init__.py',submodule_search_locations=[str(source)])
    if spec is None or spec.loader is None:raise ImportError('Cannot load reference AS2 package')
    package=importlib.util.module_from_spec(spec);sys.modules['_blind']=package;spec.loader.exec_module(package)
    source_networks=importlib.import_module('_blind.networks')
    ck=torch.load(reference_policy,map_location=env.device,weights_only=False)
    if ck.get('cfg',{}).get('robot')!='as2':raise ValueError('Calibration checkpoint is not AS2')
    policy=source_networks.Policy(tuple(ck['cfg']['actor_hidden'])).to(env.device).eval().requires_grad_(False)
    policy.load_state_dict(ck['policy'],strict=True)
    for name,value in ck.get('consts',{}).items():
        if name=='default_q':continue  # Explicit old-policy/new-environment joint-zero adapter.
        if not torch.allclose(getattr(env.consts,name),value.to(env.device),rtol=1e-5,atol=1e-6):
            raise ValueError('Calibration checkpoint physical constants disagree: '+name)
    old_q=ck['consts']['default_q'].to(env.device)
    offset=(old_q-env.consts.default_q)/env.consts.action_scale
    old_history=torch.zeros(env.num_envs,450,device=env.device)
    @torch.no_grad()
    def action(snapshot):
        reset=snapshot['frame_index'].reshape(-1)==0
        old_history[reset]=0
        state=snapshot['s'].clone();state[:,st.Q]+=env.consts.default_q-old_q
        previous=torch.where(reset[:,None],torch.zeros_like(snapshot['a_prev']),snapshot['a_prev']-offset)
        old_history.copy_(st.push_history(old_history,st.frame(state,previous,snapshot['command'],noise=True)))
        priv=st.privileged(state,previous,snapshot['command'],snapshot['xi'],snapshot['relief'])
        return policy(old_history,priv,1)+offset
    result=calibrate_as2_reward(env,action,out/'reward_calibration.json',reference_policy,branch='student')
    revision=subprocess.run(['git','-C',str(project),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    result['height_protocol']='h036_rescored_reference_policy_original_joint_zero'
    result['source'].update(reward_height_target_m=env.cfg.base_height_target,reference_joint_zero=old_q.tolist(),environment_joint_zero=env.consts.default_q.tolist(),joint_zero_adapter='absolute PD joint targets preserved; proprio/history rebuilt at original reference zero',height_capability_claim=False,reference_project=str(project),reference_source_commit=revision,
        source_network_sha256=hashlib.sha256((source/'networks.py').read_bytes()).hexdigest())
    (out/'reward_calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    from isaaclab.sim import SimulationContext
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--usd',required=True);parser.add_argument('--resume')
    parser.add_argument('--calibrate-rewards',action='store_true',help='Dedicated mature blind-AS2 reward measurement; builds no WM or depth cache')
    parser.add_argument('--reference-policy');parser.add_argument('--reference-project')
    add_launcher_args(parser);args=parser.parse_args()
    cfg=TrainCfg.from_record(json.loads(Path(args.config).read_text()))
    if args.calibrate_rewards and (not args.reference_policy or not args.reference_project):
        raise ValueError('Calibration requires --reference-policy and --reference-project')
    cfg.validate(production=not args.calibrate_rewards)
    if args.calibrate_rewards and cfg.variant!='simple':raise ValueError('Dedicated calibration configuration must be simple')
    root=Path(__file__).resolve().parents[1]
    for key in ('depth_package','reward_calibration'):
        value=getattr(cfg,key)
        if value and not Path(value).is_absolute():setattr(cfg,key,str(root/value))
    version=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    commit=version.stdout.strip() if version.returncode==0 else None
    dirty=subprocess.run(['git','-C',str(root),'status','--porcelain'],capture_output=True,text=True)
    if not args.calibrate_rewards and (commit is None or dirty.returncode or dirty.stdout.strip()):
        raise RuntimeError('Formal launch requires a clean versioned release')
    if cfg.variant!='simple' and not cfg.reward_calibration:
        raise ValueError('Four-task launch requires the measured mature-AS2 reward budget')
    import torch
    torch.manual_seed(cfg.seed);torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    if cfg.reward_calibration:
        from .task import configure_reward_budget
        configure_reward_budget(cfg.reward_calibration)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sim_cfg=physics_configuration(cfg)
    with launch_simulation(sim_cfg,args):
        from .physx_env import PhysXEnv
        sim=SimulationContext(sim_cfg);env=PhysXEnv(cfg,sim,args.usd)
        write_contract(env,cfg,args.usd,out,sim_cfg,commit)
        identity={'pid':os.getpid(),'started_unix':time.time(),'backend':type(env.robot).__module__,
            'mode':'calibration' if args.calibrate_rewards else 'formal','command':sys.argv,'code_commit':commit,
            'dirty_worktree':bool(dirty.stdout.strip()),'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
        (out/'process.json').write_text(json.dumps(identity,indent=2))
        if args.calibrate_rewards:
            result=calibrate_reference(env,args.reference_project,args.reference_policy,out)
            print(json.dumps({k:v for k,v in result.items() if k!='episodes'},allow_nan=False),flush=True)
            return
        from .learner import Learner
        learner=Learner(cfg,env,out)
        if args.resume:learner.load(args.resume,continue_run=True)
        def stop(signum,frame):learner.stop_requested=True
        learner.stop_requested=False;signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        learner.run()
        learner.export(out/'student_latest.pt')

if __name__=='__main__':main()
