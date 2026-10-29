"""Only supported launch entry: Isaac Sim plus PhysX. Old checkpoints rejected."""
import argparse,json,os,signal,sys,time,hashlib,subprocess
from pathlib import Path
from .config import TrainCfg
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')

def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    from isaaclab.app.sim_launcher import _make_physics_cfg
    from isaaclab.sim import SimulationCfg,SimulationContext,RigidBodyMaterialCfg
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--usd',required=True);parser.add_argument('--resume');parser.add_argument('--continue-run',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-iterations',type=int,default=8)
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--probe-steps',type=int,default=1000)
    add_launcher_args(parser);args=parser.parse_args()
    cfg=TrainCfg.from_record(json.loads(Path(args.config).read_text()));cfg.validate(production=not args.smoke)
    code_root=Path(__file__).resolve().parents[1]
    version=subprocess.run(['git','-C',str(code_root),'rev-parse','HEAD'],capture_output=True,text=True)
    code_commit=version.stdout.strip() if version.returncode==0 else None
    if not args.smoke:
        dirty=subprocess.run(['git','-C',str(code_root),'status','--porcelain'],capture_output=True,text=True)
        if code_commit is None or dirty.returncode or dirty.stdout.strip():raise RuntimeError('Formal launch requires a clean versioned release')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    import torch
    torch.manual_seed(cfg.seed);torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    physics=_make_physics_cfg('isaacsim_physx')
    # Exact pinned Go2RLLab PhysX preset; default patch buffers overflow with
    #4096 robots on the complete direct-mesh terrain during actual diagnostics.
    physics.gpu_max_rigid_patch_count=2*1024*1024
    physics.gpu_collision_stack_size=512*1024*1024
    physics.enable_external_forces_every_iteration=True
    sim_cfg=SimulationCfg(dt=cfg.physics_dt,device=cfg.device,render_interval=4,physics=physics,physics_material=RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
    with launch_simulation(sim_cfg,args):
        from .physx_env import PhysXEnv
        from .driver import MoeCtsDriver,FlatExpert
        from .learner import Learner
        sim=SimulationContext(sim_cfg);env=PhysXEnv(cfg,sim,args.usd)
        driver=MoeCtsDriver(env,cfg.driver_policy) if cfg.driver_frac else None
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        from .physx_env import tensor
        masses=tensor(env.robot.data.body_mass).sum(-1)
        contract={'physics_backend':type(env.robot).__module__,'usd_path':args.usd,'usd_sha256':hashlib.sha256(Path(args.usd).read_bytes()).hexdigest(),
                  'joint_names':list(env.robot.joint_names),'body_names':list(env.robot.body_names),'torch':torch.__version__,
                  'total_mass_range_kg':[float(masses.min()),float(masses.max())],'context_dim':5,'wm_state_dim':70,'wm_input_dim':176,'wm_output_dim':73,'teacher_input_dim':167,'critic_input_dim':169,
                  'student_input_dim':450,'shared_actor_input_dim':77,'policy_hz':50,'physics_hz':200,
                  'motor':{k:v.tolist() for k,v in vars(env.consts).items()},'payload_model':'rigid point mass at existing base COM, sampled only on reset','payload_max_kg':cfg.payload_max_kg}
        contract['curriculum_clock']='learner iterations; collect1; not PPO24-step equivalent'
        contract['robot']=cfg.robot
        contract['reward_height_target_m']=env.profile.reward_height
        contract['terrain']=None if env.field is None else {'allocation':env.terrain_lookup.metadata,'collision_height_map':env.field.metadata}
        contract['actual_base_mass_range_kg']=[float(env.actual_base_mass_kg.min()),float(env.actual_base_mass_kg.max())]
        contract['reward_scope']='baseline13 plus pinned body-height and vertical-velocity definitions/courses; not full official15-term reward'
        contract['physx_preset']={k:getattr(physics,k) for k in ('gpu_max_rigid_patch_count','gpu_collision_stack_size','enable_external_forces_every_iteration')}
        (out/'physics_contract.json').write_text(json.dumps(contract,indent=2))
        identity={'pid':os.getpid(),'started_unix':time.time(),'backend':type(env.robot).__module__,'mode':'smoke' if args.smoke else 'formal','command':sys.argv,'code_commit':code_commit,'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
        (out/'process.json').write_text(json.dumps(identity,indent=2))
        if args.probe_only:
            from .validation import physical_probe
            result=physical_probe(env,driver,args.probe_steps)
            (out/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
            return
        expert=FlatExpert(cfg.expert_policy,cfg.device) if cfg.expert_frac else None
        if args.smoke and cfg.stage=='terrain':
            import runpy
            audit=runpy.run_path(str(code_root/'tests/terrain_runtime_audit.py'))['audit_environment'](env)
            (out/'terrain_audit.json').write_text(json.dumps(audit,indent=2))
        learner=Learner(cfg,env,out,driver,expert)
        if args.resume:learner.load(args.resume,args.continue_run)
        def stop(signum,frame):
            learner.stop_requested=True
        learner.stop_requested=False
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        if args.smoke:
            from .validation import learner_smoke
            result=learner_smoke(learner,args.smoke_iterations)
            (out/'smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        else:
            learner.run();learner.export(out/'student_latest.pt')

if __name__=='__main__':main()
