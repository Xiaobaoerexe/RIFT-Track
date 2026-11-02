"""Isaac Sim/PhysX depth multitask training and measured reward calibration."""
import os
# Set before config/Isaac/torch imports can initialize CUDA. Validated on the
# pinned PyTorch runtime; explicit caller allocator settings remain respected.
if 'PYTORCH_ALLOC_CONF' not in os.environ and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
import argparse,hashlib,importlib.metadata,json,signal,subprocess,sys,time
from pathlib import Path
from dataclasses import asdict
from .config import TrainCfg
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')

def physics_configuration(cfg):
    from isaaclab.app.sim_launcher import _make_physics_cfg
    from isaaclab.sim import SimulationCfg,RigidBodyMaterialCfg
    physics=_make_physics_cfg('isaacsim_physx')
    physics.gpu_max_rigid_patch_count=2*1024*1024
    physics.gpu_collision_stack_size=512*1024*1024
    # Measured4096-env flat cold start requests74,162,569 aggregate pairs;
    # mixed-phase smoke requested34,745,924. Keep35% cold-start headroom.
    physics.gpu_found_lost_aggregate_pairs_capacity=96*1024*1024
    # Cold flat startup at4096envs requests3,020,271 broadphase pairs and
    # 2,101,248 total aggregate pairs, exceeding both2**21 defaults.
    physics.gpu_found_lost_pairs_capacity=2**23
    physics.gpu_total_aggregate_pairs_capacity=2**23
    physics.enable_external_forces_every_iteration=True
    return SimulationCfg(dt=cfg.physics_dt,device=cfg.device,render_interval=4,physics=physics,
        physics_material=RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))

def write_contract(env,cfg,usd,out,sim_cfg,code_commit):
    import torch
    from .physx_env import tensor
    from . import UPSTREAM_COMMIT
    masses=tensor(env.robot.data.body_mass).sum(-1)
    versions={}
    for name in ('isaacsim','isaaclab','torch','numpy','warp-lang','trimesh'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not a registered distribution'
    record={'schema':'depth_physics_contract_v1','physics_backend':type(env.robot).__module__,
        'usd_path':usd,'usd_sha256':hashlib.sha256(Path(usd).read_bytes()).hexdigest(),
        'code_commit':code_commit,'parent_training_commit':'fa3c776af150986776b5c513130c0e7a8b38b4c6',
        'upstream_supermimic':UPSTREAM_COMMIT,'upstream_go2rllab':'6c95d9feef4814c25eb6536e61bb38e1d1d718d4',
        'rapid_grid_commit':'f5143ef940e934849c00284e34caf164d6ce7b6e','versions':versions,
        'joint_names':list(env.robot.joint_names),'body_names':list(env.robot.body_names),
        'total_mass_range_kg':[float(masses.min()),float(masses.max())],
        'context_dim':5,'wm_state_dim':70,'full_pose_dim':7,'wm_pose_delta_dim':6,
        'teacher_map':[2,64,128],'wm_local_map':[2,41,49],'depth_image':[1,48,64],
        'student_history_dim':450,'student_hidden_dim':256,'task_critics':10,
        'actor_experts':cfg.actor_experts,'proprio_experts':cfg.proprio_experts,
        'policy_hz':1/cfg.dt,'physics_hz':1/cfg.physics_dt,'depth_hz':1/(cfg.dt*cfg.depth_period),
        'network_precision':'bf16_mixed' if cfg.mixed_precision else 'fp32',
        'cuda_allocator_settings':torch.cuda.memory._snapshot().get('allocator_settings',{}),
        'motor':{k:v.tolist() for k,v in vars(env.consts).items()},
        'payload_model':'rigid point mass at existing base COM, sampled only on reset','payload_max_kg':cfg.payload_max_kg,
        'curriculum_clock':'learner iterations, one real vector step per iteration',
        'terrain':env.terrain_metadata,'reward_calibration':cfg.reward_calibration,
        'physx_preset':{k:getattr(sim_cfg.physics,k) for k in ('gpu_max_rigid_patch_count','gpu_collision_stack_size','gpu_found_lost_aggregate_pairs_capacity','gpu_found_lost_pairs_capacity','gpu_total_aggregate_pairs_capacity','enable_external_forces_every_iteration')}}
    if cfg.reward_calibration:
        record['reward_calibration_sha256']=hashlib.sha256(Path(cfg.reward_calibration).read_bytes()).hexdigest()
    (out/'physics_contract.json').write_text(json.dumps(record,indent=2))
    return record

def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    from isaaclab.sim import SimulationContext
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--usd',required=True);parser.add_argument('--resume')
    parser.add_argument('--calibrate-rewards',action='store_true',help='Run complete official-policy episodes and save permanent reward calibration evidence')
    add_launcher_args(parser);args=parser.parse_args()
    cfg=TrainCfg.from_record(json.loads(Path(args.config).read_text()))
    cfg.validate(production=not args.calibrate_rewards)
    root=Path(__file__).resolve().parents[1]
    version=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    code_commit=version.stdout.strip() if version.returncode==0 else None
    if not args.calibrate_rewards:
        dirty=subprocess.run(['git','-C',str(root),'status','--porcelain'],capture_output=True,text=True)
        if code_commit is None or dirty.returncode or dirty.stdout.strip():raise RuntimeError('Formal launch requires a clean versioned release')
    import torch
    torch.manual_seed(cfg.seed);torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    if cfg.reward_calibration:
        from .task import configure_reward_budget
        configure_reward_budget(cfg.reward_calibration)
    sim_cfg=physics_configuration(cfg)
    with launch_simulation(sim_cfg,args):
        from .physx_env import PhysXEnv
        from .driver import MoeCtsDriver
        sim=SimulationContext(sim_cfg);env=PhysXEnv(cfg,sim,args.usd)
        driver=MoeCtsDriver(env,cfg.driver_policy) if cfg.driver_policy else None
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        write_contract(env,cfg,args.usd,out,sim_cfg,code_commit)
        identity={'pid':os.getpid(),'started_unix':time.time(),'backend':type(env.robot).__module__,
                  'mode':'calibration' if args.calibrate_rewards else 'formal','command':sys.argv,
                  'code_commit':code_commit,'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
        (out/'process.json').write_text(json.dumps(identity,indent=2))
        if args.calibrate_rewards:
            from .calibration import calibrate_official_reward
            result=calibrate_official_reward(env,driver,out/'reward_calibration.json',cfg.driver_policy)
            print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
            return
        from .robogauge_assets import export_task_maps
        export_task_maps(env.terrain_metadata,out/'robogauge_maps')
        from .learner import Learner
        learner=Learner(cfg,env,out,driver)
        if args.resume:learner.load(args.resume,continue_run=True)
        def stop(signum,frame):learner.stop_requested=True
        learner.stop_requested=False
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        learner.run();learner.export(out/'student_latest.pt')

if __name__=='__main__':main()
