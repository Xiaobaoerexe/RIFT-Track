"""Bounded four-environment asset/load/step check; no model or optimizer updates."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.app.sim_launcher import _make_physics_cfg
from isaaclab.sim import SimulationCfg,SimulationContext
p=argparse.ArgumentParser();p.add_argument('--robot',required=True,choices=('go2','as2'));p.add_argument('--usd',required=True);p.add_argument('--output',required=True)
add_launcher_args(p);a=p.parse_args()
import torch
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.physx_env import PhysXEnv,tensor
torch.set_num_threads(2)
cfg=TrainCfg(robot=a.robot,num_envs=4,compile=False)
sc=SimulationCfg(dt=.005,device='cuda:0',physics=_make_physics_cfg('isaacsim_physx'))
with launch_simulation(sc,a):
    env=PhysXEnv(cfg,SimulationContext(sc),a.usd)
    nominal=env._nominal_masses.sum(-1)
    if a.robot=='as2':assert torch.allclose(nominal,torch.full_like(nominal,17.64),atol=1e-4)
    env.set_payload(env.ids,torch.tensor([0.,.5,1.,2.],device=cfg.device))
    start=env.actual_base_mass_kg.clone();falls=0
    for _ in range(50):
        before=env.episode.clone()
        snap,done,metrics=env.step(torch.zeros(4,12,device=cfg.device))
        assert torch.isfinite(snap['s']).all()
        same=(before==env.episode).flatten()
        assert torch.allclose(start[same],env.actual_base_mass_kg[same],atol=1e-6)
        start=env.actual_base_mass_kg.clone();falls+=metrics['falls_all']
    actual={}
    for name in ('body_mass','body_inertia','body_com_pose_b','joint_pos_limits','joint_vel_limits','joint_effort_limits','joint_armature'):
        try:actual[name]=tensor(getattr(env.robot.data,name)).cpu().tolist()
        except AttributeError:pass
    result=dict(robot=a.robot,backend=type(env.robot).__module__,nominal_mass_kg=nominal.cpu().tolist(),joint_names=env.robot.joint_names,body_names=env.robot.body_names,consts={k:v.cpu().tolist() for k,v in vars(env.consts).items()},actual=actual,control_steps=50,falls=falls,policy_updates=0,capability_claim=False)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('actual','consts')}))
