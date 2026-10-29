"""Read actual initialized PhysX tensors and composed USD; no policy updates."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,'/home/abc/Documents/go2_supermimic_lab/physx_rebuild')
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.app.sim_launcher import _make_physics_cfg
from isaaclab.sim import SimulationCfg,SimulationContext,RigidBodyMaterialCfg
p=argparse.ArgumentParser();p.add_argument('--output',required=True);add_launcher_args(p);a=p.parse_args()
import torch
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.physx_env import PhysXEnv,tensor
torch.set_num_threads(4)
cfg=TrainCfg(num_envs=2,payload_max_kg=1.,compile=False)
sc=SimulationCfg(dt=.005,device='cuda:0',render_interval=4,physics=_make_physics_cfg('isaacsim_physx'),physics_material=RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
usd='/home/abc/data2/go2_supermimic_lab/assets/go2/usd/go2.usd'
def conv(x):
 if isinstance(x,(str,int,float,bool,type(None))):return x
 if isinstance(x,(list,tuple)):return [conv(v) for v in x]
 try:return tensor(x).detach().cpu().tolist()
 except Exception:
  try:return x.numpy().tolist() if hasattr(x,'numpy') else x.tolist()
  except Exception:return str(x)
with launch_simulation(sc,a):
 import isaaclab.sim as su
 sim=SimulationContext(sc);env=PhysXEnv(cfg,sim,usd)
 view=env.robot.root_view
 out={'backend':type(env.robot).__module__,'usd_sha256':hashlib.sha256(Path(usd).read_bytes()).hexdigest(),'joint_names':env.robot.joint_names,'body_names':env.robot.body_names,'consts':{k:conv(v) for k,v in vars(env.consts).items()},'actual':{},'unavailable':{},'view_getters':[k for k in dir(view) if k.startswith('get_')],'data_attributes':[k for k in dir(env.robot.data) if not k.startswith('_')],'sim_attributes':[k for k in dir(sim) if 'stage' in k or 'view' in k],'payload_kg':conv(env.xi[:,4])}
 for name in ('get_masses','get_coms','get_inertias','get_dof_armatures','get_dof_dampings','get_dof_friction_coefficients','get_dof_friction_properties','get_dof_max_velocities','get_dof_max_forces','get_dof_limits','get_material_properties','get_contact_offsets','get_rest_offsets','get_dof_stiffnesses'):
  try:out['actual'][name]=conv(getattr(view,name)())
  except Exception as e:out['unavailable'][name]=str(e)
 out['actuators']={}
 for name,act in env.robot.actuators.items():
  out['actuators'][name]={k:conv(getattr(act,k)) for k in ('stiffness','damping','effort_limit','velocity_limit','armature','friction','saturation_effort') if hasattr(act,k)}
 for name in ('body_mass','body_inertia','body_com_pose_b','default_joint_pos','joint_pos_limits','joint_vel_limits','joint_effort_limits','joint_friction_coeff','joint_armature','joint_damping','joint_stiffness'):
  try:out['actual']['data.'+name]=conv(getattr(env.robot.data,name))
  except Exception as e:out['unavailable']['data.'+name]=str(e)
 meta=view.get_metatype(0);out['view_metadata']={k:conv(getattr(meta,k)) for k in ('dof_names','link_names','root_joint_name') if hasattr(meta,k)}
 from pxr import Usd,UsdPhysics,UsdGeom,UsdShade,PhysxSchema
 stage=sim.stage
 out['stage_type']=str(type(stage));prims=[]
 for prim in stage.Traverse():
  path=str(prim.GetPath())
  if ('/env_0/' in path or '/Ground' in path or '/physics' in path.lower()) and (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.MaterialAPI) or prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
   attrs={x.GetName():conv(x.Get()) for x in prim.GetAttributes() if x.HasAuthoredValueOpinion() or x.GetName().startswith(('physics:','physx'))}
   rels={x.GetName():[str(t) for t in x.GetTargets()] for x in prim.GetRelationships()}
   if prim.HasAPI(UsdPhysics.MaterialAPI):
    mat=PhysxSchema.PhysxMaterialAPI(prim)
    attrs.update(friction_combine_mode=conv(mat.GetFrictionCombineModeAttr().Get()),restitution_combine_mode=conv(mat.GetRestitutionCombineModeAttr().Get()))
   prims.append({'path':path,'type':prim.GetTypeName(),'attributes':attrs,'relationships':rels})
 out['composed_physics_prims']=prims
 Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps({'output':a.output,'backend':out['backend'],'prim_count':len(prims)}))
