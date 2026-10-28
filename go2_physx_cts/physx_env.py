"""Isaac Lab3 / Isaac Sim PhysX adapter. No Newton task or learner imports.

All public learner tensors use leg-major ordering and XYZW quaternions. Motor
parameters follow the pinned source; raw actions are not tanh-squashed.
"""
import math
import torch
from . import state as st
from .task import RobotConsts,reward,fell,advance_commands
from .terrain import relief,ground

def tensor(value):return value.torch if hasattr(value,'torch') else value

def quat_inverse(q,v):
    xyz,w=q[...,:3],q[...,3:4]
    return v-2*w*torch.cross(xyz,v,dim=-1)+2*torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1)

def motor_values(device):
    inertia=torch.tensor([.000111842*(6.33*(1.5 if j=='calf' else 1))**2 for _ in st.LEGS for j in ('hip','thigh','calf')],device=device)
    kp=inertia*(2*math.pi*10)**2;kd=4*inertia*(2*math.pi*10)
    effort=torch.tensor([23.7,23.7,45.43]*4,device=device)
    return inertia,kp,kd,effort,.25*effort/kp

class PhysXEnv:
    def __init__(self,cfg,sim,usd_path):
        import isaaclab.sim as su
        from isaaclab.assets import ArticulationCfg,AssetBaseCfg
        from isaaclab.actuators import DCMotorCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass,instantiate
        from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg,PhysxArticulationCfg
        self.cfg,self.sim,self.device,self.num_envs=cfg,sim,cfg.device,cfg.num_envs
        self.field=None
        inertia,kp,kd,effort,scale=motor_values(cfg.device)
        def mapping(v):return {n:float(x) for n,x in zip(st.JOINT_NAMES,v)}
        robot_cfg=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',joint_ordering=st.JOINT_NAMES,
            spawn=su.UsdFileCfg(usd_path=usd_path,activate_contact_sensors=False,
                rigid_props=PhysxRigidBodyCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=1.),
                articulation_props=PhysxArticulationCfg(enabled_self_collisions=True,solver_position_iteration_count=4,solver_velocity_iteration_count=0)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,.29),joint_pos={'.*L_hip_joint':.1,'.*R_hip_joint':-.1,'.*_thigh_joint':.9,'.*_calf_joint':-1.8},joint_vel={'.*':0.}),
            soft_joint_pos_limit_factor=.9,
            actuators={'legs':DCMotorCfg(joint_names_expr=['.*'],stiffness=mapping(kp),damping=mapping(kd),armature=mapping(inertia),friction=0.,
                saturation_effort=mapping(2*effort),actuator_effort_limit=mapping(effort),actuator_velocity_limit={'.*_hip_joint':30.1,'.*_thigh_joint':30.1,'.*_calf_joint':20.07},
                joint_velocity_limit={'.*_hip_joint':30.1,'.*_thigh_joint':30.1,'.*_calf_joint':20.07})})
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground=AssetBaseCfg(prim_path='/World/Ground',spawn=su.GroundPlaneCfg(physics_material=su.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)))
            robot=robot_cfg
        scene_cfg=SceneCfg(num_envs=cfg.num_envs,env_spacing=3.,replicate_physics=True)
        if cfg.stage=='terrain':
            from .terrain_scene import terrain_config
            scene_cfg.ground=None;scene_cfg.terrain=terrain_config()
        self.scene=instantiate(scene_cfg);self.robot=self.scene['robot'];sim.reset()
        if 'isaaclab_physx' not in type(self.robot).__module__:raise RuntimeError(f'Wrong backend: {type(self.robot)}')
        if tuple(self.robot.joint_names)!=st.JOINT_NAMES:raise RuntimeError('Joint ordering mismatch')
        self.consts=RobotConsts(tensor(self.robot.data.default_joint_pos)[0].clone(),scale,kp,kd,effort,tensor(self.robot.data.soft_joint_pos_limits)[0].clone())
        self.body_ids={part:[list(self.robot.body_names).index(f'{leg}_{part}') for leg in st.LEGS] for part in ('hip','thigh','calf','foot') if all(f'{leg}_{part}' in self.robot.body_names for leg in st.LEGS)}
        if not all(k in self.body_ids for k in ('hip','thigh','calf')):raise RuntimeError(f'Incomplete limb body mapping: {self.robot.body_names}')
        self.origins=tensor(self.scene.env_origins).clone()
        if cfg.stage=='terrain':
            from .terrain_scene import field_from_mesh
            self.field=field_from_mesh(self.scene.terrain,cfg.device)
            self.origins=tensor(self.scene.terrain.env_origins).clone()
        n=cfg.num_envs;dev=cfg.device
        self.xi=torch.zeros(n,st.CONTEXT_DIM,device=dev);self.xi[:,0]=1.
        self.command=torch.zeros(n,3,device=dev);self.timer=torch.zeros(n,1,device=dev)
        self.hist=torch.zeros(n,450,device=dev);self.last=torch.zeros(n,12,device=dev)
        self.steps=torch.zeros(n,dtype=torch.long,device=dev);self.episode=torch.zeros(n,1,dtype=torch.int32,device=dev)
        self.own_mask=torch.ones(n,dtype=torch.bool,device=dev)
        self.delay=torch.randint(3,(n,),device=dev);self.target_ring=self.consts.default_q.expand(3,n,12).clone();self.ring_ptr=0
        self.ids=torch.arange(n,device=dev);self.iteration=0
        # Payload is a fixed rigid mass at the existing base COM, changed only at reset.
        self._mass_view=None
        if cfg.payload_max_kg:
            self._mass_view=True
            self._base_id=list(self.robot.body_names).index('base')
            self._nominal_masses=tensor(self.robot.data.body_mass).clone()
        self.reset(self.ids);self._snapshot=self._observe()

    def _observe(self):
        d=self.robot.data;rp=tensor(d.root_link_pose_w);q=rp[:,3:7];pos=rp[:,:3]
        yaw=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1].square()+q[:,2].square()))
        vel=tensor(d.root_link_vel_w);g=torch.zeros_like(pos);g[:,2]=-1
        grav=quat_inverse(q,g);vy=st.yaw_inverse(vel[:,:3],yaw);wb=quat_inverse(q,vel[:,3:])
        body=tensor(d.body_link_pose_w);points=[]
        # Source points are hip joint, knee joint and foot. Link origins coincide
        # with hip/thigh/calf joints; foot may be a merged fixed link in PhysX.
        for part in ('thigh','calf','foot'):
            if part in self.body_ids:pw=body[:,self.body_ids[part],:3]
            else:
                bp=body[:,self.body_ids['calf']];offset=torch.zeros_like(bp[:,:,:3]);offset[:,:,2]=-.213
                qc=bp[:,:,3:7];pw=bp[:,:,:3]+quat_inverse(torch.cat((-qc[:,:,:3],qc[:,:,3:]),-1),offset)
            points.append(st.yaw_inverse(pw-pos[:,None,:],yaw[:,None]).flatten(1))
        s=torch.cat(((pos[:,2]-ground(pos[:,:2],self.field))[:,None],grav,vy,wb,
                     tensor(d.joint_pos)-self.consts.default_q,tensor(d.joint_vel),*points),-1)
        pose=torch.cat((pos[:,:2],yaw[:,None]),-1)
        return {'s':s,'pose':pose,'a_prev':self.last.clone(),'obs':self.hist.clone(),'command':self.command.clone(),
                'timer':self.timer.clone(),'episode':self.episode.clone(),'relief':relief(pose,self.field)}

    def snapshot(self):return self._snapshot

    def reset(self,ids):
        if not len(ids):return
        n=len(ids);dev=self.device
        if self.cfg.stage=='terrain':
            max_level=min(4,int(5*self.iteration/max(1,self.cfg.terrain_curriculum_iterations)))
            terrain=self.scene.terrain;levels=torch.randint(max_level+1,(n,),device=dev)
            terrain.terrain_levels[ids]=levels
            terrain.env_origins[ids]=terrain.terrain_origins[levels,terrain.terrain_types[ids]]
            self.origins[ids]=terrain.env_origins[ids]
        pose=tensor(self.robot.data.default_root_pose)[ids].clone();pose[:,:3]+=self.origins[ids]
        pose[:,:2]+=(torch.rand(n,2,device=dev)-.5)*.5
        yaw=(torch.rand(n,device=dev)*2-1)*math.pi;pose[:,3:]=0;pose[:,5]=torch.sin(yaw/2);pose[:,6]=torch.cos(yaw/2)
        pos=self.consts.default_q.expand(n,12).clone()
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=dev),env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=pos,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos),env_ids=ids)
        self.robot.reset(ids)
        self.hist[ids]=0;self.last[ids]=0;self.steps[ids]=0;self.episode[ids]+=1
        self.target_ring[:,ids]=pos;self.delay[ids]=torch.randint(3,(n,),device=dev)
        self.command[ids],self.timer[ids]=advance_commands(self.command[ids],torch.zeros(n,1,device=dev),self.cfg.command_max)
        if self._mass_view is not None:
            payload=torch.rand(n,device=dev)*self.cfg.payload_max_kg;self.xi[ids,4]=payload
            masses=(self._nominal_masses[ids,self._base_id]+payload)[:,None]
            self.robot.set_masses_index(masses=masses,body_ids=[self._base_id],env_ids=ids)

    @torch.no_grad()
    def step(self,a):
        before=self._snapshot['s'];target=self.consts.default_q+a*self.consts.action_scale
        for _ in range(round(self.cfg.dt/self.cfg.physics_dt)):
            self.target_ring[self.ring_ptr]=target
            delayed=self.target_ring[(self.ring_ptr-self.delay)%3,self.ids];self.ring_ptr=(self.ring_ptr+1)%3
            self.robot.set_joint_position_target_index(target=delayed)
            self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.cfg.physics_dt)
        self.steps+=1;after=self._observe();s=after['s']
        r,terms,_=reward(before,s,a,self.last,self.command,self.consts,self.cfg.stage)
        falls=fell(s);timeout=self.steps>=round(self.cfg.episode_seconds/self.cfg.dt);done=falls|timeout
        if not torch.isfinite(s).all():raise FloatingPointError('Nonfinite PhysX state')
        own=self.own_mask;vb=st.velocity_body(s)
        metrics={'height':float(s[own,0].mean()),'xy_rmse':float((vb[own,:2]-self.command[own,:2]).square().mean().sqrt()),
                 'yaw_rmse':float((s[own,9]-self.command[own,2]).square().mean().sqrt()),'reward':float(r[own].mean()),'falls':int((falls&own).sum()),'falls_all':int(falls.sum())}
        self.last=a.clone();self.command,self.timer=advance_commands(self.command,self.timer,self.cfg.command_max)
        self.hist=st.push_history(self.hist,st.frame(s,a,self.command,noise=True))
        self.reset(done.nonzero().flatten());self._snapshot=self._observe()
        # Return terminal state for labels; replay masks all reset transitions.
        return after,done,metrics
