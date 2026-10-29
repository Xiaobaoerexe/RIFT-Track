"""Isaac Lab3 / Isaac Sim PhysX adapter. No Newton task or learner imports.

All public learner tensors use leg-major ordering and XYZW quaternions. Motor
parameters follow the pinned source; raw actions are not tanh-squashed.
"""
import math
import torch
from . import state as st
from .task import RobotConsts,reward,fell
from .commands import OfficialCommands
from .terrain import relief,ground

def tensor(value):return value.torch if hasattr(value,'torch') else value


class PlaneTerrainLookup:
    names=('flat',)
    def query(self,xy):return torch.zeros(xy.shape[:-1],dtype=torch.long,device=xy.device)
    __call__=query

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
        from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg
        from .robots import profile,as2_motor_values
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass,instantiate
        from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg,PhysxArticulationCfg
        self.cfg,self.sim,self.device,self.num_envs=cfg,sim,cfg.device,cfg.num_envs
        self.field=None;self.terrain_lookup=PlaneTerrainLookup();self.terrain_names=('flat',)
        self.terrain_metadata={'names':['flat'],'geometry':'infinite_ground_plane'}
        self.evaluation_mode=False;self.command_override=None
        self.profile=profile(cfg.robot)
        inertia,kp,kd,effort,scale=(motor_values(cfg.device) if cfg.robot=='go2' else as2_motor_values(cfg.device))
        def mapping(v):return {n:float(x) for n,x in zip(st.JOINT_NAMES,v)}
        velocity={f'.*_{j}_joint':v for j,v in zip(('hip','thigh','calf'),self.profile.velocity_limits)}
        actuator_args=dict(joint_names_expr=['.*'],stiffness=mapping(kp),damping=mapping(kd),armature=mapping(inertia),friction=0.,actuator_effort_limit=mapping(effort),actuator_velocity_limit=velocity,joint_velocity_limit=velocity)
        actuator=(DCMotorCfg(saturation_effort=mapping(2*effort),**actuator_args) if cfg.robot=='go2' else IdealPDActuatorCfg(**actuator_args))
        robot_cfg=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',joint_ordering=st.JOINT_NAMES,
            spawn=su.UsdFileCfg(usd_path=usd_path,activate_contact_sensors=False,
                rigid_props=PhysxRigidBodyCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=1.),
                articulation_props=PhysxArticulationCfg(enabled_self_collisions=True,solver_position_iteration_count=4,solver_velocity_iteration_count=0)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,self.profile.initial_height),joint_pos={'.*L_hip_joint':self.profile.left_hip,'.*R_hip_joint':-self.profile.left_hip,'.*_thigh_joint':.9,'.*_calf_joint':-1.8},joint_vel={'.*':0.}),
            soft_joint_pos_limit_factor=.9,
            actuators={'legs':actuator})
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground=AssetBaseCfg(prim_path='/World/Ground',spawn=su.GroundPlaneCfg(physics_material=su.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)))
            robot=robot_cfg
        scene_cfg=SceneCfg(num_envs=cfg.num_envs,env_spacing=3.,replicate_physics=True)
        if cfg.stage=='terrain':
            from .terrain_scene import terrain_config
            scene_cfg.ground=None;scene_cfg.terrain=terrain_config(cfg.seed)
        self.scene=instantiate(scene_cfg);self.robot=self.scene['robot'];sim.reset()
        if 'isaaclab_physx' not in type(self.robot).__module__:raise RuntimeError(f'Wrong backend: {type(self.robot)}')
        if tuple(self.robot.joint_names)!=st.JOINT_NAMES:raise RuntimeError('Joint ordering mismatch')
        self.consts=RobotConsts(tensor(self.robot.data.default_joint_pos)[0].clone(),scale,kp,kd,effort,tensor(self.robot.data.soft_joint_pos_limits)[0].clone())
        self.body_ids={part:[list(self.robot.body_names).index(f'{leg}_{part}') for leg in st.LEGS] for part in ('hip','thigh','calf','foot') if all(f'{leg}_{part}' in self.robot.body_names for leg in st.LEGS)}
        if not all(k in self.body_ids for k in ('hip','thigh','calf')):raise RuntimeError(f'Incomplete limb body mapping: {self.robot.body_names}')
        if cfg.robot=='as2' and 'foot' not in self.body_ids:raise RuntimeError('As2 requires actual fixed foot bodies; Go2 foot offsets are invalid')
        self.origins=tensor(self.scene.env_origins).clone()
        if cfg.stage=='terrain':
            from .terrain_scene import field_from_mesh,TerrainPositionLookup
            self.field=field_from_mesh(self.scene.terrain,cfg.device)
            self.origins=tensor(self.scene.terrain.env_origins).clone()
            self.terrain_lookup=TerrainPositionLookup(self.scene.terrain.cfg.terrain_generator,cfg.device)
            self.terrain_names=self.terrain_lookup.names
            self.terrain_metadata=dict(self.terrain_lookup.metadata,collision_heightfield=self.field.metadata)
        n=cfg.num_envs;dev=cfg.device
        self.xi=torch.zeros(n,st.CONTEXT_DIM,device=dev);self.xi[:,0]=1.
        self.command_sampler=OfficialCommands(cfg.dt,cfg.episode_seconds,terrain_lookup=self.terrain_lookup)
        self.cmd_state=self.command_sampler.initial(self.origins[:,:2],self.origins[:,:2],iteration=0)
        self.command=self.cmd_state[:,3:6];self.timer=self.cmd_state[:,6:7]
        self.hist=torch.zeros(n,450,device=dev);self.last=torch.zeros(n,12,device=dev)
        self.steps=torch.zeros(n,dtype=torch.long,device=dev);self.episode=torch.zeros(n,1,dtype=torch.int32,device=dev)
        self.own_mask=torch.ones(n,dtype=torch.bool,device=dev)
        self.delay=torch.randint(3,(n,),device=dev);self.target_ring=self.consts.default_q.expand(3,n,12).clone();self.ring_ptr=0
        self.ids=torch.arange(n,device=dev);self.iteration=0
        self._terrain_max_levels=torch.full((len(self.terrain_names),),-1.,device=dev)
        self.terrain_level_statistics={}
        # Payload is a fixed rigid mass at the existing base COM, changed only at reset.
        if not 0.<=cfg.payload_max_kg<=2.:raise ValueError('Payload must be in0..2kg')
        self._mass_view=True
        self._base_id=list(self.robot.body_names).index('base')
        self._nominal_masses=tensor(self.robot.data.body_mass).clone()
        self.actual_base_mass_kg=self._nominal_masses[:,self._base_id].clone()
        self.reset(self.ids);self._snapshot=self._observe()

    def _observe(self,measured_crop=None):
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
                bp=body[:,self.body_ids['calf']];offset=torch.zeros_like(bp[:,:,:3]);offset[:,:,2]=-self.profile.calf_length
                qc=bp[:,:,3:7];pw=bp[:,:,:3]+quat_inverse(torch.cat((-qc[:,:,:3],qc[:,:,3:]),-1),offset)
            points.append(st.yaw_inverse(pw-pos[:,None,:],yaw[:,None]).flatten(1))
        s=torch.cat(((pos[:,2]-ground(pos[:,:2],self.field))[:,None],grav,vy,wb,
                     tensor(d.joint_pos)-self.consts.default_q,tensor(d.joint_vel),*points),-1)
        pose=torch.cat((pos[:,:2],yaw[:,None]),-1)
        if self.field is None:
            height_above_crop=pos[:,2];height_crop_valid=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
            terrain_id=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
            assigned_id=terrain_id;level=terrain_id
        else:
            height_above_crop,height_crop_valid=(self.field.raycaster.height_crop(pos,yaw)
                if measured_crop is None else measured_crop)
            terrain_id=self.terrain_lookup.query(pos)
            assigned_id=self.terrain_lookup.column_types[self.scene.terrain.terrain_types]
            level=self.scene.terrain.terrain_levels
        return {'s':s,'pose':pose,'a_prev':self.last.clone(),'obs':self.hist.clone(),'command':self.command.clone(),
                'timer':self.timer.clone(),'episode':self.episode.clone(),'relief':relief(pose,self.field),
                'cmd_state':self.cmd_state.clone(),'height_above_crop':height_above_crop,
                'height_crop_valid':height_crop_valid,'terrain_id':terrain_id,'assigned_terrain_id':assigned_id.clone(),
                'terrain_level':level.clone(),'episode_steps':self.steps.clone(),'xi':self.xi.clone(),
                'actual_base_mass_kg':self.actual_base_mass_kg.clone(),'payload_kg':self.xi[:,4].clone()}

    def snapshot(self):
        if self._snapshot is None:self._snapshot=self._observe()
        if self.command_override is not None:
            self._command_views()
            self._snapshot['command']=self.command.clone();self._snapshot['cmd_state']=self.cmd_state.clone()
        return self._snapshot

    def reset(self,ids):
        if not len(ids):return
        n=len(ids);dev=self.device
        if self.cfg.stage=='terrain':
            terrain=self.scene.terrain
            # Initial assignments remain the source max-init level5; only completed
            # training episodes can promote/demote, independent of training iteration.
            eligible=ids[self.steps[ids]>0]
            if len(eligible) and not self.evaluation_mode:
                from .terrain_scene import terrain_level_decisions
                gen=terrain.cfg.terrain_generator
                up,down=terrain_level_decisions(self.cmd_state[eligible,12],self.cmd_state[eligible,7:9],
                    self.command_sampler.resampling_time,self.command_sampler.zero_command_prob(self.iteration),
                    gen.size[0],gen.sub_terrain_border_width)
                terrain.update_env_origins(eligible,up,down)
            self.origins[ids]=tensor(terrain.env_origins)[ids]
            self._record_terrain_levels()
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
        mask=torch.zeros(self.num_envs,dtype=torch.bool,device=dev);mask[ids]=True
        all_positions=tensor(self.robot.data.root_link_pose_w)[:,:2].clone()
        all_positions[ids]=pose[:,:2]
        self.cmd_state=self.command_sampler.reset(self.cmd_state,mask,all_positions,self.origins[:,:2],self.iteration)
        self._command_views()
        if not self.evaluation_mode:self.set_payload(ids,torch.rand(n,device=dev)*self.cfg.payload_max_kg)
        self._snapshot=None

    def _command_views(self):
        if self.command_override is not None:
            if self.command_override.shape!=(self.num_envs,3):raise ValueError('command_override must be[N,3]')
            self.cmd_state[:,0:3]=self.command_override;self.cmd_state[:,3:6]=self.command_override
        self.command=self.cmd_state[:,3:6];self.timer=self.cmd_state[:,6:7]

    def set_payload(self,ids,payload):
        """Apply a point payload at existing COM only at an episode boundary."""
        if not len(ids):return
        if bool((self.steps[ids]!=0).any()):raise ValueError('Payload may change only after reset, never inside an episode')
        payload=torch.as_tensor(payload,device=self.device,dtype=self.xi.dtype).expand(len(ids))
        if not bool(torch.isfinite(payload).all()) or bool(((payload<0)|(payload>2.)).any()):
            raise ValueError('Payload must be finite and in0..2kg')
        masses=(self._nominal_masses[ids,self._base_id]+payload)[:,None]
        self.robot.set_masses_index(masses=masses,body_ids=[self._base_id],env_ids=ids)
        actual=tensor(self.robot.data.body_mass)[ids,self._base_id]
        if not torch.allclose(actual,masses[:,0],rtol=1e-6,atol=1e-6):raise RuntimeError('PhysX payload mass readback differs')
        self.actual_base_mass_kg[ids]=actual
        self.xi[ids,4]=actual-self._nominal_masses[ids,self._base_id]
        if getattr(self,'_snapshot',None) is not None:
            self._snapshot['xi']=self.xi.clone();self._snapshot['payload_kg']=self.xi[:,4].clone()
            self._snapshot['actual_base_mass_kg']=self.actual_base_mass_kg.clone()

    def _record_terrain_levels(self):
        terrain=self.scene.terrain;levels=terrain.terrain_levels.float()
        assigned=self.terrain_lookup.column_types[terrain.terrain_types]
        self.terrain_level_statistics={'mean_level':float(levels.mean())}
        for index,name in enumerate(self.terrain_names):
            values=levels[assigned==index]
            if len(values):self._terrain_max_levels[index]=torch.maximum(self._terrain_max_levels[index],values.max())
            self.terrain_level_statistics[name+'/mean_level']=float(values.mean()) if len(values) else -1.
            self.terrain_level_statistics[name+'/max_level_reached']=float(self._terrain_max_levels[index])

    def terrain_type_at(self,xy):
        return torch.zeros(xy.shape[:-1],dtype=torch.long,device=xy.device) if self.terrain_lookup is None else self.terrain_lookup.query(xy)

    def curriculum_state_dict(self):
        return {'terrain_levels':None if self.field is None else self.scene.terrain.terrain_levels.clone(),
                'max_levels_reached':self._terrain_max_levels.clone()}

    def load_curriculum_state_dict(self,state):
        self._terrain_max_levels.copy_(state['max_levels_reached'])
        if self.field is not None and state['terrain_levels'] is not None:
            terrain=self.scene.terrain;terrain.terrain_levels.copy_(state['terrain_levels'])
            terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
            self.origins[:]=terrain.env_origins
            self._record_terrain_levels()

    @torch.no_grad()
    def step(self,a):
        self._command_views()
        before=self.snapshot()['s'];target=self.consts.default_q+a*self.consts.action_scale
        for _ in range(round(self.cfg.dt/self.cfg.physics_dt)):
            self.target_ring[self.ring_ptr]=target
            delayed=self.target_ring[(self.ring_ptr-self.delay)%3,self.ids];self.ring_ptr=(self.ring_ptr+1)%3
            self.robot.set_joint_position_target_index(target=delayed)
            self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.cfg.physics_dt)
        self.steps+=1;after=self._observe();s=after['s']
        r,terms,_=reward(before,s,a,self.last,self.command,self.consts,self.cfg.stage,
                         iteration=self.iteration,height_above_crop=after['height_above_crop'],base_height_target=self.profile.reward_height)
        falls=fell(s);timeout=self.steps>=round(self.cfg.episode_seconds/self.cfg.dt);done=falls|timeout
        if not torch.isfinite(s).all():raise FloatingPointError('Nonfinite PhysX state')
        own=self.own_mask;vb=st.velocity_body(s)
        metrics={'height':float(s[own,0].mean()),'xy_rmse':float((vb[own,:2]-self.command[own,:2]).square().mean().sqrt()),
                 'yaw_rmse':float((s[own,9]-self.command[own,2]).square().mean().sqrt()),'reward':float(r[own].mean()),'falls':int((falls&own).sum()),'falls_all':int(falls.sum())}
        self.last=a.clone()
        if self.command_override is None:
            self.cmd_state=self.command_sampler.advance(self.cmd_state,after['pose'][:,:2],self.iteration)
        self._command_views()
        self.hist=st.push_history(self.hist,st.frame(s,a,self.command,noise=True))
        reset_ids=done.nonzero().flatten();self.reset(reset_ids)
        cached_crop=None if len(reset_ids) else (after['height_above_crop'],after['height_crop_valid'])
        self._snapshot=self._observe(measured_crop=cached_crop)
        # Return terminal state for labels; replay masks all reset transitions.
        return after,done,metrics
