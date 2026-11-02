"""Isaac Sim/PhysX ten-task adapter with complete terminal transitions."""
import math
import torch
from . import state as st
from .task import RobotConsts,task_reward,initial_task_context,reward_budget,episode_time_reached,COMPLETED,ELAPSED,EPISODE_LIMIT,RECOVERY_DEADLINE,ROUTE_LENGTH
from .commands import OfficialCommands,TERRAIN_NAMES,TASK_TERRAINS,task_assignments,phase,speed_cap,GRID_BIN
from .terrain import relief,ground

def tensor(value):return value.torch if hasattr(value,'torch') else value
quat_inverse=st.quat_inverse

def motor_values(device):
    inertia=torch.tensor([.000111842*(6.33*(1.5 if j=='calf' else 1))**2 for _ in st.LEGS for j in ('hip','thigh','calf')],device=device)
    kp=inertia*(2*math.pi*10)**2;kd=4*inertia*(2*math.pi*10);effort=torch.tensor([23.7,23.7,45.43]*4,device=device)
    return inertia,kp,kd,effort,.25*effort/kp

def recovery_configuration(n,default_q,limits,device):
    """Varied feasible joint ranges and side/back/belly/front/rear orientations."""
    kind=torch.randint(6,(n,),device=device);roll=torch.zeros(n,device=device);pitch=torch.zeros_like(roll)
    roll=torch.where(kind==0,torch.full_like(roll,math.pi/2),roll);roll=torch.where(kind==1,torch.full_like(roll,-math.pi/2),roll);roll=torch.where(kind==2,torch.full_like(roll,math.pi),roll)
    pitch=torch.where(kind==4,torch.full_like(pitch,math.pi/3),pitch);pitch=torch.where(kind==5,torch.full_like(pitch,-math.pi/3),pitch)
    roll+=(torch.rand(n,device=device)-.5)*.35;pitch+=(torch.rand(n,device=device)-.5)*.3;yaw=(torch.rand(n,device=device)*2-1)*math.pi
    rotations=st.quat_mul(st.rotvec_quat(torch.stack((torch.zeros_like(yaw),torch.zeros_like(yaw),yaw),-1)),st.quat_mul(st.rotvec_quat(torch.stack((torch.zeros_like(pitch),pitch,torch.zeros_like(pitch)),-1)),st.rotvec_quat(torch.stack((roll,torch.zeros_like(roll),torch.zeros_like(roll)),-1))))
    q=default_q.expand(n,12).clone().reshape(n,4,3)
    q[:,:,0]+=(torch.rand(n,4,device=device)-.5)*1.1
    q[:,:,1]=torch.rand(n,4,device=device)*2.1-.1
    q[:,:,2]=-(.9+torch.rand(n,4,device=device)*1.7)
    q=q.flatten(1).clamp(limits[:,0]+.04,limits[:,1]-.04)
    return rotations,q,kind

def clearance_height(position_xy,quaternion,q,field):
    """Place sampled body/leg collision envelopes above local terrain at reset."""
    h,k,f=st.forward_kinematics(q);segments=[]
    # URDF body box corners, foot spheres, link envelopes and head proxy.
    corners=q.new_tensor([[x,y,z] for x in (-.1881,.1881) for y in (-.04675,.04675) for z in (-.057,.057)]).expand(len(q),-1,-1)
    segments.extend((corners,h,k,f))
    for alpha in (.25,.5,.75):segments.extend((h*(1-alpha)+k*alpha,k*(1-alpha)+f*alpha))
    head=q.new_tensor([[.24,0,.03]]).expand(len(q),-1,-1);segments.append(head)
    body=torch.cat(segments,1);relative=st.quat_apply(quaternion[:,None,:],body)
    xy=position_xy[:,None,:]+relative[...,:2];surface=ground(xy,field)
    radius=torch.full_like(surface,.035);radius[:,:8]=.012;radius[:,-1]=.07
    return (surface+radius-relative[...,2]).amax(-1)+.015

class PhysXEnv:
    def __init__(self,cfg,sim,usd_path):
        import isaaclab.sim as su
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import DCMotorCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils import configclass,instantiate
        from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg,PhysxArticulationCfg
        from .terrain_scene import terrain_config,field_from_mesh,TerrainPositionLookup
        self.cfg,self.sim,self.device,self.num_envs=cfg,sim,cfg.device,cfg.num_envs
        self.evaluation_mode=False;self.calibration_flat=False;self.command_override=None;self.iteration=0;self._phase=-1;self._snapshot=None
        inertia,kp,kd,effort,scale=motor_values(cfg.device)
        def mapping(v):return {n:float(x) for n,x in zip(st.JOINT_NAMES,v)}
        robot_cfg=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',joint_ordering=st.JOINT_NAMES,
            spawn=su.UsdFileCfg(usd_path=usd_path,activate_contact_sensors=True,
                rigid_props=PhysxRigidBodyCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=1.),
                articulation_props=PhysxArticulationCfg(enabled_self_collisions=True,solver_position_iteration_count=4,solver_velocity_iteration_count=0)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,.29),joint_pos={'.*L_hip_joint':.1,'.*R_hip_joint':-.1,'.*_thigh_joint':.9,'.*_calf_joint':-1.8},joint_vel={'.*':0.}),soft_joint_pos_limit_factor=.9,
            actuators={'legs':DCMotorCfg(joint_names_expr=['.*'],stiffness=mapping(kp),damping=mapping(kd),armature=mapping(inertia),friction=0.,
                saturation_effort=mapping(2*effort),actuator_effort_limit=mapping(effort),actuator_velocity_limit={'.*_hip_joint':30.1,'.*_thigh_joint':30.1,'.*_calf_joint':20.07},joint_velocity_limit={'.*_hip_joint':30.1,'.*_thigh_joint':30.1,'.*_calf_joint':20.07})})
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot=robot_cfg
            contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',update_period=0.,history_length=3,debug_vis=False)
        sc=SceneCfg(num_envs=cfg.num_envs,env_spacing=3.,replicate_physics=True)
        budget=reward_budget();runway_seconds=cfg.episode_seconds if budget is None else budget['tasks'][1]['episode_seconds']
        sc.terrain=terrain_config(cfg.seed,runway_seconds,None if budget is None else [row['episode_seconds'] for row in budget['tasks']])
        self.scene=instantiate(sc);self.robot=self.scene['robot'];sim.reset();self.sensor=self.scene['contacts']
        if 'isaaclab_physx' not in type(self.robot).__module__:raise RuntimeError(f'Wrong backend: {type(self.robot)}')
        if tuple(self.robot.joint_names)!=st.JOINT_NAMES:raise RuntimeError('Joint ordering mismatch')
        self.consts=RobotConsts(tensor(self.robot.data.default_joint_pos)[0].clone(),scale,kp,kd,effort,tensor(self.robot.data.soft_joint_pos_limits)[0].clone())
        names=list(self.robot.body_names);self.body_ids={part:[names.index(f'{leg}_{part}') for leg in st.LEGS] for part in ('hip','thigh','calf','foot') if all(f'{leg}_{part}' in names for leg in st.LEGS)}
        if not all(k in self.body_ids for k in ('hip','thigh','calf','foot')):raise RuntimeError(f'Contact/FK contract requires distinct feet and leg links, found {names}')
        sensor_names=list(self.sensor.body_names);self.contact_ids={part:[sensor_names.index(f'{leg}_{part}') for leg in st.LEGS] for part in ('thigh','calf','foot')}
        self.base_contact_id=sensor_names.index('base');self.head_contact_ids=[i for i,n in enumerate(sensor_names) if 'head' in n.lower()]
        self.contact_metadata={'sensor_bodies':sensor_names,'head_separate':bool(self.head_contact_ids),'merged_head_handling':'head channel unavailable/zero; base-contact conservative illegal-body terminal' if not self.head_contact_ids else 'distinct head force'}
        self.field=field_from_mesh(self.scene.terrain,cfg.device);self.terrain_lookup=TerrainPositionLookup(self.scene.terrain.cfg.terrain_generator,cfg.device,self.scene.terrain.arenas)
        self._build_placement_tables()
        self.terrain_names=self.terrain_lookup.names;self.terrain_metadata=dict(self.terrain_lookup.metadata,collision_heightfield=self.field.metadata,contacts=self.contact_metadata,background_boxes=self.scene.terrain.background_boxes)
        n=cfg.num_envs;dev=cfg.device;self.ids=torch.arange(n,device=dev)
        self.origins=torch.zeros(n,3,device=dev);self.goals=torch.zeros_like(self.origins);self.bounds=torch.zeros(n,4,device=dev)
        self.task_ids=task_assignments(n,0,dev);self.terrain_ids=torch.full((n,),7,dtype=torch.long,device=dev);self.terrain_levels=torch.zeros(n,dtype=torch.long,device=dev)
        self.arena_index=torch.full((n,),-1,dtype=torch.long,device=dev);self.recovery_kind=torch.full_like(self.arena_index,-1)
        self.xi=torch.zeros(n,st.CONTEXT_DIM,device=dev);self.xi[:,0]=1.
        self.hist=torch.zeros(n,450,device=dev);self.last=torch.zeros(n,12,device=dev);self.last2=torch.zeros_like(self.last)
        self.steps=torch.zeros(n,dtype=torch.long,device=dev);self.episode=torch.zeros(n,1,dtype=torch.int32,device=dev);self.own_mask=torch.ones(n,dtype=torch.bool,device=dev)
        self.delay=torch.randint(3,(n,),device=dev);self.target_ring=self.consts.default_q.expand(3,n,12).clone();self.ring_ptr=0
        self._terrain_max_levels=torch.full((len(self.terrain_names),),-1.,device=dev);self.terrain_level_statistics={}
        self.command_sampler=OfficialCommands(cfg.dt,cfg.episode_seconds,terrain_lookup=self.terrain_lookup)
        self.cmd_state=self.command_sampler.initial(self.origins[:,:2],self.origins[:,:2],task_id=self.task_ids,terrain_id=self.terrain_ids)
        self.command=self.cmd_state[:,3:6];self.timer=self.cmd_state[:,6:7]
        self.task_context=torch.zeros(n,21,device=dev)
        self._segment_progress=torch.zeros(n,device=dev);self._segment_required=torch.zeros(n,device=dev);self._segment_steps=torch.zeros(n,device=dev);self._segment_yaw=torch.zeros(n,device=dev)
        self._episode_success=torch.zeros(n,dtype=torch.bool,device=dev);self._episode_all_own=torch.ones(n,dtype=torch.bool,device=dev)
        self._base_id=names.index('base');self._nominal_masses=tensor(self.robot.data.body_mass).clone();self.actual_base_mass_kg=self._nominal_masses[:,self._base_id].clone()
        self._phase=phase(0);self._assign_terrains();self.reset(self.ids,advance_curriculum=False);self._snapshot=self._observe()

    @property
    def task_id(self):return self.task_ids
    @task_id.setter
    def task_id(self,v):self.task_ids=v
    def _assign_terrains(self):
        for t in range(10):
            ids=(self.task_ids==t).nonzero().flatten();n=len(ids)
            if not n:continue
            choices=TASK_TERRAINS[t]
            if t==0:
                if self.iteration<2500:self.terrain_ids[ids]=7
                else:
                    cuts=[round(n*.4),round(n*.6),round(n*.8),n];begin=0
                    for tid,end in zip(choices,cuts):self.terrain_ids[ids[begin:end]]=tid;begin=end
            elif t==6:
                flat=n//2;self.terrain_ids[ids[:flat]]=7
                other=torch.tensor(choices[1:],device=self.device);self.terrain_ids[ids[flat:]]=other[torch.arange(n-flat,device=self.device)%9]
            else:self.terrain_ids[ids]=torch.tensor(choices,device=self.device)[torch.arange(n,device=self.device)%len(choices)]
        self.terrain_levels.zero_();self.terrain_levels[self.task_ids==6]=5
    def _ensure_phase(self):
        p=phase(self.iteration)
        if not self.evaluation_mode and p!=self._phase:
            old_tasks=self.task_ids.clone();old_terrain=self.terrain_ids.clone();old_levels=self.terrain_levels.clone()
            self.task_ids=task_assignments(self.num_envs,self.iteration,self.device);self._phase=p;self._assign_terrains()
            retained=(old_tasks==self.task_ids)&(old_terrain==self.terrain_ids)&(self.task_ids!=6)
            self.terrain_levels[retained]=old_levels[retained]
            self.reset(self.ids,advance_curriculum=False)
    def _build_placement_tables(self):
        lookup=self.terrain_lookup;dev=self.device
        cols=lookup.column_types.cpu().tolist();pool=torch.zeros(14,len(cols),device=dev,dtype=torch.long);counts=torch.ones(14,device=dev,dtype=torch.long)
        for tid in range(10):
            values=[i for i,v in enumerate(cols) if v==tid];counts[tid]=len(values);pool[tid,:len(values)]=torch.tensor(values,device=dev)
        self._column_pool,self._column_count=pool,counts
        arenas=lookup.arenas;self._arena_origins=torch.tensor([a['origin'] for a in arenas],device=dev,dtype=torch.float32);self._arena_goals=torch.tensor([a['goal'] for a in arenas],device=dev,dtype=torch.float32);self._arena_lengths=torch.tensor([a['length'] for a in arenas],device=dev,dtype=torch.float32)
        self._arena_pool=torch.zeros(14,10,10,device=dev,dtype=torch.long);self._arena_count=torch.ones(14,10,device=dev,dtype=torch.long)
        for tid in (7,10,11,12,13):
            for level in range(10):
                values=[i for i,a in enumerate(arenas) if a['terrain_id']==tid and a['level']==level]
                self._arena_count[tid,level]=len(values);self._arena_pool[tid,level,:len(values)]=torch.tensor(values,device=dev)
    def _place_arenas(self,ids):
        terrain=self.scene.terrain;gen=terrain.cfg.terrain_generator;tid=self.terrain_ids[ids];levels=self.terrain_levels[ids]
        normal=(tid<10)&(self.task_ids[ids]!=1)&(not self.calibration_flat)
        cols=self._column_pool[tid,ids%self._column_count[tid]]
        original=tensor(terrain.terrain_origins)[levels,cols]
        center=self.terrain_lookup.minimum_xy+torch.stack((levels,cols),-1)*self.terrain_lookup.size+self.terrain_lookup.size/2
        original_bounds=torch.stack((center[:,0]-gen.size[0]/2,center[:,0]+gen.size[0]/2,center[:,1]-gen.size[1]/2,center[:,1]+gen.size[1]/2),-1)
        arena_levels=torch.full_like(levels,9) if self.calibration_flat else levels
        layouts=(torch.rand(len(ids),device=self.device)*self._arena_count[tid,arena_levels]).long()
        chosen=self._arena_pool[tid,arena_levels,layouts]
        custom=self._arena_origins[chosen];bounds=self.terrain_lookup.extra_bounds[chosen]
        if self.calibration_flat:custom=torch.stack(((bounds[:,0]+bounds[:,1])/2,(bounds[:,2]+bounds[:,3])/2,custom[:,2]),-1)
        self.origins[ids]=torch.where(normal[:,None],original,custom)
        self.goals[ids]=torch.where(normal[:,None],original+original.new_tensor([gen.size[0]*.6,0,0]),self._arena_goals[chosen])
        self.bounds[ids]=torch.where(normal[:,None],original_bounds,bounds)
        self.arena_index[ids]=torch.where(normal,torch.full_like(chosen,-1),chosen)
        terrain.terrain_levels[ids]=levels;terrain.terrain_types[ids]=cols;terrain.env_origins[ids]=self.origins[ids]

    def _telemetry(self):
        d=self.robot.data;forces=tensor(self.sensor.data.net_forces_w_history).norm(dim=-1).amax(dim=1)
        foot_force=forces[:,self.contact_ids['foot']];base=forces[:,self.base_contact_id];head=forces[:,self.head_contact_ids].amax(-1) if self.head_contact_ids else torch.zeros_like(base)
        limb=forces[:,self.contact_ids['thigh']+self.contact_ids['calf']]
        contact_force=torch.cat((foot_force,base[:,None],head[:,None],limb),-1)
        return {'contact_force':contact_force,'contact_prob':(contact_force>5.).float(),'applied_torque':tensor(d.applied_torque).clone(),
            'foot_velocity':tensor(d.body_link_vel_w)[:,self.body_ids['foot'],:3].clone(),'foot_positions':tensor(d.body_link_pose_w)[:,self.body_ids['foot'],:3].clone()}
    def _observe(self,measured_crop=None):
        d=self.robot.data;pose=tensor(d.root_link_pose_w).clone();q=pose[:,3:];pos=pose[:,:3];yaw=st.yaw_from_pose(pose)
        vel=tensor(d.root_link_vel_w);grav=st.quat_inverse(q,pos.new_tensor([0.,0.,-1.]).expand_as(pos));body=tensor(d.body_link_pose_w)
        points=[st.yaw_inverse(body[:,self.body_ids[part],:3]-pos[:,None,:],yaw[:,None]).flatten(1) for part in ('thigh','calf','foot')]
        s=torch.cat(((pos[:,2]-ground(pos[:,:2],self.field))[:,None],grav,st.yaw_inverse(vel[:,:3],yaw),st.quat_inverse(q,vel[:,3:]),tensor(d.joint_pos)-self.consts.default_q,tensor(d.joint_vel),*points),-1)
        crop,valid=self.field.raycaster.height_crop(pos,yaw) if measured_crop is None else measured_crop
        telemetry=self._telemetry();scene_ids=self.field.scene_ids(pose)
        return {'s':s,'pose':pose,'a_prev':self.last.clone(),'a_prev2':self.last2.clone(),'obs':self.hist.clone(),'command':self.command.clone(),'timer':self.timer.clone(),'episode':self.episode.clone(),
            'relief':relief(pose,self.field),'cmd_state':self.cmd_state.clone(),'height_above_crop':crop,'height_crop_valid':valid,'task_id':self.task_ids.clone(),'task_ids':self.task_ids.clone(),
            'terrain_id':self.terrain_lookup.query(pos),'terrain_ids':self.terrain_lookup.query(pos),'assigned_terrain_id':self.terrain_ids.clone(),'terrain_level':self.terrain_levels.clone(),'episode_steps':self.steps.clone(),
            'xi':self.xi.clone(),'actual_base_mass_kg':self.actual_base_mass_kg.clone(),'payload_kg':self.xi[:,4].clone(),'task_context':self.task_context.clone(),'telemetry':telemetry,
            'contacts':telemetry['contact_prob'],'contact_prob':telemetry['contact_prob'],'foot_velocity':telemetry['foot_velocity'].flatten(1),'applied_torque':telemetry['applied_torque'],'scene_ids':scene_ids,
            'frame_index':self.steps.clone(),'trajectory_ids':self.ids.clone(),'history_valid':torch.stack((self.steps>0,self.steps>1),-1)}
    def snapshot(self):
        self._ensure_phase()
        if self._snapshot is None:self._snapshot=self._observe()
        if self.command_override is not None:self._command_views();self._snapshot['command']=self.command.clone();self._snapshot['cmd_state']=self.cmd_state.clone()
        return self._snapshot
    def _command_views(self):
        if self.command_override is not None:self.cmd_state[:,0:3]=self.command_override;self.cmd_state[:,3:6]=self.command_override
        self.command=self.cmd_state[:,3:6];self.timer=self.cmd_state[:,6:7]
    def _episode_limits(self,ids):
        b=reward_budget()
        return torch.full((len(ids),),self.cfg.episode_seconds,device=self.device) if b is None else torch.tensor([t['episode_seconds'] for t in b['tasks']],device=self.device,dtype=torch.float32)[self.task_ids[ids]]
    def reset(self,ids,advance_curriculum=True):
        if not len(ids):return
        n=len(ids);dev=self.device
        if advance_curriculum and not self.evaluation_mode:
            from .terrain_scene import terrain_level_decisions
            active=ids[self.steps[ids]>0]
            if len(active):
                progress=self.cmd_state[active,12]/self.task_context[active,ROUTE_LENGTH].clamp_min(.1)
                up,down=terrain_level_decisions(progress,self._episode_success[active],self.task_ids[active])
                self.terrain_levels[active]=(self.terrain_levels[active]+up.long()-down.long()).clamp(0,9)
        self._place_arenas(ids)
        pose=tensor(self.robot.data.default_root_pose)[ids].clone();pose[:,:3]+=self.origins[ids]
        normal=self.arena_index[ids]<0;pose[:,:2]+=(torch.rand(n,2,device=dev)-.5)*.3*normal[:,None]
        yaw=(torch.rand(n,device=dev)*2-1)*math.pi;directional=(self.task_ids[ids]==1)|(self.task_ids[ids]>=5)|self.calibration_flat;yaw=torch.where(directional,torch.zeros_like(yaw),yaw)
        pose[:,3:]=st.rotvec_quat(torch.stack((torch.zeros_like(yaw),torch.zeros_like(yaw),yaw),-1));q=self.consts.default_q.expand(n,12).clone()
        rec=self.task_ids[ids]==6
        if bool(rec.any()):
            quat,rq,kind=recovery_configuration(int(rec.sum()),self.consts.default_q,self.consts.soft_limits,dev);pose[rec,3:]=quat;q[rec]=rq
            pose[rec,2]=clearance_height(pose[rec,:2],quat,rq,self.field);self.recovery_kind[ids[rec]]=kind
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids);self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=dev),env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=q,env_ids=ids);self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q),env_ids=ids);self.robot.reset(ids);self.sensor.reset(ids)
        self.hist[ids]=0;self.last[ids]=0;self.last2[ids]=0;self.steps[ids]=0;self.episode[ids]+=1;self._episode_success[ids]=False;self._episode_all_own[ids]=True
        self.target_ring[:,ids]=q;self.delay[ids]=torch.randint(3,(n,),device=dev);self.xi[ids,3]=self.delay[ids]*self.cfg.physics_dt
        mask=torch.zeros(self.num_envs,dtype=torch.bool,device=dev);mask[ids]=True;xy=tensor(self.robot.data.root_link_pose_w)[:,:2].clone();xy[ids]=pose[:,:2]
        self.cmd_state=self.command_sampler.reset(self.cmd_state,mask,xy,xy,self.iteration,task_id=self.task_ids,terrain_id=self.terrain_ids);self._command_views()
        ep=self._episode_limits(ids);length=torch.full((n,),8.,device=dev)
        custom=self.arena_index[ids]>=0
        if bool(custom.any()):length[custom]=self._arena_lengths[self.arena_index[ids[custom]]]
        fast=self.task_ids[ids]==1
        if bool(fast.any()):
            speed=self.command_sampler.grid.extent[1,0].clamp(max=speed_cap(1,self.iteration));available=(ep[fast]-1.).clamp_min(.5)
            ramp=torch.minimum(available,speed/2.);travel=ramp.square()+speed*(available-ramp).clamp_min(0.)
            length[fast]=((.30+.65*self.terrain_levels[ids[fast]]/9.)*travel/.6).clamp_min(12.)
        cmd=self.command[ids];cy=st.yaw_from_pose(pose);worlddir=torch.stack((cy.cos()*cmd[:,0]-cy.sin()*cmd[:,1],cy.sin()*cmd[:,0]+cy.cos()*cmd[:,1]),-1)
        worlddir=worlddir/worlddir.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        defaultdir=torch.stack((cy.cos(),cy.sin()),-1);worlddir=torch.where((cmd[:,:2].norm(dim=-1)<.05)[:,None],defaultdir,worlddir)
        ordinary=self.arena_index[ids]<0;goals=self.goals[ids].clone()
        goals[:,:2]=torch.where(ordinary[:,None],pose[:,:2]+worlddir*length[:,None]*.7,goals[:,:2])
        goals[:,2]=ground(goals[:,:2],self.field);self.goals[ids]=goals
        self.task_context[ids]=initial_task_context(pose,goals,ep,8.,.8,length)
        initial_state=torch.zeros(n,st.STATE_DIM,device=dev);initial_state[:,st.Q]=q-self.consts.default_q
        initial_state=st.reconcile_state(initial_state,pose,self.field,self.consts.default_q)
        self.hist[ids]=st.push_history(self.hist[ids],st.frame(initial_state,self.last[ids],self.command[ids],noise=True))
        self._segment_progress[ids]=0;self._segment_required[ids]=0;self._segment_steps[ids]=0;self._segment_yaw[ids]=0
        if not self.evaluation_mode:self.set_payload(ids,torch.rand(n,device=dev)*self.cfg.payload_max_kg)
        self._snapshot=None
        if self.iteration%50==0:self._record_terrain_levels()
    def set_payload(self,ids,payload):
        if not len(ids):return
        if bool((self.steps[ids]!=0).any()):raise ValueError('Payload changes only at episode reset')
        payload=torch.as_tensor(payload,device=self.device,dtype=self.xi.dtype).expand(len(ids))
        if not bool(torch.isfinite(payload).all()) or bool(((payload<0)|(payload>1.2)).any()):raise ValueError('Payload outside approved0..1.2kg')
        masses=(self._nominal_masses[ids,self._base_id]+payload)[:,None];self.robot.set_masses_index(masses=masses,body_ids=[self._base_id],env_ids=ids)
        actual=tensor(self.robot.data.body_mass)[ids,self._base_id]
        if not torch.allclose(actual,masses[:,0],rtol=1e-6,atol=1e-6):raise RuntimeError('PhysX mass readback mismatch')
        self.actual_base_mass_kg[ids]=actual;self.xi[ids,4]=actual-self._nominal_masses[ids,self._base_id];self._snapshot=None
    def _record_terrain_levels(self):
        self.terrain_level_statistics={'mean_level':float(self.terrain_levels.float().mean())}
        for t,name in enumerate(self.terrain_names):
            v=self.terrain_levels[self.terrain_ids==t]
            if len(v):self._terrain_max_levels[t]=torch.maximum(self._terrain_max_levels[t],v.max());self.terrain_level_statistics[name+'/mean_level']=float(v.float().mean());self.terrain_level_statistics[name+'/max_level_reached']=float(self._terrain_max_levels[t])
    def terrain_type_at(self,xy):return self.terrain_lookup.query(xy)
    def validate_kinematics(self,tolerance=.003):
        snap=self._observe();projected=st.reconcile_state(snap['s'],snap['pose'],self.field,self.consts.default_q)
        error=(projected[:,st.HIPS.start:st.FEET.stop]-snap['s'][:,st.HIPS.start:st.FEET.stop]).abs().amax()
        if float(error)>tolerance:raise RuntimeError(f'Loaded USD/FK disagreement {float(error):.6f}m')
        return {'maximum_link_position_error_m':float(error),'tolerance_m':tolerance}

    def place_calibration_flat(self):
        self.evaluation_mode=True;self.calibration_flat=True
        self.task_ids.zero_();self.terrain_ids.fill_(7);self.terrain_levels.zero_()
        self.reset(self.ids,advance_curriculum=False)
    prepare_flat_calibration=place_calibration_flat
    def set_diagnostic_levels(self,level=0):
        self.terrain_levels.fill_(int(level));self.terrain_levels[self.task_ids==6]=5
        self._place_arenas(self.ids);self._snapshot=None

    def curriculum_state_dict(self):
        return {'terrain_levels':self.terrain_levels.clone(),'terrain_ids':self.terrain_ids.clone(),'task_ids':self.task_ids.clone(),'phase':self._phase,'max_levels_reached':self._terrain_max_levels.clone(),'grid':self.command_sampler.state_dict()}
    def load_curriculum_state_dict(self,s):
        for name in ('terrain_levels','terrain_ids','task_ids'):getattr(self,name).copy_(s[name])
        self._phase=int(s['phase']);self._terrain_max_levels.copy_(s['max_levels_reached']);self.command_sampler.load_state_dict(s['grid']);self._place_arenas(self.ids)
    def _record_grid(self,before,after,done):
        command=self.command;yaw=st.yaw_from_pose(before['pose']);body_delta=st.yaw_inverse(after['pose'][:,:3]-before['pose'][:,:3],yaw)[:,:2]
        speed=command[:,:2].norm(dim=-1);direction=command[:,:2]/speed[:,None].clamp_min(1e-6)
        own=self.own_mask;self._episode_all_own&=own
        self._segment_progress+=torch.where(own,(body_delta*direction).sum(-1),0.);self._segment_required+=torch.where(own,speed*self.cfg.dt,0.);self._segment_steps+=own
        wz=after['s'][:,9];self._segment_yaw+=torch.where(own,(1-(wz-command[:,2]).abs()/(command[:,2].abs()+.2)).clamp(0,1),0.)
        ends=done|(self.timer[:,0]<=self.cfg.dt)|(before['terrain_ids']!=after['terrain_ids'])
        score=(self._segment_progress/self._segment_required.clamp_min(.05)).clamp(0,1)
        nearzero=self._segment_required<.05
        stationary=(1-st.velocity_body(after['s'])[:,:2].norm(dim=-1)/.2).clamp(0,1)*(-after['s'][:,3]).clamp(0,1)
        score=torch.where(nearzero,torch.where(command[:,2].abs()>.05,self._segment_yaw/self._segment_steps.clamp_min(1),stationary),score)
        special=(self.task_ids==5)|(self.task_ids==8)|(self.task_ids==9)
        score=torch.where(special,self._episode_success.float(),score)
        self.command_sampler.grid.record(self.task_ids,before['terrain_ids'],self.cmd_state[:,GRID_BIN].long(),score,self.own_mask&ends&(~special|(done&self._episode_all_own)))
        self._segment_progress[ends]=0;self._segment_required[ends]=0;self._segment_steps[ends]=0;self._segment_yaw[ends]=0
        self.command_sampler.grid.update(self.iteration)
    @torch.no_grad()
    def step(self,a):
        before=self.snapshot();self._command_views();target=self.consts.default_q+a*self.consts.action_scale
        for _ in range(round(self.cfg.dt/self.cfg.physics_dt)):
            self.target_ring[self.ring_ptr]=target;delayed=self.target_ring[(self.ring_ptr-self.delay)%3,self.ids];self.ring_ptr=(self.ring_ptr+1)%3
            self.robot.set_joint_position_target_index(target=delayed);self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.cfg.physics_dt)
        self.steps+=1;after=self._observe();s=after['s'];after['telemetry']['joint_acc']=(s[:,st.QD]-before['s'][:,st.QD])/self.cfg.dt
        r,terms,context,terminated,success=task_reward(before['s'],s,a,self.last,self.command,self.consts,task_id=self.task_ids,terrain_id=after['terrain_ids'],context=self.task_context,pose0=before['pose'],pose=after['pose'],telemetry=after['telemetry'],previous2=self.last2,history_valid=before['history_valid'],iteration=self.iteration,field=self.field,height_above_crop=after['height_above_crop'])
        self.task_context=context;self._episode_success|=success.bool();timeout=episode_time_reached(context[:,ELAPSED],context[:,EPISODE_LIMIT])
        done=terminated.bool()|timeout
        if not torch.isfinite(s).all():raise FloatingPointError('Nonfinite PhysX state')
        after.update(reward=r,positive_reward=terms['positive'],negative_reward=terms['negative'],terminated=terminated,timeout=timeout,success=success,task_context=context.clone(),done=done)
        self.last_transition=after
        if not self.evaluation_mode:self._record_grid(before,after,done)
        own=self.own_mask;vb=st.velocity_body(s);metrics={'height':float(s[own,0].mean()),'xy_rmse':float((vb[own,:2]-self.command[own,:2]).square().mean().sqrt()),'yaw_rmse':float((s[own,9]-self.command[own,2]).square().mean().sqrt()),'reward':float(r[own].mean()),'falls':int((terminated.bool()&own).sum()),'falls_all':int(terminated.sum())}
        counts=torch.bincount(self.task_ids[own],minlength=10).float();bad=torch.bincount(self.task_ids[own&terminated.bool()],minlength=10).float()
        task_rates=(bad/counts.clamp_min(1)*60/self.cfg.dt).tolist();task_counts=counts.tolist()
        metrics['task_metrics']={str(t):{'falls_per_env_min':task_rates[t],'own_envs':int(task_counts[t])} for t in range(10) if task_counts[t]>0}
        self.last2=self.last.clone();self.last=a.clone()
        if self.command_override is None:self.cmd_state=self.command_sampler.advance(self.cmd_state,after['pose'][:,:2],self.iteration)
        self._command_views();self.hist=st.push_history(self.hist,st.frame(s,a,self.command,noise=True))
        self.reset(done.nonzero().flatten());self._snapshot=self._observe()
        return after,done,metrics
