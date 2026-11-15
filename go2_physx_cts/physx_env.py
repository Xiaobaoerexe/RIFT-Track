"""Isaac Sim/PhysX AS2 perception adapter with complete terminal transitions."""
import math
import torch
from . import state as st
from .task import RobotConsts,task_reward,initial_task_context,reward_budget,episode_time_reached,COMPLETED,ELAPSED,EPISODE_LIMIT,ROUTE_LENGTH,TASK_CONTEXT_DIM,POST_COMMAND,PREVIOUS_COMMAND,COMMAND_INITIALIZED
from .commands import OfficialCommands,TERRAIN_NAMES,TASK_TERRAINS,task_assignments,phase,speed_cap,GRID_BIN
from .terrain import relief,ground

def tensor(value):return value.torch if hasattr(value,'torch') else value
quat_inverse=st.quat_inverse


def materialize_metrics(metrics):
    """Transfer a log window once; preserve the variant's task reductions."""
    if not metrics or '_packed' not in metrics[0]:return metrics
    has_sources='source_counts' in metrics[0];tasks=int(metrics[0].get('_task_count',1))
    packed=[torch.cat((m['_packed'],m['source_counts'].double())) if has_sources else m['_packed'] for m in metrics]
    result=[]
    for row in torch.stack(packed).tolist():
        item=dict(zip(('height','xy_rmse','yaw_rmse','reward'),row[:4]))
        item.update(falls=int(row[4]),falls_all=int(row[5]),task_metrics={str(t):{'falls_per_env_min':row[6+t],'own_envs':int(row[6+tasks+t])} for t in range(tasks) if row[6+tasks+t]>0})
        if has_sources:item['source_counts']=[int(v) for v in row[6+2*tasks:]]
        result.append(item)
    return result


class PhysXEnv:
    def __init__(self,cfg,sim,usd_path):
        import isaaclab.sim as su
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from .robots import profile,as2_motor_values
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils import configclass,instantiate
        from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg,PhysxArticulationCfg
        from .terrain_scene import terrain_config,field_from_mesh,TerrainPositionLookup
        self.cfg,self.sim,self.device,self.num_envs=cfg,sim,cfg.device,cfg.num_envs
        self.evaluation_mode=False;self.calibration_flat=False;self.command_override=None;self.iteration=0;self._phase=-1;self._snapshot=None
        self.profile=profile(cfg.robot)
        if cfg.robot!="as2":raise ValueError("AS2 asset required")
        inertia,kp,kd,effort,scale=as2_motor_values(cfg.device)
        def mapping(v):return {n:float(x) for n,x in zip(st.JOINT_NAMES,v)}
        robot_cfg=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',joint_ordering=st.JOINT_NAMES,
            spawn=su.UsdFileCfg(usd_path=usd_path,activate_contact_sensors=True,
                rigid_props=PhysxRigidBodyCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=1.),
                articulation_props=PhysxArticulationCfg(enabled_self_collisions=True,solver_position_iteration_count=4,solver_velocity_iteration_count=0)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,self.profile.initial_height),joint_pos={'.*L_hip_joint':self.profile.left_hip,'.*R_hip_joint':-self.profile.left_hip,'.*_thigh_joint':st.DEFAULT_THIGH,'.*_calf_joint':st.DEFAULT_CALF},joint_vel={'.*':0.}),soft_joint_pos_limit_factor=.9,
            actuators={'legs':IdealPDActuatorCfg(joint_names_expr=['.*'],stiffness=mapping(kp),damping=mapping(kd),armature=mapping(inertia),friction=0.,
                actuator_effort_limit=mapping(effort),actuator_velocity_limit={'.*_hip_joint':24.,'.*_thigh_joint':24.,'.*_calf_joint':16.},joint_velocity_limit={'.*_hip_joint':24.,'.*_thigh_joint':24.,'.*_calf_joint':16.})})
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
        self.task_assignment_version=0
        self.task_ids=task_assignments(n,0,dev,cfg.variant);self.terrain_ids=torch.full((n,),7,dtype=torch.long,device=dev);self.terrain_levels=torch.zeros(n,dtype=torch.long,device=dev)
        self.pending_terrain=torch.full((n,),-1,dtype=torch.long,device=dev);self.pending_level=self.pending_terrain.clone()
        self.allocation_target=None
        self.arena_index=torch.full((n,),-1,dtype=torch.long,device=dev)
        self.xi=torch.zeros(n,st.CONTEXT_DIM,device=dev);self.xi[:,0]=1.
        self.hist=torch.zeros(n,450,device=dev);self.last=torch.zeros(n,12,device=dev);self.last2=torch.zeros_like(self.last)
        self.steps=torch.zeros(n,dtype=torch.long,device=dev);self.episode=torch.zeros(n,1,dtype=torch.int32,device=dev);self.own_mask=torch.ones(n,dtype=torch.bool,device=dev)
        self.delay=torch.randint(3,(n,),device=dev);self.target_ring=self.consts.default_q.expand(3,n,12).clone();self.ring_ptr=0
        self._terrain_max_levels=torch.full((len(self.terrain_names),),-1.,device=dev);self.terrain_level_statistics={}
        self.command_sampler=OfficialCommands(cfg.dt,cfg.episode_seconds,terrain_lookup=self.terrain_lookup,variant=cfg.variant,
            special_episode_seconds=8. if budget is None else budget["tasks"][3]["episode_seconds"])
        self.terrain_metadata["special_command_feasibility"]=self.command_sampler.special_contract
        self.cmd_state=self.command_sampler.initial(self.origins[:,:2],self.origins[:,:2],task_id=self.task_ids,terrain_id=self.terrain_ids)
        self.command=self.cmd_state[:,3:6];self.timer=self.cmd_state[:,6:7]
        self.task_context=torch.zeros(n,TASK_CONTEXT_DIM,device=dev)
        self.source=torch.zeros(n,dtype=torch.long,device=dev)
        self.group_sums=torch.zeros(4*2*len(self.terrain_names)*4,21,device=dev)
        self.fixed_reset=None
        self._segment_progress=torch.zeros(n,device=dev);self._segment_required=torch.zeros(n,device=dev);self._segment_steps=torch.zeros(n,device=dev);self._segment_yaw=torch.zeros(n,device=dev)
        self._episode_success=torch.zeros(n,dtype=torch.bool,device=dev);self._episode_all_own=torch.ones(n,dtype=torch.bool,device=dev)
        self._episode_grid_bin=torch.zeros(n,dtype=torch.long,device=dev)
        self._base_id=names.index('base');self._nominal_masses=tensor(self.robot.data.body_mass).clone();self.actual_base_mass_kg=self._nominal_masses[:,self._base_id].clone()
        self._phase=phase(0);self._assign_terrains();self.reset(self.ids,advance_curriculum=False);self._snapshot=self._observe()

    @property
    def task_id(self):return self.task_ids
    @task_id.setter
    def task_id(self,v):
        self.task_ids=v;self.task_assignment_version+=1
    def _assign_terrains(self):
        for task in range(4):
            ids=(self.task_ids==task).nonzero().flatten();n=len(ids)
            if not n:continue
            if task==0:
                if self.iteration<2500:self.terrain_ids[ids]=7
                else:
                    source=self.terrain_lookup.column_types
                    self.terrain_ids[ids]=source[torch.arange(n,device=self.device)%len(source)]
            else:
                choices=torch.tensor(TASK_TERRAINS[task],device=self.device)
                self.terrain_ids[ids]=choices[torch.arange(n,device=self.device)%len(choices)]
        self.terrain_levels=torch.randint(0,6,(self.num_envs,),device=self.device)
        self.terrain_levels[self.terrain_ids>=10]=0
        if self.iteration<2500:self.terrain_levels=self.ids%10
    def _ensure_phase(self):
        p=phase(self.iteration)
        if not self.evaluation_mode and p!=self._phase:
            old_tasks=self.task_ids.clone();old_terrain=self.terrain_ids.clone();old_levels=self.terrain_levels.clone()
            self.task_ids=task_assignments(self.num_envs,self.iteration,self.device,self.cfg.variant);self._phase=p;self._assign_terrains()
            self.task_assignment_version+=1
            retained=(old_tasks==self.task_ids)&(old_terrain==self.terrain_ids)
            self.terrain_levels[retained]=old_levels[retained]
            self.reset(self.ids,advance_curriculum=False)
    def _build_placement_tables(self):
        lookup=self.terrain_lookup;dev=self.device
        cols=lookup.column_types.cpu().tolist();pool=torch.zeros(13,len(cols),device=dev,dtype=torch.long);counts=torch.ones(13,device=dev,dtype=torch.long)
        for tid in range(10):
            values=[i for i,v in enumerate(cols) if v==tid];counts[tid]=len(values);pool[tid,:len(values)]=torch.tensor(values,device=dev)
        self._column_pool,self._column_count=pool,counts
        arenas=lookup.arenas;self._arena_origins=torch.tensor([a['origin'] for a in arenas],device=dev,dtype=torch.float32);self._arena_goals=torch.tensor([a['goal'] for a in arenas],device=dev,dtype=torch.float32);self._arena_lengths=torch.tensor([a['length'] for a in arenas],device=dev,dtype=torch.float32)
        self._arena_pool=torch.zeros(13,10,10,device=dev,dtype=torch.long);self._arena_count=torch.ones(13,10,device=dev,dtype=torch.long)
        for tid in (7,10,11,12):
            for level in range(10):
                values=[i for i,a in enumerate(arenas) if a['terrain_id']==tid and a['level']==level]
                self._arena_count[tid,level]=len(values);self._arena_pool[tid,level,:len(values)]=torch.tensor(values,device=dev)
    def _place_arenas(self,ids):
        terrain=self.scene.terrain;gen=terrain.cfg.terrain_generator;tid=self.terrain_ids[ids];levels=self.terrain_levels[ids]
        runway=(tid==7)&(self.task_ids[ids]==1)
        normal=(tid<10)&~runway&(not self.calibration_flat)
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
            'foot_velocity':st.yaw_inverse(tensor(d.body_link_vel_w)[:,self.body_ids['foot'],:3],st.yaw_from_pose(tensor(d.root_link_pose_w))[:,None]).clone(),'foot_positions':tensor(d.body_link_pose_w)[:,self.body_ids['foot'],:3].clone(),
            'joint_acceleration':tensor(d.joint_acc).clone(),'log_contact_force':torch.log1p(limb).clone()}
    def _observe(self,measured_crop=None):
        d=self.robot.data;pose=tensor(d.root_link_pose_w).clone();q=pose[:,3:];pos=pose[:,:3];yaw=st.yaw_from_pose(pose)
        vel=tensor(d.root_link_vel_w);grav=st.quat_inverse(q,pos.new_tensor([0.,0.,-1.]).expand_as(pos));body=tensor(d.body_link_pose_w)
        points=[st.yaw_inverse(body[:,self.body_ids[part],:3]-pos[:,None,:],yaw[:,None]).flatten(1) for part in ('thigh','calf','foot')]
        s=torch.cat(((pos[:,2]-ground(pos[:,:2],self.field))[:,None],grav,st.yaw_inverse(vel[:,:3],yaw),st.quat_inverse(q,vel[:,3:]),tensor(d.joint_pos)-self.consts.default_q,tensor(d.joint_vel),*points),-1)
        crop,valid=self.field.raycaster.height_crop(pos,yaw) if measured_crop is None else measured_crop
        telemetry=self._telemetry();scene_ids=self.field.scene_ids(pose)
        aux=torch.cat((telemetry['applied_torque'],telemetry['joint_acceleration'],telemetry['foot_velocity'][:,:,:2].flatten(1),telemetry['log_contact_force']),-1)
        return {'s':s,'pose':pose,'aux':aux,'a_prev':self.last.clone(),'a_prev2':self.last2.clone(),'obs':self.hist.clone(),'command':self.command.clone(),'timer':self.timer.clone(),'episode':self.episode.clone(),
            'relief':relief(pose,self.field),'cmd_state':self.cmd_state.clone(),'height_above_crop':crop,'height_crop_valid':valid,'task_id':self.task_ids.clone(),'task_ids':self.task_ids.clone(),
            'terrain_id':self.terrain_lookup.query(pos),'terrain_ids':self.terrain_lookup.query(pos),'assigned_terrain_id':self.terrain_ids.clone(),'terrain_level':self.terrain_levels.clone(),'episode_steps':self.steps.clone(),
            'xi':self.xi.clone(),'actual_base_mass_kg':self.actual_base_mass_kg.clone(),'payload_kg':self.xi[:,4].clone(),'task_context':self.task_context.clone(),'telemetry':telemetry,
            'contacts':telemetry['contact_prob'],'contact_prob':telemetry['contact_prob'],'foot_velocity':telemetry['foot_velocity'].flatten(1),'applied_torque':telemetry['applied_torque'],'scene_ids':scene_ids,
            'joint_acceleration':telemetry['joint_acceleration'],'log_contact_force':telemetry['log_contact_force'],
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
                tile_length=self.task_context[active,ROUTE_LENGTH].clamp_min(.1)
                up,down=terrain_level_decisions(self.cmd_state[active,12],self.cmd_state[active,7:9],self.command_sampler.resampling_time,
                    self.command_sampler.zero_command_prob(self.iteration),tile_length,0.,
                    success=self._episode_success[active],task_ids=self.task_ids[active] if self.cfg.variant!='simple' else None)
                self.terrain_levels[active]=(self.terrain_levels[active]+up.long()-down.long()).clamp(0,9)
            self._apply_terrain_transfers(ids)
        self._place_arenas(ids)
        pose=tensor(self.robot.data.default_root_pose)[ids].clone();pose[:,:3]+=self.origins[ids]
        normal=self.arena_index[ids]<0;pose[:,:2]+=(torch.rand(n,2,device=dev)-.5)*.3*normal[:,None]
        yaw=(torch.rand(n,device=dev)*2-1)*math.pi;directional=(self.arena_index[ids]>=0)|self.calibration_flat;yaw=torch.where(directional,torch.zeros_like(yaw),yaw)
        pose[:,3:]=st.rotvec_quat(torch.stack((torch.zeros_like(yaw),torch.zeros_like(yaw),yaw),-1));q=self.consts.default_q.expand(n,12).clone()
        if self.fixed_reset is not None:
            pose[:,:2]=self.origins[ids,:2]+self.fixed_reset["jitter"][ids]
            yaw=self.fixed_reset["yaw"][ids];pose[:,3:]=st.rotvec_quat(torch.stack((torch.zeros_like(yaw),torch.zeros_like(yaw),yaw),-1))
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids);self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=dev),env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=q,env_ids=ids);self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q),env_ids=ids);self.robot.reset(ids);self.sensor.reset(ids)
        self.hist[ids]=0;self.last[ids]=0;self.last2[ids]=0;self.steps[ids]=0;self.episode[ids]+=1;self._episode_success[ids]=False;self._episode_all_own[ids]=True
        self.target_ring[:,ids]=q;self.delay[ids]=torch.randint(3,(n,),device=dev);self.xi[ids,3]=self.delay[ids]*self.cfg.physics_dt
        mask=torch.zeros(self.num_envs,dtype=torch.bool,device=dev);mask[ids]=True;xy=tensor(self.robot.data.root_link_pose_w)[:,:2].clone();xy[ids]=pose[:,:2]
        self.cmd_state=self.command_sampler.reset(self.cmd_state,mask,xy,xy,self.iteration,task_id=self.task_ids,terrain_id=self.terrain_ids);self._command_views()
        self._episode_grid_bin[ids]=self.cmd_state[ids,GRID_BIN].long()
        ep=self._episode_limits(ids);length=torch.full((n,),8.,device=dev)
        custom=self.arena_index[ids]>=0
        if bool(custom.any()):length[custom]=self._arena_lengths[self.arena_index[ids[custom]]]
        fast=self.task_ids[ids]==1
        if bool(fast.any()):
            speed=torch.as_tensor(speed_cap(1,self.iteration),device=dev);available=(ep[fast]-1.).clamp_min(.5)
            ramp=torch.minimum(available,speed/2.);travel=ramp.square()+speed*(available-ramp).clamp_min(0.)
            length[fast]=((.30+.65*self.terrain_levels[ids[fast]]/9.)*travel/.6).clamp_min(12.)
        cmd=self.command[ids];cy=st.yaw_from_pose(pose);worlddir=torch.stack((cy.cos()*cmd[:,0]-cy.sin()*cmd[:,1],cy.sin()*cmd[:,0]+cy.cos()*cmd[:,1]),-1)
        worlddir=worlddir/worlddir.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        defaultdir=torch.stack((cy.cos(),cy.sin()),-1);worlddir=torch.where((cmd[:,:2].norm(dim=-1)<.05)[:,None],defaultdir,worlddir)
        ordinary=self.arena_index[ids]<0;goals=self.goals[ids].clone()
        goals[:,:2]=torch.where(ordinary[:,None],pose[:,:2]+worlddir*length[:,None]*.7,goals[:,:2])
        goals[:,2]=ground(goals[:,:2],self.field);self.goals[ids]=goals
        stable_seconds=torch.where(self.terrain_ids[ids]==10,.64,.32)
        self.task_context[ids]=initial_task_context(pose,goals,ep,.8,length,stable_seconds=stable_seconds)
        self.task_context[ids,PREVIOUS_COMMAND]=self.command[ids]
        self.task_context[ids,COMMAND_INITIALIZED]=1.
        post=(torch.rand(n,3,device=dev)*2-1)*torch.tensor([min(2.,speed_cap(0,self.iteration)),min(1.,speed_cap(0,self.iteration)),1.5],device=dev)
        self.task_context[ids,POST_COMMAND]=post
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
        if not bool(torch.isfinite(payload).all()) or bool(((payload<0)|(payload>2.)).any()):raise ValueError('Payload outside AS2 baseline0..2kg')
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
        self.task_assignment_version+=1
        self.reset(self.ids,advance_curriculum=False)
    def set_diagnostic_levels(self,level=0):
        self.terrain_levels.fill_(int(level))
        self._place_arenas(self.ids);self._snapshot=None

    def curriculum_state_dict(self):
        return {'terrain_levels':self.terrain_levels.clone(),'terrain_ids':self.terrain_ids.clone(),'task_ids':self.task_ids.clone(),'phase':self._phase,'max_levels_reached':self._terrain_max_levels.clone(),'grid':self.command_sampler.state_dict(),
                'allocation_target':self.allocation_target,**{k:getattr(self,k).clone() for k in ('pending_terrain','pending_level')}}
    def load_curriculum_state_dict(self,s):
        for name in ('terrain_levels','terrain_ids','task_ids'):getattr(self,name).copy_(s[name])
        self.task_assignment_version+=1
        self.allocation_target=s.get('allocation_target')
        for name in ('pending_terrain','pending_level'):
            if name in s:getattr(self,name).copy_(s[name])
            else:getattr(self,name).fill_(-1)
        self._phase=int(s['phase']);self._terrain_max_levels.copy_(s['max_levels_reached']);self.command_sampler.load_state_dict(s['grid']);self._place_arenas(self.ids)

    def allocation_status(self):
        return {'actual':torch.bincount(self.task_ids,minlength=4).tolist(),'target':self.allocation_target,'pending':int((self.pending_terrain>=0).sum())}

    def _apply_terrain_transfers(self,ids):
        # Tasks remain equal; AS2 allocator may only transfer terrain slots.
        moved=ids[self.pending_terrain[ids]>=0]
        if not len(moved):return
        self.terrain_ids[moved]=self.pending_terrain[moved];self.terrain_levels[moved]=self.pending_level[moved]
        self.pending_terrain[moved]=-1;self.pending_level[moved]=-1;self.task_assignment_version+=1

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
        special=self.task_ids==3
        score=torch.where(special,self._episode_success.float(),score)
        bins=torch.where(special,self._episode_grid_bin,self.cmd_state[:,GRID_BIN].long())
        terrain=torch.where(special,self.terrain_ids,before['terrain_ids'])
        self.command_sampler.grid.record(self.task_ids,terrain,bins,score,self.own_mask&ends&(~special|(done&self._episode_all_own)))
        self._segment_progress[ends]=0;self._segment_required[ends]=0;self._segment_steps[ends]=0;self._segment_yaw[ends]=0
        self.command_sampler.grid.update(self.iteration)
    @torch.no_grad()
    def step(self,a,*,defer_metrics=False):
        before=self.snapshot();self._command_views();target=self.consts.default_q+a*self.consts.action_scale
        for _ in range(round(self.cfg.dt/self.cfg.physics_dt)):
            self.target_ring[self.ring_ptr]=target;delayed=self.target_ring[(self.ring_ptr-self.delay)%3,self.ids];self.ring_ptr=(self.ring_ptr+1)%3
            self.robot.set_joint_position_target_index(target=delayed);self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.cfg.physics_dt)
        self.steps+=1;after=self._observe();s=after['s']
        r,terms,context,terminated,success=task_reward(before['s'],s,a,self.last,self.command,self.consts,task_id=self.task_ids,terrain_id=after['terrain_ids'],context=self.task_context,pose0=before['pose'],pose=after['pose'],telemetry=after['telemetry'],previous2=self.last2,history_valid=before['history_valid'],iteration=self.iteration,field=self.field,height_above_crop=after['height_above_crop'],variant=self.cfg.variant,base_height_target=self.cfg.base_height_target)
        self.task_context=context;self._episode_success|=success.bool();timeout=episode_time_reached(context[:,ELAPSED],context[:,EPISODE_LIMIT])
        done=terminated.bool()|timeout
        if not torch.isfinite(s).all():raise FloatingPointError('Nonfinite PhysX state')
        after.update(reward=r,positive_reward=terms['positive'],negative_reward=terms['negative'],terminated=terminated,timeout=timeout,success=success,task_context=context.clone(),done=done)
        self.last_transition=after
        if not self.evaluation_mode and self.cfg.variant!='simple':self._record_grid(before,after,done)
        own=self.own_mask;vb=st.velocity_body(s)
        scalars=(s[own,0].mean(),(vb[own,:2]-self.command[own,:2]).square().mean().sqrt(),
                 (s[own,9]-self.command[own,2]).square().mean().sqrt(),r[own].mean(),
                 (terminated.bool()&own).sum(),terminated.sum())
        counts=torch.bincount(self.task_ids[own],minlength=4).float();bad=torch.bincount(self.task_ids[own&terminated.bool()],minlength=4).float()
        rates=bad/counts.clamp_min(1)*60/self.cfg.dt
        metrics={'_packed':torch.cat((torch.stack([v.double() for v in scalars]),rates.double(),counts.double())).detach(),'_task_count':4}
        if not defer_metrics:metrics=materialize_metrics([metrics])[0]
        if not self.evaluation_mode:self._record_training_groups(after,terms,r,terminated)
        self.last2=self.last.clone();self.last=a.clone()
        if self.command_override is None:self.cmd_state=self.command_sampler.advance(self.cmd_state,after['pose'][:,:2],self.iteration,completed=context[:,COMPLETED],post_command=context[:,POST_COMMAND])
        self._command_views();self.hist=st.push_history(self.hist,st.frame(s,a,self.command,noise=True))
        after.update(command=self.command.clone(),cmd_state=self.cmd_state.clone(),timer=self.timer.clone(),obs=self.hist.clone(),a_prev=a.clone(),a_prev2=self.last2.clone(),
            event_reward=terms['event_reward'],continuous_reward=terms['continuous_reward'],bootstrap_mask=terms['bootstrap_mask'])
        self.reset(done.nonzero().flatten());self._snapshot=self._observe()
        return after,done,metrics

    def _record_training_groups(self,after,terms,reward,terminated):
        from .task import reward_weights
        kind=after['terrain_id'];s=after['s'];vb=st.velocity_body(s);speed=self.command[:,:2].norm(dim=-1)
        band=torch.bucketize(speed,speed.new_tensor([.5,1.5,2.5]));own=self.own_mask&(self.source>=0)&(self.source<2)
        valid=own&(kind>=0)&(kind<len(self.terrain_names))
        group=((self.task_ids*2+self.source.long())*len(self.terrain_names)+kind.clamp_min(0))*4+band
        weights=reward_weights(self.iteration,self.cfg.stage)
        base=[torch.ones_like(reward),reward,(vb[:,:2]-self.command[:,:2]).square().mean(-1),terminated.float(),(s[:,9]-self.command[:,2]).square(),s[:,0]]
        base.extend(terms.get(k,torch.zeros_like(reward))*weights[k]*self.cfg.dt for k in weights)
        values=torch.stack(base,-1)
        self.group_sums.index_add_(0,group[valid],values[valid])

    def flush_training_groups(self):
        from .task import WEIGHTS
        result={};width=len(self.terrain_names)*4
        for group,row in enumerate(self.group_sums.cpu().tolist()):
            count=row[0]
            if count<100:continue
            combined=group//width;task=combined//2;branch=combined%2;terrain=(group%width)//4;band=group%4
            # Keep branch first for the inherited window scheduler; include task
            # in the stratum so unrelated rewards cannot masquerade as drift.
            key=f'{branch}/{terrain}/{band}' if self.cfg.task_count==1 else f'{branch}/{task}:{terrain}/{band}'
            result[key]={'count':count,'reward':row[1]/count,'xy_rmse':(row[2]/count)**.5,
                'falls':row[3]/count*60/self.cfg.dt,'yaw_rmse':(row[4]/count)**.5,'height':row[5]/count,
                'reward_terms':{k:row[6+i]/count for i,k in enumerate(WEIGHTS)}}
        self.group_sums.zero_();return result

    def allocation_counts(self):
        eligible=self.task_ids==0
        return torch.bincount(self.terrain_ids[eligible],minlength=len(self.terrain_names)).tolist()

    def allocation_pending(self):return bool((self.pending_terrain>=0).any())

    def queue_allocation(self,target):
        """Source terrain allocator, never transfers environments between tasks."""
        if self.iteration<5000 or self.allocation_pending():return False
        current=self.allocation_counts()
        if len(target)!=len(current) or sum(target)!=sum(current) or any(type(n) is not int or n<0 for n in target):
            raise ValueError('Terrain allocation must conserve its eligible subgroup')
        if self.cfg.variant!='simple' and any(target[t] for t in range(10,len(target))):
            raise ValueError('Event task0 allocation cannot acquire special terrains')
        eligible=self.task_ids==0
        pool=[]
        for terrain,(a,b) in enumerate(zip(current,target)):
            if a>b:
                ids=(eligible&(self.terrain_ids==terrain)).nonzero().flatten()
                pool.extend(ids[torch.linspace(0,len(ids)-1,a-b,device=self.device).round().long()].tolist())
        offset=0
        for terrain,(a,b) in enumerate(zip(current,target)):
            if b>a:
                ids=torch.tensor(pool[offset:offset+b-a],device=self.device,dtype=torch.long);offset+=b-a
                population=self.terrain_levels[eligible&(self.terrain_ids==terrain)]
                self.pending_terrain[ids]=terrain;self.pending_level[ids]=population.median() if len(population) else 0
        if offset!=len(pool):raise RuntimeError('Allocation does not conserve environments')
        self.allocation_target=list(target);return bool(pool)
