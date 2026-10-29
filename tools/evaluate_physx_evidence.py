"""Frozen-policy branch evaluation in the original PhysX environment; no learner."""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path

CODE=Path('/home/abc/Documents/go2_supermimic_lab/physx_rebuild')
sys.path.insert(0,str(CODE))

def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    from isaaclab.app.sim_launcher import _make_physics_cfg
    from isaaclab.sim import SimulationCfg,SimulationContext,RigidBodyMaterialCfg
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--usd',required=True);p.add_argument('--iterations',default='5000,10000,40000')
    add_launcher_args(p);args=p.parse_args()
    import torch
    import numpy as np
    from go2_physx_cts.config import TrainCfg
    from go2_physx_cts import state as st
    from go2_physx_cts.networks import Policy
    from go2_physx_cts.physx_env import PhysXEnv,tensor,quat_inverse
    from go2_physx_cts.task import fell
    torch.manual_seed(17);torch.set_num_threads(8);torch.set_grad_enabled(False)
    commands=[(0,0,0),(.5,0,0),(-.3,0,0),(0,.3,0),(0,0,.8),(1.,0,0),(1.5,0,0),(2.,0,0)]
    cases=[{'branch':'teacher' if branch==0 else 'student','command':cmd,'payload_kg':payload,'in_training_range':ci<5}
           for branch in (0,1) for ci,cmd in enumerate(commands) for payload in (0.,1.)]
    cfg=TrainCfg.from_record(json.loads((Path(args.run)/'flat/config.json').read_text()))
    cfg.num_envs=len(cases);cfg.compile=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sim_cfg=SimulationCfg(dt=cfg.physics_dt,device=cfg.device,render_interval=4,physics=_make_physics_cfg('isaacsim_physx'),physics_material=RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
    with launch_simulation(sim_cfg,args):
        sim=SimulationContext(sim_cfg);env=PhysXEnv(cfg,sim,args.usd)
        dev=cfg.device;n=cfg.num_envs
        # Spatially separate test trajectories; policy state is origin-independent.
        ids=torch.arange(n,device=dev);origins=torch.zeros(n,3,device=dev)
        origins[:,0]=(ids%6-2.5)*10;origins[:,1]=(ids//6-2.5)*10
        command=torch.tensor([c['command'] for c in cases],device=dev,dtype=torch.float32)
        payload=torch.tensor([c['payload_kg'] for c in cases],device=dev)
        results=[]
        for it in map(int,args.iterations.split(',')):
            checkpoint=Path(args.run)/'flat'/f'model_{it:08d}.pt'
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ck=torch.load(checkpoint,map_location=dev,weights_only=False)
            policy=Policy(tuple(ck['cfg']['actor_hidden'])).to(dev).eval();policy.load_state_dict(ck['policy']);policy.requires_grad_(False)
            for key,val in ck['consts'].items():
                assert torch.allclose(getattr(env.consts,key),val.to(dev),atol=1e-6),(key,'physics constant mismatch')
            env.robot.reset(ids)
            pose=tensor(env.robot.data.default_root_pose).clone();pose[:,:3]=origins;pose[:,2]=.29
            pose[:,3:]=0;pose[:,6]=1
            q0=env.consts.default_q.expand(n,12).clone()
            env.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
            env.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=dev),env_ids=ids)
            env.robot.write_joint_position_to_sim_index(position=q0,env_ids=ids)
            env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0),env_ids=ids)
            env.robot.set_masses_index(masses=(env._nominal_masses[:,env._base_id]+payload)[:,None],body_ids=[env._base_id],env_ids=ids)
            env.xi[:,4]=payload;env.command=command.clone();env.last.zero_();env.hist.zero_()
            env.scene.write_data_to_sim();env.scene.update(0.)
            hist=torch.zeros(n,450,device=dev);last=torch.zeros(n,12,device=dev)
            ring=q0.repeat(3,1,1);rp=0
            live=torch.ones(n,dtype=torch.bool,device=dev);last_step=torch.full((n,),500,device=dev)
            accum=torch.zeros(n,9,device=dev);count=torch.zeros(n,device=dev);traces=[];first_inputs=None
            slip_sum=torch.zeros(n,device=dev);slip_count=torch.zeros(n,device=dev);slip_values=[[] for _ in range(n)];poses=[];joints=[];foot_speeds=[];contact_masks=[]
            started=time.monotonic()
            for step in range(500):
                snap=env._observe();s=snap['s']
                hist=st.push_history(hist,st.frame(s,last,command,noise=False))
                priv=st.privileged(s,last,command,env.xi,snap['relief'])
                at=policy(hist,priv,0);ast=policy(hist,priv,1)
                a=torch.cat((at[:n//2],ast[n//2:]))
                zt=policy.teacher_encoder(policy.teacher_norm(priv));zs=policy.student_encoder(policy.history_norm(hist))[0]
                vb=st.velocity_body(s);wb=s[:,st.W_B]
                foot_ids=env.body_ids['foot'];bp=tensor(env.robot.data.body_link_pose_w)[:,foot_ids];bv=tensor(env.robot.data.body_link_vel_w)[:,foot_ids]
                offset=torch.zeros_like(bp[:,:,:3]);offset[:,:,0]=-.002
                offset=quat_inverse(torch.cat((-bp[:,:,3:6],bp[:,:,6:7]),-1),offset)
                center=bp[:,:,:3]+offset;lever=offset.clone();lever[:,:,2]-=.022
                tangential=bv[:,:,:3]+torch.cross(bv[:,:,3:6],lever,dim=-1)
                speed=torch.linalg.vector_norm(tangential[:,:,:2],dim=-1);contact=(center[:,:,2]<=.024)&live[:,None]
                if step>=50:
                    slip_sum+=(speed*contact).sum(-1);slip_count+=contact.sum(-1)
                    for ni in range(n):slip_values[ni].extend(speed[ni,contact[ni]].cpu().tolist())
                pose_frame=tensor(env.robot.data.root_link_pose_w).clone();pose_frame[:,:3]-=origins
                poses.append(pose_frame.cpu().numpy());joints.append(tensor(env.robot.data.joint_pos).cpu().numpy().copy());foot_speeds.append(speed.cpu().numpy());contact_masks.append(contact.cpu().numpy())
                if first_inputs is None:first_inputs={'state':s.cpu().tolist(),'observation':hist.cpu().tolist(),'action':a.cpu().tolist()}
                if step>=50:
                    values=torch.stack(((vb[:,:2]-command[:,:2]).square().mean(-1),(wb[:,2]-command[:,2]).square(),s[:,0],vb[:,0],vb[:,1],wb[:,2],(at-ast).square().mean(-1),(zt-zs).square().mean(-1),a.abs().amax(-1)),-1)
                    accum+=values*live[:,None];count+=live
                if step%5==0:traces.append({'step':step,'live':live.cpu().tolist(),'vx':vb[:,0].cpu().tolist(),'vy':vb[:,1].cpu().tolist(),'wz':wb[:,2].cpu().tolist(),'height':s[:,0].cpu().tolist()})
                target=env.consts.default_q+a*env.consts.action_scale
                for _ in range(4):
                    ring[rp]=target;delayed=ring[(rp-1)%3];rp=(rp+1)%3
                    env.robot.set_joint_position_target_index(target=delayed)
                    env.scene.write_data_to_sim();sim.step(render=False);env.scene.update(cfg.physics_dt)
                last=a
                failed=fell(s)|~torch.isfinite(s).all(-1)
                newly=live&failed;last_step[newly]=step+1;live&=~failed
            means=accum/count.clamp_min(1)[:,None];rows=[]
            for i,c in enumerate(cases):
                v=means[i].cpu().tolist()
                rows.append({**c,'survived':bool(live[i]),'seconds':float(last_step[i])*.02,'xy_rmse':math.sqrt(v[0]),'yaw_rmse':math.sqrt(v[1]),'height':v[2],'mean_vx':v[3],'mean_vy':v[4],'mean_wz':v[5],'paired_action_rmse':math.sqrt(v[6]),'paired_latent_mse':v[7],'mean_abs_action_max':v[8],'slip_geometric_mean_mps':float(slip_sum[i]/slip_count[i].clamp_min(1)),'slip_geometric_p95_mps':float(np.quantile(slip_values[i],.95)) if slip_values[i] else None,'contact_geometric_samples':int(slip_count[i]),'samples':int(count[i]),'pass':bool(live[i]) and math.sqrt(v[0])<.2 and math.sqrt(v[1])<.3 and v[2]>.22})
            summaries={}
            for branch in ('teacher','student'):
                rr=[x for x in rows if x['branch']==branch and x['in_training_range']]
                summaries[branch]={k:sum(float(x[k]) for x in rr)/len(rr) for k in ('xy_rmse','yaw_rmse','height','survived')}
                summaries[branch]['passed_cases']=sum(x['pass'] for x in rr);summaries[branch]['cases']=len(rr)
            result={'iteration':it,'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'backend':type(env.robot).__module__,'summary':summaries,'rows':rows,'training_updates':0,'protocol':{'seconds':10,'ignore_initial_seconds':1,'noise':False,'delay_substeps':1,'initial_yaw':0,'initial_height':.29,'payload_kg':[0,1],'command_resampling':False,'history':'zero then append current frame before first action','slip_definition':'foot sphere bottom point tangential speed, geometric contact if center_z<=radius+2mm; kinematic proxy, not force-threshold contact'},'elapsed_seconds':time.monotonic()-started}
            np.savez_compressed(out/f'poses_{it:08d}.npz',root_pose=np.array(poses),joint_pos=np.array(joints),foot_speed=np.array(foot_speeds),geometric_contact=np.array(contact_masks))
            (out/f'physx_{it:08d}.json').write_text(json.dumps(result,indent=2))
            (out/f'trace_{it:08d}.json').write_text(json.dumps({'cases':cases,'traces':traces,'first_inputs':first_inputs}))
            results.append(result);print(json.dumps({'iteration':it,'summary':summaries}),flush=True)
        (out/'summary.json').write_text(json.dumps({'results':results,'training_updates':0,'source_usd_sha256':hashlib.sha256(Path(args.usd).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
