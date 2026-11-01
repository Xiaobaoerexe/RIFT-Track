"""Frozen native/model diagnostics. No curriculum or transfer-based rollback."""
import torch
from . import state as st
from .commands import RANGE_ITERATIONS,RANGE_STAGES,TERRAIN_RANGES,coupled_clip

def terrain_ids(env,xy):
    lookup=env.command_sampler.terrain_lookup
    return (torch.zeros(len(xy),device=xy.device,dtype=torch.long),('flat',)) if lookup is None else (lookup.query(xy),lookup.names)

@torch.no_grad()
def native_diagnostics(learner,steps=500):
    env,p=learner.env,learner.policy;dev=env.device;n=env.num_envs
    stage=sum(learner.it>=v for v in RANGE_ITERATIONS)
    current=RANGE_STAGES[stage][0][1];adjacent=RANGE_STAGES[min(stage+1,len(RANGE_STAGES)-1)][0][1]
    commands=torch.tensor([(0,0,0),(.5,0,0),(-.3,0,0),(0,.3,0),(0,0,.8),(current,0,0),(adjacent,0,0)],device=dev)
    index=torch.arange(n,device=dev);ci=index%len(commands);payload=(index//len(commands)%3).float()
    rows=[];env.evaluation_mode=True
    try:
        for branch in (0,1):
            env.command_override=None;env.reset(env.ids);env.set_payload(env.ids,payload)
            snap=env._observe();kind,names=terrain_ids(env,snap['pose'][:,:2])
            table=commands.new_tensor([TERRAIN_RANGES['flat']]+[TERRAIN_RANGES[name] for name in names])
            limits=table[kind+1];cmd=coupled_clip(commands[ci].clamp(min=limits[...,0],max=limits[...,1]))
            env.command_override=cmd;env.command=cmd.clone();env._snapshot=env._observe()
            alive=torch.ones(n,device=dev,dtype=torch.bool);samples=torch.zeros(n,device=dev)
            sums=torch.zeros(n,6,device=dev)
            for i in range(steps):
                b=env.snapshot();priv=st.privileged(b['s'],b['a_prev'],cmd,env.xi,b['relief'])
                action=p(b['obs'],priv,branch);after,done,_=env.step(action)
                alive &= ~done
                if i>=50:
                    s=after['s'];v=st.velocity_body(s)
                    value=torch.stack(((v[:,:2]-cmd[:,:2]).square().mean(-1),(s[:,9]-cmd[:,2]).square(),s[:,0],v[:,0],v[:,1],s[:,9]),-1)
                    sums+=value*alive[:,None];samples+=alive
            means=sums/samples.clamp_min(1)[:,None];xy=means[:,0].sqrt();yaw=means[:,1].sqrt()
            # Preserve historical low-speed thresholds; higher-speed results are
            # diagnostics with a20%-command tolerance, never a course gate.
            tolerance=torch.maximum(torch.full_like(xy,.2),.2*cmd[:,:2].norm(dim=-1))
            success=alive&(samples>0)&(xy<tolerance)&(yaw<.3)&(means[:,2]>.22)
            for k in kind.unique().tolist():
                for c in range(len(commands)):
                    for load in (0,1,2):
                        mask=(kind==k)&(ci==c)&(payload==load)
                        if not mask.any():continue
                        m=means[mask]
                        rows.append({'branch':('teacher','student')[branch],'terrain':'border' if k<0 else names[k],
                            'case':('standing','forward_low','reverse_low','lateral_low','turn_low','current_speed','adjacent_speed')[c],
                            'payload_kg':load,'command':cmd[mask][0].tolist(),'trials':int(mask.sum()),
                            'survival_fraction':float(alive[mask].float().mean()),'success_fraction':float(success[mask].float().mean()),
                            'xy_rmse':float(xy[mask].mean()),'yaw_rmse':float(yaw[mask].mean()),
                            'height':float(m[:,2].mean()),'vx':float(m[:,3].mean()),'vy':float(m[:,4].mean()),'wz':float(m[:,5].mean()),
                            'task_ratio_target':.8,'adjacent_failure_allowance':.5,'automatic_course_gate':False})
    finally:
        env.command_override=None;env.reset(env.ids)
        env.set_payload(env.ids,torch.rand(n,device=dev)*env.cfg.payload_max_kg)
        env.evaluation_mode=False;env._snapshot=env._observe()
        learner.ou.zero_()
        if learner.driver is not None:learner.driver.reset(torch.ones(n,device=dev,dtype=torch.bool))
    return {'iteration':learner.it,'rows':rows,'duration_seconds':steps*env.cfg.dt,
            'note':'Frozen native diagnostics on current terrain levels; each trial is one env. No replay or optimizer updates. High-speed thresholds are diagnostic, not source-certified capability criteria.'}

@torch.no_grad()
def model_diagnostics(learner,batch=2048):
    w=learner.buf.windows(batch,learner.cfg.wm_window,learner.env.xi)
    s=w['s'][:,0];errors=[]
    for j in range(learner.cfg.wm_window):
        s,_=learner.wm(s,w['a'][:,j],w['a_prev'][:,j],w['xi'],w['relief'][:,j].float())
        errors.append((s-w['s'][:,j+1]).square().mean(-1))
    err=torch.stack(errors,1);valid=w['valid'];mse=(err*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    kind,names=terrain_ids(learner.env,w['pose'][:,0,:2]);speed=w['command'][:,:2].norm(dim=-1)
    rows=[]
    for k in kind.unique().tolist():
        for lo,hi in ((0,.5),(.5,1.),(1.,2.),(2.,5.)):
            for a,b in ((0.,1.),(1.,2.01)):
                mask=(kind==k)&(speed>=lo)&(speed<hi)&(w['xi'][:,4]>=a)&(w['xi'][:,4]<b)&(valid.sum(-1)>0)
                if mask.any():rows.append({'terrain':'border' if k<0 else names[k],'command_speed_bin':[lo,hi],
                    'payload_bin':[a,b],'windows':int(mask.sum()),'state70_raw_mse':float(mse[mask].mean())})
    return {'iteration':learner.it,'rows':rows,'sampling':'uniform replay diagnostic; no sampling-weight changes',
            'note':'Raw mixed-unit state MSE, not a capability gate or normalized reward metric.'}


@torch.no_grad()
def native_candidate_probe(learner,policy,steps=128,stability=False):
    """Discarded deterministic real-PhysX LR trial, never added to Replay."""
    env=learner.env;cfg=learner.cfg;n=env.num_envs;dev=env.device
    index=torch.arange(n,device=dev);teacher=index%2==0
    commands=torch.tensor([(0.,0.,0.),(.5,0.,0.),(0.,.3,0.),(0.,0.,.8)],device=dev)[(index//2)%4]
    if not stability:
        commands=commands.clone();commands[(index//2)%4==1,0]=RANGE_STAGES[sum(learner.it>=v for v in RANGE_ITERATIONS)][0][1]
    saved_types=saved_levels=saved_origins=None
    if env.field is not None and stability:
        terrain=env.scene.terrain
        saved_types=terrain.terrain_types.clone();saved_levels=terrain.terrain_levels.clone();saved_origins=terrain.env_origins.clone()
        columns=(env.terrain_lookup.column_types==env.terrain_names.index('flat')).nonzero().flatten()
        terrain.terrain_types[:]=columns[index%len(columns)];terrain.terrain_levels[:]=(index//len(columns))%terrain.cfg.terrain_generator.num_rows
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    env.evaluation_mode=True;env.command_override=commands
    try:
        env.reset(env.ids);env.set_payload(env.ids,torch.zeros(n,device=dev));env._snapshot=env._observe()
        alive=torch.ones(n,device=dev,dtype=torch.bool)
        returns=torch.zeros(n,device=dev);error=torch.zeros(n,device=dev);yaw_error=torch.zeros(n,device=dev);samples=torch.zeros(n,device=dev)
        for step in range(steps):
            before=env.snapshot()
            priv=st.privileged(before['s'],before['a_prev'],commands,env.xi,before['relief'])
            action=torch.zeros(n,12,device=dev)
            for branch,mask in ((0,teacher),(1,~teacher)):action[mask]=policy(before['obs'][mask],priv[mask],branch)
            after,done,_=env.step(action)
            from .task import reward
            r,_,_=reward(before['s'],after['s'],action,before['a_prev'],commands,env.consts,cfg.stage,
                iteration=learner.it,height_above_crop=after['height_above_crop'],base_height_target=env.profile.reward_height,
                aux=after['aux'],previous_previous=before['obs'][:,-24:-12],on_flat=after['terrain_id']==env.terrain_names.index('flat'))
            returns+=r*alive;error+=(st.velocity_body(after['s'])[:,:2]-commands[:,:2]).square().mean(-1)*alive
            yaw_error+=(after['s'][:,9]-commands[:,2]).square()*alive;samples+=alive
            alive &= ~done
        rmse=(error/samples.clamp_min(1)).sqrt()
        success=alive&(rmse<.2)&((yaw_error/samples.clamp_min(1)).sqrt()<.3)&(after['s'][:,0]>.22)
        return {'real_return':float(returns[teacher].mean()+returns[~teacher].mean()),
            'falls_per_env_min':float((~alive).float().mean())*60/(steps*cfg.dt),
            'tracking_rmse':float(rmse.mean()),'teacher_return':float(returns[teacher].mean()),
            'teacher_success':float(success[teacher].float().mean())}
    finally:
        if saved_types is not None:
            terrain.terrain_types.copy_(saved_types);terrain.terrain_levels.copy_(saved_levels);terrain.env_origins.copy_(saved_origins)
        env.command_override=None;env.reset(env.ids);env.evaluation_mode=False
        env.set_payload(env.ids,torch.rand(n,device=dev)*cfg.payload_max_kg);env._snapshot=env._observe()
