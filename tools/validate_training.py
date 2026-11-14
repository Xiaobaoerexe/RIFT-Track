"""Permanent GPU acceptance entry; bounded updates, no automatic formal launch.

Usage: python tools/validate_training.py --config ... --output ... --usd ...
       --validation-start-iteration 5500 --validation-iterations 5
No visualizer option means headless on the pinned Isaac Lab launcher.
The forwarded training CLI supplies --validate. Small-scale configuration is
permitted, but every measured workload dimension is recorded in the result.
"""
import json,math,statistics,subprocess,sys,threading,time,traceback
from dataclasses import asdict
from pathlib import Path


class GpuSampler:
    def __init__(self,index):
        self.index=index;self.rows=[];self.errors=[];self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        while not self.stop_event.is_set():
            try:
                result=subprocess.run(['nvidia-smi',f'--id={self.index}',
                    '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
                values=[None if v.strip() in ('N/A','[N/A]','[Not Supported]') else float(v.strip()) for v in result.stdout.strip().split(',')]
                if len(values)!=4 or any(v is None for v in values[:3]):raise ValueError('Required GPU utilization/memory counters are unavailable')
                self.rows.append(dict(zip(('utilization_pct','used_mib','total_mib','power_w'),values),monotonic=time.monotonic()))
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                self.errors.append(type(exc).__name__+': '+str(exc))
            self.stop_event.wait(1.)
    def start(self):self.thread.start()
    def stop(self):self.stop_event.set();self.thread.join(timeout=4.)


def _memory(torch,device,label):
    free,total=torch.cuda.mem_get_info(device)
    return {'stage':label,'allocated_bytes':torch.cuda.memory_allocated(device),
        'reserved_bytes':torch.cuda.memory_reserved(device),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'device_free_bytes':free,'device_total_bytes':total}


def _latent_gradient_probe(learner):
    import torch
    from go2_physx_cts.learner import frozen
    from go2_physx_cts.terrain import geometry_crop
    from go2_physx_cts import state as st
    seed=learner.buf.seeds(16,burnin=0)
    s,pose=seed['s'],seed['pose'];ids=lambda k:seed[k].reshape(-1).long()
    with torch.no_grad():
        images=learner.depth.captured(seed['depth_pose'],ids('depth_scene_ids'),ids('depth_frame_index'),ids('episode'),ids('trajectory_ids'))
        initial=learner.policy.student_encoder.encode_depth(images)
    with frozen(learner.wm,learner.critic,learner.policy.student_encoder):
        action=seed['a'].detach().clone().requires_grad_(True)
        # Explicitly probe a refresh transition. This checks the GPU derivative
        # interface; it is not a measured future-image accuracy claim.
        frame=torch.full((len(s),),learner.cfg.depth_period-1,device=s.device,dtype=torch.long)
        _,_,_,next_z=learner.wm.predict(s,action,seed['a_prev'],seed['xi'],geometry_crop(s,pose,learner.env.field),
            pose,initial,frame,seed['depth_pose'])
        weight=torch.linspace(-1,1,next_z.shape[-1],device=s.device)
        action_vjp=torch.autograd.grad((next_z*weight).sum(),action)[0]
        z=initial.detach().clone().requires_grad_(True)
        hist=learner.policy.history_norm(seed['obs'].float())
        single=learner.policy.single_norm(st.single_from_history(seed['obs'].float()))
        fused,_,hidden=learner.policy.student_encoder.forward_features(hist,single,z,torch.zeros(len(s),256,device=s.device))
        output=learner.policy.actor(fused,single)
        latent_vjp=torch.autograd.grad(output[:,0].sum(),z)[0]
    if not bool(torch.isfinite(action_vjp).all() and torch.isfinite(latent_vjp).all()):
        raise FloatingPointError('Nonfinite visual imagination gradient')
    if not bool(action_vjp.abs().sum()>0 and latent_vjp.abs().sum()>0):
        raise RuntimeError('A required latent imagination input gradient is absent')
    return {'action_to_refreshed_latent_vjp_l2':float(action_vjp.norm()),
        'latent_to_student_actor_vjp_l2':float(latent_vjp.norm()),
        'frozen_module_parameter_grads_absent':all(p.grad is None for m in (learner.wm,learner.policy.student_encoder) for p in m.parameters()),
        'scope':'GPU VJP on real replay states; forced refresh phase; not latent prediction accuracy'}


def _event_probe(learner):
    import torch
    if learner.cfg.variant=='simple':return {'enabled':False,'reason':'No event rewards/value output in simple variant'}
    from go2_physx_cts.learner import frozen
    x,y=learner.event_training_data()
    with frozen(learner.critic):
        live=x.detach().clone().requires_grad_(True)
        value=learner.critic.forward_event(live)
        gradient=torch.autograd.grad(value.sum(),live)[0]
    if not bool(torch.isfinite(value).all() and torch.isfinite(y).all() and torch.isfinite(gradient).all()):
        raise FloatingPointError('Invalid event-value supervision/VJP')
    return {'enabled':True,'states':len(x),'feature_dim':x.shape[-1],
        'target_range':[float(y.min()),float(y.max())],
        'value_input_vjp_l2':float(gradient.norm()),'nonzero_targets':int((y!=0).sum()),
        'scope':'Real-event TD interface and input derivative; does not establish event-value calibration or success improvement'}


def validate_training(learner,*,start_iteration=1500,prefill=96,wm_warmup=4,warmup_iterations=2,iterations=5,diagnostics_steps=0):
    import torch
    if not torch.cuda.is_available() or torch.device(learner.device).type!='cuda':
        raise RuntimeError('GPU acceptance requires an actual CUDA PhysX run')
    if start_iteration<learner.cfg.policy_start:raise ValueError('Validation must cover Actor backpropagation, not only prewarm')
    if min(prefill,iterations)<1 or min(warmup_iterations,wm_warmup)<0:raise ValueError('Invalid validation workload')
    device=torch.device(learner.device);index=device.index or 0
    monitor=GpuSampler(index);report={'schema':'as2_depth_gpu_validation_v1','status':'running','config':asdict(learner.cfg),
        'start_iteration':start_iteration,'iterations':iterations,'warmup_iterations':warmup_iterations,
        'cold_compile_included_in_warmup':True,'formal_training_started':False,'memory':[],'timings':[],
        'progress':[],'instrumentation_seconds':0.}
    path=learner.output/'validation.json';monitor.start()
    validation_begin=time.monotonic()
    def publish(stage,**fields):
        # Persist before entering each potentially long compile/update. An
        # interrupted process then leaves an honest last-started phase instead
        # of no evidence. Replacing the sibling file keeps readers from seeing
        # a half-written JSON document.
        io_begin=time.monotonic()
        memory=_memory(torch,device,stage)
        event={'stage':stage,'elapsed_seconds':time.monotonic()-validation_begin,
            'iteration':learner.it,'memory':memory,**fields}
        report['current_stage']=stage;report['progress'].append(event);report['memory'].append(memory)
        report['counts']=learner.counts.copy()
        temporary=path.with_name(path.name+'.partial')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(path)
        print(json.dumps({'validation_progress':event},allow_nan=False),flush=True)
        report['instrumentation_seconds']+=time.monotonic()-io_begin
    try:
        publish('initialized')
        learner.it=start_iteration;learner.env.iteration=start_iteration
        publish('rollout_initialization_begin')
        learner.env.snapshot();learner.refresh_providers();learner.reset_rollout_state()
        publish('rollout_initialization_end')
        required=max(prefill,learner.cfg.cts_burnin+learner.cfg.cts_sequence_steps+learner.cfg.wm_window)
        publish('prefill_begin',vector_steps_required=required)
        tick=time.monotonic()
        for step in range(required):
            learner.collect()
            if (step+1)%100==0:
                publish('prefill_progress',vector_steps_completed=step+1,vector_steps_required=required)
        torch.cuda.synchronize(device)
        report['prefill']={'vector_steps':required,'real_transitions':required*learner.cfg.num_envs,
            'seconds':time.monotonic()-tick,'counts_as_learning_iterations':False}
        publish('prefill_end',vector_steps_completed=required,vector_steps_required=required)
        publish('loaded_fk_check_begin')
        report['loaded_fk_check']=learner.env.validate_kinematics()
        publish('loaded_fk_check_end')
        learner.learning_rates()
        tick=time.monotonic()
        for call in range(wm_warmup):
            publish('wm_preconditioning_begin',call=call+1,calls=wm_warmup)
            loss=learner.fit_world_model()
            publish('wm_preconditioning_end',call=call+1,calls=wm_warmup,world_model_loss=float(loss))
        torch.cuda.synchronize(device)
        report['wm_preconditioning']={'calls':wm_warmup,'optimizer_steps':wm_warmup*learner.cfg.wm_updates_for(start_iteration),
            'seconds':time.monotonic()-tick,'counts_as_learning_iterations':False}
        steady_start=None
        for iteration in range(warmup_iterations+iterations):
            is_warmup=iteration<warmup_iterations
            if iteration==warmup_iterations:
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);steady_start=time.monotonic()
            learner.it=start_iteration+iteration;learner.env.iteration=learner.it
            torch.cuda.synchronize(device);begin=time.monotonic();metrics={};phases={}
            instrumentation_begin=report['instrumentation_seconds']
            publish('complete_update_begin',update_index=iteration,warmup=is_warmup)
            learner.learning_rates()
            for name,operation in (
                ('collect',learner.collect),('world_model',learner.fit_world_model),
                ('actor_critic',lambda:learner.improve_policy(detailed_timing=learner.it%learner.cfg.log_interval==0)),('student',learner.update_cts)):
                publish(name+'_begin',update_index=iteration,warmup=is_warmup)
                tick=time.monotonic();result=operation();phases[name+'_host_seconds']=time.monotonic()-tick
                if name in ('world_model','student'):metrics[name+'_loss']=float(result)
                elif isinstance(result,dict):metrics.update(result)
                publish(name+'_end',update_index=iteration,warmup=is_warmup,host_seconds=phases[name+'_host_seconds'])
            torch.cuda.synchronize(device);elapsed=time.monotonic()-begin
            if not all(math.isfinite(v) for v in metrics.values() if isinstance(v,(int,float))):
                raise FloatingPointError('Nonfinite full-update metric')
            row={'iteration':learner.it,'warmup':is_warmup,'seconds':elapsed,'host_phase_times':phases,'metrics':metrics,
                'progress_instrumentation_seconds':report['instrumentation_seconds']-instrumentation_begin,
                'replay_transport_cumulative':learner.buf.transport.metadata(),
                'archive_transport_cumulative':learner.archive.transport.metadata()}
            report['timings'].append(row)
            publish('complete_update_end',update_index=iteration,warmup=is_warmup,seconds=elapsed)
            print(json.dumps({'validation_iteration':row},allow_nan=False),flush=True)
        steady_end=time.monotonic()
        measured=[r['seconds'] for r in report['timings'] if not r['warmup']]
        report['steady']={'mean_seconds':statistics.mean(measured),'median_seconds':statistics.median(measured),
            'minimum_seconds':min(measured),'maximum_seconds':max(measured),'updates':len(measured),
            'real_transitions_per_second':learner.cfg.num_envs/statistics.mean(measured),
            'imagined_transitions_per_second':learner.cfg.imag_batch*learner.cfg.horizon/statistics.mean(measured),
            'projected_120k_hours_without_evaluation_or_checkpoint':statistics.mean(measured)*120000/3600}
        publish('latent_gradient_probe_begin')
        report['latent_gradient_probe']=_latent_gradient_probe(learner)
        publish('latent_gradient_probe_end')
        publish('event_value_probe_begin')
        report['event_value_probe']=_event_probe(learner)
        publish('event_value_probe_end')
        if diagnostics_steps:
            from go2_physx_cts.diagnostics import native_diagnostics
            publish('native_diagnostics_begin',steps=diagnostics_steps)
            report['native_diagnostics']=native_diagnostics(learner,steps=diagnostics_steps,seeds=(17,))
            publish('native_diagnostics_end',steps=diagnostics_steps)
        samples=[r for r in monitor.rows if steady_start<=r['monotonic']<=steady_end]
        report['gpu_steady_samples']=samples
        if samples:
            report['gpu_utilization']={'mean_pct':statistics.mean(r['utilization_pct'] for r in samples),
                'maximum_pct':max(r['utilization_pct'] for r in samples),'samples':len(samples)}
        report['counts']=learner.counts.copy();report['status']='passed'
        report['limitations']=['Bounded runtime/finite-gradient qualification, not learned locomotion capability.',
            'Host phase timings include asynchronous launches; synchronized full-iteration wall time is the throughput measure.',
            'Full-iteration wall time includes progress instrumentation; its measured host overhead is recorded separately.',
            'Small configurations establish functionality only; production shape must be measured independently.',
            'No policy checkpoint or formal training is created by this entry.']
        return report
    except Exception as exc:
        report['status']='failed';report['error']=type(exc).__name__+': '+str(exc);report['traceback']=traceback.format_exc()
        raise
    finally:
        monitor.stop();report['gpu_sampler_errors']=monitor.errors
        report['gpu_samples']=monitor.rows
        report['replay_storage']=learner.buf.metadata()
        report['wm_archive_storage']=learner.archive.metadata()
        meminfo=Path('/proc/meminfo')
        if meminfo.exists():
            report['host_memory_kib']={line.split(':')[0]:int(line.split()[1]) for line in meminfo.read_text().splitlines()
                if line.split(':')[0] in ('MemTotal','MemAvailable','SwapTotal','SwapFree')}
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    if '--validate' not in sys.argv:sys.argv.append('--validate')
    from go2_physx_cts.train import main
    main()
