"""One authorized formal run: cold flat, terrain only after two physical gates."""
import argparse,json,os,subprocess,sys,time,fcntl
from pathlib import Path
from dataclasses import asdict
from .config import TrainCfg

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--usd',required=True);p.add_argument('--driver',required=True);p.add_argument('--xml',required=True);p.add_argument('--actor-microbatch',type=int,default=8192)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock_path=Path.home()/'Documents/go2_supermimic_lab/physx_training.lock';lock=lock_path.open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=TrainCfg(driver_policy=a.driver,actor_microbatch=a.actor_microbatch,payload_max_kg=1.,evaluation_xml=a.xml,evaluation_python=sys.executable)
    state={'supervisor_pid':os.getpid(),'started_unix':time.time(),'physics':'isaacsim_physx','hard_wall_clock_cap':None,'status':'launching_flat'}
    (out/'supervisor.json').write_text(json.dumps(state,indent=2))
    checkpoint=None
    for stage in ('flat','terrain'):
        stage_out=out/stage;stage_out.mkdir(exist_ok=True)
        if stage=='terrain':
            ready=out/'flat/flat_ready.json'
            if not ready.exists():
                state['status']='flat_finished_requires_review';break
            checkpoint=json.loads(ready.read_text())['checkpoint']
            cfg=TrainCfg.terrain(driver_policy=a.driver,expert_policy=checkpoint,actor_microbatch=a.actor_microbatch,payload_max_kg=1.,evaluation_xml=a.xml,evaluation_python=sys.executable)
        cfg.validate(production=True)
        conf=stage_out/'config.json';conf.write_text(json.dumps(asdict(cfg),indent=2))
        command=[sys.executable,'-m','go2_physx_cts.train','--config',str(conf),'--output',str(stage_out),'--usd',a.usd]
        if checkpoint:command+=['--resume',checkpoint]
        with (stage_out/'console.log').open('w') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state.update(status='running_'+stage,stage=stage,learner_pid=child.pid,command=command)
            (out/'supervisor.json').write_text(json.dumps(state,indent=2))
            code=child.wait()
        state['last_exit_code']=code
        if code:
            state['status']='failed_'+stage;break
        state['status']='finished_'+stage
    state['ended_unix']=time.time();(out/'supervisor.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
