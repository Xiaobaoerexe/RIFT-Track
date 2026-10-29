"""Authorized mixed-terrain stage from the measured flat baseline; no time cap."""
import argparse,json,os,subprocess,sys,time,fcntl,hashlib
from pathlib import Path
from dataclasses import asdict
from .config import TrainCfg

def main():
    p=argparse.ArgumentParser()
    for name in ('output','usd','driver','resume','evaluation-python'):p.add_argument('--'+name,required=True)
    p.add_argument('--actor-microbatch',type=int,default=8192)
    p.add_argument('--iterations',type=int,default=60000)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    lock_path=Path.home()/'Documents/go2_supermimic_lab/physx_training.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checkpoint=Path(a.resume).resolve()
        cfg=TrainCfg.terrain(driver_policy=a.driver,expert_policy=str(checkpoint),
            actor_microbatch=a.actor_microbatch,payload_max_kg=2.,
            reset_critic=True,evaluation_python=a.evaluation_python,iterations=a.iterations)
        cfg.validate(production=True)
        stage_out=out/'terrain';stage_out.mkdir()
        conf=stage_out/'config.json';conf.write_text(json.dumps(asdict(cfg),indent=2))
        command=[sys.executable,'-m','go2_physx_cts.train','--config',str(conf),
                 '--output',str(stage_out),'--usd',a.usd,'--resume',str(checkpoint)]
        state={'supervisor_pid':os.getpid(),'started_unix':time.time(),'physics':'isaacsim_physx',
               'hard_wall_clock_cap':None,'stage':'terrain','status':'launching',
               'parent_checkpoint':str(checkpoint),'parent_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'initial_iteration_review':cfg.iterations,'final_velocity_stage_iteration':45000,
               'review_reason':'60000 covers all official velocity stages plus15000 iterations at final range; review native and transfer evidence separately',
               'command':command}
        with (stage_out/'console.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state.update(status='running_terrain',learner_pid=child.pid)
            (out/'supervisor.json').write_text(json.dumps(state,indent=2))
            code=child.wait()
        state.update(last_exit_code=code,ended_unix=time.time(),status='failed' if code else 'iteration_review_required')
        (out/'supervisor.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
