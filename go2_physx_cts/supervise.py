"""Cold-start depth CTS supervisor with an iteration budget and no hour cap."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from dataclasses import asdict
from .config import TrainCfg

def main():
    p=argparse.ArgumentParser()
    for key in ('output','usd','driver','reward-calibration','evaluation-python'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--actor-microbatch',type=int,default=2048)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    lock_path=Path.home()/'Documents/go2_depth_training.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=TrainCfg(driver_policy=a.driver,reward_calibration=a.reward_calibration,
                     actor_microbatch=a.actor_microbatch,evaluation_python=a.evaluation_python)
        cfg.validate(production=True)
        config=out/'config.json';config.write_text(json.dumps(asdict(cfg),indent=2))
        command=[sys.executable,'-m','go2_physx_cts.train','--config',str(config),
                 '--output',str(out),'--usd',a.usd]
        state={'supervisor_pid':os.getpid(),'started_unix':time.time(),'physics':'isaacsim_physx',
               'hard_wall_clock_cap':None,'status':'launching','initialization':'cold',
               'iteration_budget':120000,'command':command,'data_storage':'data2'}
        with (out/'console.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state.update(status='running',learner_pid=child.pid)
            (out/'supervisor.json').write_text(json.dumps(state,indent=2))
            code=child.wait()
        state.update(last_exit_code=code,ended_unix=time.time(),status='failed' if code else 'iteration_review_required')
        (out/'supervisor.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()
