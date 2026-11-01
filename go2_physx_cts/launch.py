"""Separate robot selection, explicit stage and isolated output; no implicit launch."""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from .config import TrainCfg

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot',required=True,choices=('go2','as2'))
    p.add_argument('--stage',choices=('flat','terrain'),default='flat')
    p.add_argument('--output',required=True)
    p.add_argument('--usd')
    p.add_argument('--resume')
    p.add_argument('--driver',default='')
    p.add_argument('--flat-expert',default='')
    p.add_argument('--evaluation-python',default='')
    p.add_argument('--iterations',type=int,default=120000)
    p.add_argument('--print-config',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg=TrainCfg(robot=a.robot,stage=a.stage,wm_updates=2 if a.stage=='flat' else 4,
               iterations=a.iterations,driver_policy=a.driver,driver_frac=.1 if a.driver else 0.,
               expert_policy=a.flat_expert,expert_frac=.1 if a.flat_expert else 0.,
               allow_missing_flat_expert=True,evaluation_python=a.evaluation_python,
               reset_critic=bool(a.resume and a.stage=='terrain'))
    cfg.validate()
    usd=Path(a.usd) if a.usd else root/'assets'/a.robot/('usd/go2.usd' if a.robot=='go2' else 'as2.usd')
    if not usd.is_file():raise FileNotFoundError(usd)
    if a.print_config:
        print(json.dumps(dict(config=asdict(cfg),usd=str(usd)),indent=2));return
    # Same lock as the existing supervisor: cannot overlap the current Go2 run.
    import fcntl
    lock_path=root/'physx_training.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
        conf=out/'config.json';conf.write_text(json.dumps(asdict(cfg),indent=2))
        command=[sys.executable,'-m','go2_physx_cts.train','--config',str(conf),'--usd',str(usd),'--output',str(out)]
        if a.resume:command+=['--resume',a.resume]
        raise SystemExit(subprocess.call(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root))))

if __name__=='__main__':main()
