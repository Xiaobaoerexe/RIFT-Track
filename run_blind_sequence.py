"""Run Go2 to completion, then As2; stop on failure or interruption."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
runtime=Path.home()/'Documents/go2_supermimic_runtime/physx/bin/python'
child=None;stopped=False
def stop(signum,frame):
    global stopped
    stopped=True
    if child is not None and child.poll() is None:child.send_signal(signum)
def main():
    global child
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    with (root/'physx_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        if dirty.strip():raise RuntimeError('Formal sequence requires a clean committed release')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        stamp=time.strftime('%Y%m%d_%H%M%S')
        records=root/'runs';records.mkdir(exist_ok=True)
        state={'commit':commit,'started':time.time(),'status':'running','robots':[]}
        state_path=records/('sequence_'+stamp+'.json')
        def save():state_path.write_text(json.dumps(state,indent=2))
        for robot in ('go2','as2'):
            if stopped:break
            cfg_path=root/'configs'/('blind_'+robot+'.json');cfg=json.loads(cfg_path.read_text())
            output=records/(robot.upper()+'_BLIND_OFFICIAL_s17_'+stamp+'_175')
            usd=root/'assets'/robot/('usd/go2.usd' if robot=='go2' else 'as2.usd')
            command=[str(runtime),'-m','go2_physx_cts.train','--config',str(cfg_path),'--output',str(output),'--usd',str(usd)]
            record={'robot':robot,'output':str(output),'log':str(output)+'.log','command':command,'status':'starting'}
            state['robots'].append(record);save()
            with open(record['log'],'w') as log:
                child=subprocess.Popen(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root),OMNI_KIT_ACCEPT_EULA='YES'),stdout=log,stderr=subprocess.STDOUT)
                record.update(pid=child.pid,status='running');save();code=child.wait()
            progress=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {}
            completed=code==0 and progress.get('iteration')==cfg['iterations'] and (output/'student_latest.pt').is_file()
            record.update(exit_code=code,iteration=progress.get('iteration'),status='complete' if completed else 'stopped' if stopped else 'failed')
            save()
            if not completed:
                state['status']=record['status'];save();return 1
        state['status']='stopped' if stopped else 'complete';state['finished']=time.time();save();return int(stopped)
if __name__=='__main__':sys.exit(main())
