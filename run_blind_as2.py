"""Launch only the revised blind As2 experiment; one shared GPU training lock."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME=Path.home()/'Documents/go2_supermimic_runtime/physx/bin/python'
child=None
stopped=False
def stop(signum,frame):
    global stopped
    stopped=True
    if child is not None and child.poll() is None:child.send_signal(signum)
def main():
    global child
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    with (ROOT/'physx_training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Formal launch requires a clean committed release')
        cfg_path=ROOT/'configs/blind_as2.json';cfg=json.loads(cfg_path.read_text())
        if cfg['robot']!='as2' or cfg['iterations']!=120000 or cfg['driver_frac'] or cfg['expert_frac']:
            raise RuntimeError('This release requires As2,120000 iterations,no external experts')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        stamp=time.strftime('%Y%m%d_%H%M%S');runs=ROOT/'runs';runs.mkdir(exist_ok=True)
        output=runs/('AS2_BLIND_WINDOWED_s17_'+stamp+'_175')
        state_path=runs/('as2_launch_'+stamp+'.json')
        command=[str(RUNTIME),'-m','go2_physx_cts.train','--config',str(cfg_path),
                 '--output',str(output),'--usd',str(ROOT/'assets/as2/as2.usd')]
        state={'commit':commit,'supervisor_pid':os.getpid(),'robot':'as2','iterations':120000,
               'external_experts':False,'started_unix':time.time(),'output':str(output),
               'log':str(output)+'.log','command':command,'status':'starting'}
        def save():state_path.write_text(json.dumps(state,indent=2))
        save()
        with Path(state['log']).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,
                env=dict(os.environ,PYTHONPATH=str(ROOT),OMNI_KIT_ACCEPT_EULA='YES',PYTHONUNBUFFERED='1'),
                stdout=log,stderr=subprocess.STDOUT)
            state.update(pid=child.pid,status='running');save();print(json.dumps(state),flush=True)
            code=child.wait()
        progress=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {}
        complete=code==0 and progress.get('iteration')==120000 and (output/'student_latest.pt').exists()
        state.update(exit_code=code,finished_unix=time.time(),iteration=progress.get('iteration'),
                     status='complete' if complete else 'stopped' if stopped else 'failed')
        save();return 0 if complete else 1
if __name__=='__main__':sys.exit(main())
