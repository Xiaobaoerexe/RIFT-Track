"""Independent As2 sim2sim evaluation. Never imports a learner or updates weights."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
TOOLS=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
 p.add_argument('--iteration',type=int,default=120000)
 p.add_argument('--train-python',default=str(Path.home()/'Documents/go2_supermimic_runtime/physx/bin/python'))
 p.add_argument('--eval-python',default=str(Path.home()/'Documents/go2_supermimic_runtime/robogauge/bin/python'))
 p.add_argument('--check',action='store_true',help='38 real dynamics cases: all ten terrains,seed17,0.5m/s,both branches')
 a=p.parse_args();checkpoint=Path(a.checkpoint).expanduser().resolve();out=Path(a.output).expanduser().resolve()
 if not checkpoint.is_file():p.error('Checkpoint does not exist')
 if out.exists():p.error('Output must be a new directory; existing evidence is never overwritten')
 for executable in (a.train_python,a.eval_python):
  if not Path(executable).is_file():p.error('Runtime missing: '+executable)
 out.mkdir(parents=True);began=time.time()
 source_hashes={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'go2_physx_cts').glob('*.py')}
 original_sha=sha(checkpoint)
 env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1')
 record={'checkpoint':str(checkpoint),'checkpoint_sha256':original_sha,'iteration':a.iteration,'check_only':a.check,
  'source_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
  'tool_sha256':{f.name:sha(f) for f in TOOLS.iterdir() if f.is_file()},'training_updates':0,
  'train_python':a.train_python,'eval_python':a.eval_python,'started_unix':began}
 (out/'execution.json').write_text(json.dumps(record,indent=2))
 def phase(name,command):
  with (out/f'{name}.log').open('w') as log:r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:
   print((out/f'{name}.log').read_text(errors='replace')[-10000:]);raise RuntimeError(name+' failed; see '+str(out/f'{name}.log'))
 phase('export',[a.train_python,str(TOOLS/'export_as2.py'),'--checkpoint',str(checkpoint),'--output',str(out/'exports'),'--iteration',str(a.iteration)])
 phase('prepare',[a.train_python,str(TOOLS/'prepare_assets.py'),'--output',str(out),'--iteration',str(a.iteration)])
 phase('geometry',[a.eval_python,str(TOOLS/'audit_geometry.py'),'--output',str(out)])
 workers=[0] if a.check else [0,1,2];children=[]
 try:
  for worker in workers:
   command=[a.eval_python,str(TOOLS/'evaluate_as2.py'),'--output',str(out),'--iteration',str(a.iteration),'--worker',str(worker)]
   if a.check:command+=['--speeds','0.5']
   log=(out/f'worker_{worker}.log').open('w');child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   children.append((child,log,worker))
  failures=[]
  for child,log,worker in children:
   code=child.wait();log.close()
   if code:failures.append(worker)
  if failures:raise RuntimeError('Evaluation workers failed: '+str(failures))
 finally:
  for child,log,_ in children:
   if child.poll() is None:child.terminate();child.wait()
   if not log.closed:log.close()
 phase('summary',[sys.executable,str(TOOLS/'summarize.py'),'--output',str(out),'--workers',','.join(map(str,workers)),'--expected',str(38 if a.check else 354)])
 unchanged=all(sha(ROOT/name)==digest for name,digest in source_hashes.items()) and sha(checkpoint)==original_sha
 if not unchanged:raise RuntimeError('Training source or original checkpoint changed during evaluation')
 record.update(complete=True,finished_unix=time.time(),elapsed_seconds=time.time()-began,training_sources_unchanged=True,checkpoint_unchanged=True)
 (out/'execution.json').write_text(json.dumps(record,indent=2))
 print(json.dumps({'complete':True,'cases':38 if a.check else 354,'output':str(out),'checkpoint_unchanged':True,'training_updates':0}))

if __name__=='__main__':main()
