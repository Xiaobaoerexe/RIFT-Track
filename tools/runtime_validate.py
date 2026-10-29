import torch,numpy as np,json,sys,importlib.metadata as md,hashlib
from pathlib import Path
r=Path('/home/abc/data2/go2_supermimic_lab/reports/ROBOGAUGE_MIGRATION_20261029')
torch.set_num_threads(1);torch.set_grad_enabled(False)
versions={d.metadata['Name']:d.version for d in md.distributions()}
assert md.version('mujoco')=='3.2.3'
assert md.version('dm-control')=='1.0.23'
assert torch.__version__=='2.4.1+cu124'
checks=[]
for f in sorted((r/'exports').glob('*_golden.npz')):
 data=np.load(f);model=torch.jit.load(str(f).replace('_golden.npz','.pt')).eval();pred=[]
 for i,x in enumerate(data['inputs']):
  if i in data['reset_at']:model.reset()
  pred.append(model(torch.from_numpy(x).unsqueeze(0))[0].numpy())
 err=float(np.abs(np.asarray(pred)-data['expected']).max());assert err<1e-5,(f,err)
 checks.append({'model':f.stem,'max_abs_error':err,'reset_at':data['reset_at'].tolist()})
out={'python':sys.version,'versions':versions,'golden_checks':checks,'training_updates':0}
(r/'runtime_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
