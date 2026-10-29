"""Frozen checkpoint to official RoboGauge model ABI. Run in training runtime."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .networks import Policy,StudentExport
from . import state as st
class StatefulStudent(nn.Module):
 def __init__(self,core):
  super().__init__();self.core=core;self.register_buffer('history',torch.zeros(1,450))
 @torch.jit.export
 def reset(self):self.history.zero_()
 def forward(self,obs):
  h=self.history
  self.history.copy_(torch.cat((h[:,3:30],obs[:,:3],h[:,33:60],obs[:,3:6],h[:,63:90],obs[:,6:9],h[:,102:210],obs[:,9:21],h[:,222:330],obs[:,21:33],h[:,342:450],obs[:,33:45]),-1))
  return self.core(self.history)

class TeacherCore(nn.Module):
 def __init__(self,p):
  super().__init__();self.teacher_norm=p.teacher_norm;self.teacher=p.teacher_encoder;self.single_norm=p.single_norm;self.actor=p.actor
 def forward(self,obs):return self.actor(torch.cat((self.teacher(self.teacher_norm(obs[:,45:])),self.single_norm(obs[:,:45])),-1))

class TeacherModel(nn.Module):
 def __init__(self,core):super().__init__();self.core=core
 @torch.jit.export
 def reset(self):pass
 def forward(self,obs):return self.core(obs)

def export_checkpoint(checkpoint,output,iteration):
 torch.set_num_threads(2);torch.manual_seed(17);torch.set_grad_enabled(False)
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True);it=iteration
 path=Path(checkpoint);ck=torch.load(path,map_location='cpu',weights_only=False)
 if ck['cfg'].get('robot','go2')!='go2':raise ValueError('Pinned RoboGauge bridge is verified for Go2 only')
 p=Policy(tuple(ck['cfg']['actor_hidden'])).eval();p.load_state_dict(ck['policy'])
 meta={'iteration':it,'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'consts':{k:v.tolist() for k,v in ck['consts'].items()},'models':{},'normalizer_counts':{k:float(getattr(p,k).count) for k in ('history_norm','single_norm','teacher_norm')}}
 for branch in ('teacher','student'):
  if branch=='teacher':model=torch.jit.script(TeacherModel(torch.jit.trace(TeacherCore(p),torch.zeros(1,212))))
  else:model=torch.jit.script(StatefulStudent(torch.jit.trace(StudentExport(p),torch.zeros(1,450))))
  obs=torch.randn(50,45)*.2;priv=torch.randn(50,167)*.2;hist=torch.zeros(1,450);pred=[];actual=[];inputs=[]
  model.reset()
  for i in range(50):
   if i==25:model.reset();hist.zero_()
   hist=st.push_history(hist,obs[i:i+1]);x=torch.cat((obs[i:i+1],priv[i:i+1]),-1) if branch=='teacher' else obs[i:i+1]
   inputs.append(x[0].numpy());pred.append(model(x)[0].numpy());actual.append(p(hist,priv[i:i+1],0 if branch=='teacher' else 1)[0].numpy())
  err=float(np.max(np.abs(np.array(pred)-np.array(actual))));assert err<1e-5,(branch,err)
  model.reset();dst=out/f'{it}_{branch}.pt';model.save(str(dst))
  np.savez_compressed(out/f'{it}_{branch}_golden.npz',inputs=np.array(inputs),expected=np.array(actual),reset_at=np.array([0,25]))
  meta['models'][branch]={'path':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'export_max_error':err,'input_dim':212 if branch=='teacher' else 45}
 (out/f'{it}_metadata.json').write_text(json.dumps(meta,indent=2))
 return meta

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--iteration',required=True,type=int);a=p.parse_args()
 print(json.dumps(export_checkpoint(a.checkpoint,a.output,a.iteration)))
if __name__=='__main__':main()
