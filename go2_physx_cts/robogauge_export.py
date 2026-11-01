"""Visual recurrent CTS export for the pinned official RoboGauge control loop."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .networks import Policy,StudentExport
from . import state as st

STUDENT_INPUT=45+48*64
TEACHER_INPUT=45+st.PRIV_DIM+2*64*128+1
CHECKPOINT_SCHEMA='physx-depth-multitask-v2'
EXPORT_SCHEMA='depth_cts_export_v2'

class StatefulStudent(nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core
        self.register_buffer('history',torch.zeros(1,450));self.register_buffer('hidden',torch.zeros(1,256))
    @torch.jit.export
    def reset(self):self.history.zero_();self.hidden.zero_()
    def forward(self,obs):
        h=self.history;x=obs[:,:45]
        history=torch.cat((h[:,3:30],x[:,:3],h[:,33:60],x[:,3:6],h[:,63:90],x[:,6:9],h[:,102:210],x[:,9:21],h[:,222:330],x[:,21:33],h[:,342:450],x[:,33:45]),-1)
        self.history.copy_(history)
        action,hidden=self.core(history,obs[:,45:].reshape(-1,1,48,64),self.hidden,torch.zeros(1,dtype=torch.bool,device=obs.device))
        self.hidden.copy_(hidden)
        return action

class TeacherCore(nn.Module):
    def __init__(self,p):
        super().__init__();self.teacher_norm=p.teacher_norm;self.teacher=p.teacher_encoder
        self.single_norm=p.single_norm;self.actor=p.actor
    def forward(self,obs):
        z=self.teacher(self.teacher_norm(obs[:,45:212]),obs[:,212:-1].reshape(-1,2,64,128),obs[:,-1].long())
        return self.actor(z,self.single_norm(obs[:,:45]))

class TeacherModel(nn.Module):
    def __init__(self,core):super().__init__();self.core=core
    @torch.jit.export
    def reset(self):pass
    def forward(self,obs):return self.core(obs)

def export_checkpoint(checkpoint,output,iteration):
    torch.set_num_threads(2);torch.manual_seed(17)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    path=Path(checkpoint);ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('schema')!=CHECKPOINT_SCHEMA:
        raise ValueError('Architecture mismatch: export requires the shared-Critic-geometry checkpoint and its recorded expert count')
    config=ck['cfg'];period=int(config['depth_period']);control_dt=float(config['dt'])
    if period<=0 or control_dt<=0:raise ValueError('Invalid checkpoint camera/control cadence')
    # BF16 is a training arithmetic mode; deploy the FP32 master weights through
    # the pinned CPU official runtime, with no implicit expert-count default.
    p=Policy(tuple(config['actor_hidden']),experts=int(config['actor_experts']),mixed_precision=False).float().eval()
    p.load_state_dict(ck['policy'])
    with torch.no_grad():
        student_core=torch.jit.trace(StudentExport(p),(torch.zeros(1,450),torch.ones(1,1,48,64),torch.zeros(1,256),torch.zeros(1,dtype=torch.bool)))
        models={'student':torch.jit.script(StatefulStudent(student_core)),
                'teacher':torch.jit.script(TeacherModel(torch.jit.trace(TeacherCore(p),torch.zeros(1,TEACHER_INPUT))))}
        meta={'schema':EXPORT_SCHEMA,'architecture_version':CHECKPOINT_SCHEMA,'iteration':iteration,'checkpoint':str(path),
              'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'consts':{k:v.tolist() for k,v in ck['consts'].items()},'models':{},
              'depth_package':config['depth_package'],
              'network':{'actor_experts':int(config['actor_experts']),'proprio_experts':int(config['proprio_experts']),
                         'training_mixed_precision':config.get('mixed_precision',False),'export_precision':'fp32'},
              'camera':{'depth_shape':[48,64],'depth_units':'meters_optical_axis','rate_hz':1/(control_dt*period),'control_hz':1/control_dt,
                        'control_dt':control_dt,'depth_period':period,'noise_key_frame':'episode_control_step',
                        'mount_xyz':[.101958,0,.327],'student_hidden_dim':256},
              'normalizer_counts':{k:float(getattr(p,k).count) for k in ('history_norm','single_norm','teacher_norm')}}
        for branch,model in models.items():
            hist=torch.zeros(1,450);hidden=torch.zeros(1,256);pred=[];actual=[];inputs=[];capture_steps=[]
            model.reset()
            for i in range(50):
                if i==25:model.reset();hist.zero_();hidden.zero_()
                obs=torch.randn(1,45)*.2;priv=torch.randn(1,st.PRIV_DIM)*.2
                frame=i if i<25 else i-25
                if frame%period==0:
                    depth=torch.rand(1,1,48,64)*5.9+.1;capture_steps.append(i)
                height=torch.randn(1,2,64,128)*.2
                height[:,1]=1;task=torch.tensor([i%10]);hist=st.push_history(hist,obs)
                if branch=='teacher':x=torch.cat((obs,priv,height.flatten(1),task[:,None].float()),-1)
                else:x=torch.cat((obs,depth.flatten(1)),-1)
                expected,hidden,_,_=p.step(hist,priv,0 if branch=='teacher' else 1,depth=depth,hidden=hidden,height=height,task_ids=task)
                inputs.append(x[0].numpy());pred.append(model(x)[0].numpy());actual.append(expected[0].numpy())
            error=float(np.max(np.abs(np.array(pred)-np.array(actual))))
            if error>=1e-5:raise RuntimeError(f'{branch} export/reset mismatch {error}')
            model.reset();dst=out/f'{iteration}_{branch}.pt';model.save(str(dst))
            np.savez_compressed(out/f'{iteration}_{branch}_golden.npz',inputs=np.array(inputs),expected=np.array(actual),reset_at=np.array([0,25]))
            meta['models'][branch]={'path':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
                                    'export_max_error':error,'input_dim':TEACHER_INPUT if branch=='teacher' else STUDENT_INPUT,
                                    'golden_depth_capture_steps':capture_steps,'golden_reset_steps':[0,25]}
    (out/f'{iteration}_metadata.json').write_text(json.dumps(meta,indent=2));return meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--iteration',required=True,type=int)
    a=p.parse_args();print(json.dumps(export_checkpoint(a.checkpoint,a.output,a.iteration)))
if __name__=='__main__':main()
