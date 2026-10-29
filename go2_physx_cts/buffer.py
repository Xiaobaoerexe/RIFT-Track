"""One shared online replay for all providers and both actor branches.

Window selection/archive recipe follows pinned upstream train.py. Episode-end
transitions are excluded as in upstream; simulator reset jumps never become labels.
"""
import torch
from . import state as st
from .commands import CMD_STATE_DIM

class Replay:
    def __init__(self,n,length,device):
        self.n,self.L,self.device=n,length,device;self.ptr=self.size=0
        self.data={}
        for name,dim,dtype in [('s',70,torch.float32),('pose',3,torch.float32),('a',12,torch.float32),
                              ('a_prev',12,torch.float32),('obs',450,torch.float16),('relief',77,torch.float16),
                              ('command',3,torch.float32),('timer',1,torch.float32),('cmd_state',CMD_STATE_DIM,torch.float32),('done',1,torch.bool),
                              ('source',1,torch.uint8),('episode',1,torch.int32),('xi',st.CONTEXT_DIM,torch.float32)]:
            self.data[name]=torch.zeros((n,length,dim),device=device,dtype=dtype)
    def add(self,**values):
        if set(values)!=set(self.data):raise ValueError('Replay schema mismatch')
        for key,dst in self.data.items():dst[:,self.ptr]=values[key].reshape(dst.shape[0],dst.shape[-1])
        self.ptr=(self.ptr+1)%self.L;self.size=min(self.size+1,self.L)
    def physical(self,t):return (self.ptr-self.size+t)%self.L
    def sample_states(self,batch,source=None):
        if source is None:
            env=torch.randint(self.n,(batch,),device=self.device)
            slot=self.physical(torch.randint(self.size,(batch,),device=self.device))
        else:
            # Providers are fixed per environment, avoiding a4M-row scan every CTS minibatch.
            candidates=(self.data['source'][:,self.physical(0),0]==source).nonzero().flatten()
            if not len(candidates):raise ValueError('Requested replay source has no samples')
            env=candidates[torch.randint(len(candidates),(batch,),device=self.device)]
            slot=self.physical(torch.randint(self.size,(batch,),device=self.device))
        return env,slot
    def seeds(self,batch,xi,source=None):
        e,t=self.sample_states(batch,source)
        out={k:v[e,t].float() for k,v in self.data.items()}
        out['env']=e
        return out
    def windows(self,batch,k,xi):
        if self.size<=k+1:raise ValueError('Not enough contiguous history')
        e=torch.randint(self.n,(batch,),device=self.device)
        t=torch.randint(self.size-k,(batch,),device=self.device)
        idx=self.physical(t[:,None]+torch.arange(k+1,device=self.device));ee=e[:,None]
        done=self.data['done'][ee,idx[:,:k],0].float()
        valid=(1-done).cumprod(1)
        # Explicit episode identity catches any accidental writer boundary mismatch.
        ep=self.data['episode'][ee,idx,0]
        valid*=((ep[:,1:]==ep[:,:1]).float()).cumprod(1)
        return {'s':self.data['s'][ee,idx],'pose':self.data['pose'][ee,idx],
                'a':self.data['a'][ee,idx[:,:k]],'a_prev':self.data['a_prev'][ee,idx[:,:k]],
                'relief':self.data['relief'][ee,idx[:,:k]],'xi':self.data['xi'][e,idx[:,0]],
                'command':self.data['command'][e,idx[:,0]],'valid':valid}
    @property
    def storage_bytes(self):return sum(x.numel()*x.element_size() for x in self.data.values())

class Archive:
    def __init__(self,capacity,k,device):
        self.cap,self.device,self.ptr,self.size=capacity,device,0,0
        shapes={'s':(k+1,70),'pose':(k+1,3),'a':(k,12),'a_prev':(k,12),'xi':(st.CONTEXT_DIM,), 'command':(3,), 'valid':(k,),'relief':(k,77)}
        self.data={key:torch.zeros((capacity,*shape),device=device,dtype=torch.float16 if key=='relief' else torch.float32) for key,shape in shapes.items()}
    def add(self,w):
        n=min(len(w['s']),self.cap)
        if not n:return
        idx=(self.ptr+torch.arange(n,device=self.device))%self.cap
        for k in self.data:self.data[k][idx]=w[k][:n]
        self.ptr=(self.ptr+n)%self.cap;self.size=min(self.size+n,self.cap)
    def sample(self,n):
        idx=torch.randint(self.size,(n,),device=self.device)
        return {k:v[idx] for k,v in self.data.items()}

def rarity(w):
    s=w['s'];contact=(st.feet_world_z(s.reshape(-1,70))<.035).reshape(s.shape[0],-1,4).float()
    return (contact[:,1:]-contact[:,:-1]).abs().sum((1,2))+(contact.sum(-1)==0).float().sum(-1)+(w['valid'][:,-1]<.5).float()*4

def archive_worthy(w):
    s=w['s'];feet=st.feet_world_z(s.reshape(-1,70)).reshape(s.shape[0],-1,4)
    return (((feet>=.035).all(-1).float().sum(-1))>=2)|(s[:,:,6].abs().amax(-1)>.6)|(w['valid'][:,-1]<.5)

def mixed_windows(buf,archive,cfg,xi):
    nprio=int(cfg.wm_batch*cfg.wm_prioritized_frac)
    narch=int(cfg.wm_batch*cfg.wm_archive_frac) if archive.size else 0
    parts=[buf.windows(cfg.wm_batch-nprio-narch,cfg.wm_window,xi)]
    cand=buf.windows(4*max(nprio,1),cfg.wm_window,xi);score=rarity(cand)
    keep=archive_worthy(cand)&(torch.rand_like(score)<cfg.archive_keep_prob)
    archive.add({k:v[keep] for k,v in cand.items()})
    if nprio:
        top=(score+1e-3*torch.rand_like(score)).topk(nprio).indices
        parts.append({k:v[top] for k,v in cand.items()})
    if narch:parts.append(archive.sample(narch))
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
