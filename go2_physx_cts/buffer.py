"""Shared transition replay with terminal labels and episode-safe sequence sampling.

All providers train the same dynamics and provide imagination starting states.
Depth is reconstructed from immutable geometry/pose/key metadata, never copied
from an unrelated future trajectory. Contact-event history is retained separately.
"""
import torch
from . import state as st
from .commands import CMD_STATE_DIM

FLOAT_FIELDS = {
    's':70,'s_next':70,'pose':7,'pose_next':7,'a':12,'a_prev':12,'a_prev2':12,
    'obs':450,'relief':77,'command':3,'timer':1,'cmd_state':CMD_STATE_DIM,
    'xi':st.CONTEXT_DIM,'task_context':21,'task_context_next':21,
    'contact_prob':14,'next_contact_prob':14,'applied_torque':12,
    'next_applied_torque':12,'foot_velocity':12,'next_foot_velocity':12,
    'depth_pose':7,
}
INT_FIELDS = {'source':torch.uint8,'episode':torch.int32,'task_ids':torch.int16,
              'terrain_ids':torch.int16,'scene_ids':torch.int32,'frame_index':torch.int32,
              'depth_frame_index':torch.int32,'depth_scene_ids':torch.int32}
BOOL_FIELDS = ('done','terminated','timeout','success')
HALF_FIELDS = ('obs','relief')

class Replay:
    def __init__(self,n,length,device):
        self.n,self.L,self.device=n,length,device
        self.ptr=self.size=0
        self.total_collections=0
        self.data={k:torch.zeros((n,length,d),device=device,dtype=torch.float16 if k in HALF_FIELDS else torch.float32) for k,d in FLOAT_FIELDS.items()}
        self.data.update({k:torch.zeros((n,length,1),device=device,dtype=d) for k,d in INT_FIELDS.items()})
        self.data.update({k:torch.zeros((n,length,1),device=device,dtype=torch.bool) for k in BOOL_FIELDS})
        self.event_history=torch.zeros((n,length),device=device,dtype=torch.bool)

    def add(self,**values):
        if set(values)!=set(self.data):
            raise ValueError(f'Replay schema mismatch missing={set(self.data)-set(values)} extra={set(values)-set(self.data)}')
        for key,dst in self.data.items():
            dst[:,self.ptr]=values[key].reshape(self.n,dst.shape[-1])
        self.event_history[:,self.ptr]=False
        # Includes failures, successful completion, support loss and body contact.
        contact=values['next_contact_prob']
        event=values['terminated'].reshape(-1).bool()|values['success'].reshape(-1).bool()
        event|=(contact[:,:4].sum(-1)<.5)|(contact[:,4:6].amax(-1)>.5)
        ids=event.nonzero().flatten()
        if len(ids):
            k=min(100,self.size+1,self.L)
            slots=(self.ptr-torch.arange(k,device=self.device))%self.L
            same=self.data['episode'][ids[:,None],slots[None,:],0]==values['episode'].reshape(-1)[ids,None]
            self.event_history[ids[:,None],slots[None,:]]|=same
        self.ptr=(self.ptr+1)%self.L
        self.size=min(self.size+1,self.L)
        self.total_collections+=1

    def physical(self,t):
        return (self.ptr-self.size+t)%self.L

    def sample_states(self,batch,source=None):
        if not self.size:raise ValueError('Empty Replay')
        env=torch.randint(self.n,(batch,),device=self.device)
        logical=torch.randint(self.size,(batch,),device=self.device)
        if source is not None:
            for _ in range(32):
                bad=self.data['source'][env,self.physical(logical),0]!=source
                if not bad.any():break
                count=int(bad.sum())
                env[bad]=torch.randint(self.n,(count,),device=self.device)
                logical[bad]=torch.randint(self.size,(count,),device=self.device)
            bad=self.data['source'][env,self.physical(logical),0]!=source
            if bad.any():
                chronological=self.physical(torch.arange(self.size,device=self.device))
                eligible=(self.data['source'][:,chronological,0]==source).nonzero()
                if not len(eligible):raise ValueError('Requested source has no replay samples')
                pick=eligible[torch.randint(len(eligible),(int(bad.sum()),),device=self.device)]
                env[bad]=pick[:,0];logical[bad]=pick[:,1]
        return env,self.physical(logical)

    def _gather(self,e,slots):
        ei=e if slots.ndim==1 else e[:,None]
        out={k:v[ei,slots].float() if k in FLOAT_FIELDS else v[ei,slots] for k,v in self.data.items()}
        out['env']=e
        out['trajectory_ids']=e
        return out

    def seeds(self,batch,xi=None,source=None,burnin=32):
        e,t=self.sample_states(batch,source)
        out=self._gather(e,t)
        if burnin:
            logical=(t-(self.ptr-self.size))%self.L
            before=logical[:,None]-torch.arange(burnin,0,-1,device=self.device)[None,:]
            slots=self.physical(before.clamp_min(0))
            prefix=self._gather(e,slots)
            valid=(before>=0)&(prefix['episode'][...,0]==out['episode'][:,0,None])
            prefix['valid']=valid
            prefix['reset']=~valid
            prefix['reset'][:,0]=True
            out['prefix']=prefix
        return out

    def sequences(self,batch,loss_steps=64,burnin=32,source=None):
        total=loss_steps+burnin
        if self.size<total:raise ValueError('Replay needs a complete student sequence')
        e=torch.randint(self.n,(batch,),device=self.device)
        first=torch.randint(self.size-total+1,(batch,),device=self.device)
        if source is not None:
            for _ in range(32):
                bad=self.data['source'][e,self.physical(first+total-1),0]!=source
                if not bad.any():break
                count=int(bad.sum());e[bad]=torch.randint(self.n,(count,),device=self.device)
                first[bad]=torch.randint(self.size-total+1,(count,),device=self.device)
            bad=self.data['source'][e,self.physical(first+total-1),0]!=source
            if bad.any():
                ends=self.physical(torch.arange(total-1,self.size,device=self.device))
                eligible=(self.data['source'][:,ends,0]==source).nonzero()
                if not len(eligible):raise ValueError('Requested source has no complete sequences')
                pick=eligible[torch.randint(len(eligible),(int(bad.sum()),),device=self.device)]
                e[bad]=pick[:,0];first[bad]=pick[:,1]
        slots=self.physical(first[:,None]+torch.arange(total,device=self.device))
        out=self._gather(e,slots)
        reset=torch.ones((batch,total),device=self.device,dtype=torch.bool)
        reset[:,1:]=out['episode'][:,1:,0]!=out['episode'][:,:-1,0]
        out['reset']=reset
        out['valid']=torch.ones_like(reset)
        out['loss_mask']=torch.arange(total,device=self.device)[None,:].expand(batch,-1)>=burnin
        return out

    def windows(self,batch,k,xi=None,event=False):
        if self.size<k:raise ValueError('Not enough contiguous history')
        e=torch.randint(self.n,(batch,),device=self.device)
        start=torch.randint(self.size-k+1,(batch,),device=self.device)
        if event:
            # Bounded rejection avoids a full multi-million-row scan per update.
            for _ in range(8):
                bad=~self.event_history[e,self.physical(start)]
                if not bad.any():break
                count=int(bad.sum());e[bad]=torch.randint(self.n,(count,),device=self.device)
                start[bad]=torch.randint(self.size-k+1,(count,),device=self.device)
        idx=self.physical(start[:,None]+torch.arange(k,device=self.device));ee=e[:,None]
        required=('s','pose','s_next','pose_next','a','a_prev','xi','relief','command',
                  'episode','done','next_contact_prob','next_applied_torque',
                  'next_foot_velocity','scene_ids','task_ids')
        raw={name:self.data[name][ee,idx] for name in required}
        same=raw['episode'][...,0]==raw['episode'][:,0,0,None]
        # A final terminal transition IS a valid physical label; only later rows
        # after it or another reset are masked out.
        prior=torch.cat((torch.ones((batch,1),device=self.device),
                         (~raw['done'][:,:-1,0]).float().cumprod(1)),1)
        valid=same.float()*prior
        out={'s':raw['s'],'pose':raw['pose'],
             's_next':raw['s_next'],'pose_next':raw['pose_next'],
             'a':raw['a'],'a_prev':raw['a_prev'],'xi':raw['xi'][:,0],
             'relief':raw['relief'].float(),'command':raw['command'][:,0],
             'valid':valid,'done':raw['done'][...,0],
             'contact_prob':raw['next_contact_prob'],
             'applied_torque':raw['next_applied_torque'],
             'foot_velocity':raw['next_foot_velocity'],
             'scene_ids':raw['scene_ids'][...,0],
             'task_ids':raw['task_ids'][...,0],
             'event_history':self.event_history[ee,idx]}
        return out

    @property
    def storage_bytes(self):
        return sum(x.numel()*x.element_size() for x in self.data.values())+self.event_history.numel()

class Archive:
    def __init__(self,capacity,k,device):
        self.cap,self.device,self.ptr,self.size=capacity,device,0,0
        self.data={}
        self.admission_tick=torch.zeros(capacity,device=device,dtype=torch.long)
        self.total_admitted=0

    def add(self,w,tick=0):
        n=min(len(w['s']),self.cap)
        if not n:return
        if not self.data:
            self.data={key:torch.empty((self.cap,*value.shape[1:]),device=self.device,dtype=value.dtype) for key,value in w.items()}
        if set(w)!=set(self.data):raise ValueError('Archive schema changed')
        idx=(self.ptr+torch.arange(n,device=self.device))%self.cap
        for key in self.data:self.data[key][idx]=w[key][:n]
        self.admission_tick[idx]=int(tick);self.total_admitted+=n
        self.ptr=(self.ptr+n)%self.cap;self.size=min(self.size+n,self.cap)

    def sample(self,n):
        if not self.size:raise ValueError('Empty Archive')
        idx=torch.randint(self.size,(n,),device=self.device)
        return {k:v[idx] for k,v in self.data.items()}

    def state_dict(self):
        return {'ptr':self.ptr,'size':self.size,'data':{k:v[:self.size].cpu() for k,v in self.data.items()},
                'admission_tick':self.admission_tick[:self.size].cpu(),'total_admitted':self.total_admitted}

    def statistics(self,tick):
        if not self.size:return {'size':0,'total_admitted':self.total_admitted}
        age=(int(tick)-self.admission_tick[:self.size]).float()
        return {'size':self.size,'total_admitted':self.total_admitted,
                'oldest_age_collections':float(age.max()),'median_age_collections':float(age.median())}


def rarity(w):
    contact=w['contact_prob'][...,:4]
    valid=w['valid']
    switches=((contact[:,1:]-contact[:,:-1]).abs().sum(-1)*valid[:,1:]*valid[:,:-1]).sum(-1)
    return switches+((contact.sum(-1)<.5)*valid).sum(-1)+(w['done']*valid).sum(-1)*4+(w['event_history']*valid).sum(-1)/valid.sum(-1).clamp_min(1)*4


def archive_worthy(w):
    valid=w['valid'].bool()
    return ((w['event_history']|w['done']|(w['contact_prob'][...,:4].sum(-1)<.5))&valid).any(-1)


def mixed_windows(buf,archive,cfg,xi=None):
    nprio=int(cfg.wm_batch*cfg.wm_prioritized_frac)
    narch=int(cfg.wm_batch*cfg.wm_archive_frac) if archive.size else 0
    parts=[buf.windows(cfg.wm_batch-nprio-narch,cfg.wm_window,xi)]
    # Pinned source draws uniform candidates, then prioritizes rare windows.
    # Biasing the candidates themselves would also inflate archive admissions.
    cand=buf.windows(4*max(nprio,1),cfg.wm_window,xi)
    score=rarity(cand)
    keep=archive_worthy(cand)&(torch.rand_like(score)<cfg.archive_keep_prob)
    archive.add({k:v[keep] for k,v in cand.items()},tick=buf.total_collections)
    if nprio:
        top=(score+1e-3*torch.rand_like(score)).topk(nprio).indices
        parts.append({k:v[top] for k,v in cand.items()})
    if narch:parts.append(archive.sample(narch))
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
