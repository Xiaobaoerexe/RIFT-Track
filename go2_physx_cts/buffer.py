"""Shared transition replay with terminal labels and episode-safe sequence sampling.

All providers train the same dynamics and provide imagination starting states.
Depth capture metadata addresses the observation module's bounded frozen-image
cache. Evicted images reconstruct from the exact immutable pose/noise key; future
imagined observations always use their own predicted pose. Contact-event history
is retained separately.
"""
import time
import torch
from . import state as st
from .commands import CMD_STATE_DIM
from .task import TASK_CONTEXT_DIM

FLOAT_FIELDS = {
    's':70,'s_next':70,'pose':7,'pose_next':7,'a':12,'a_prev':12,'a_prev2':12,
    'obs':450,'relief':77,'command':3,'command_next':3,'timer':1,'cmd_state':CMD_STATE_DIM,'cmd_state_next':CMD_STATE_DIM,
    'xi':st.CONTEXT_DIM,'task_context':TASK_CONTEXT_DIM,'task_context_next':TASK_CONTEXT_DIM,
    'contact_prob':14,'next_contact_prob':14,'applied_torque':12,
    'next_applied_torque':12,'foot_velocity':12,'next_foot_velocity':12,
    'joint_acceleration':12,'next_joint_acceleration':12,
    'log_contact_force':8,'next_log_contact_force':8,
    'continuous_reward':1,'event_reward':1,
    'depth_pose':7,
}
INT_FIELDS = {'source':torch.uint8,'episode':torch.int32,'task_ids':torch.int16,
              'terrain_ids':torch.int16,'scene_ids':torch.int32,'frame_index':torch.int32,
              'depth_frame_index':torch.int32,'depth_scene_ids':torch.int32}
BOOL_FIELDS = ('done','terminated','timeout','success')
HALF_FIELDS = ('obs','relief')

def _device(device):
    device=torch.device(device)
    if device.type=='cuda' and device.index is None:
        return torch.device('cuda',torch.cuda.current_device())
    return device


class ReplayPartitionUnavailable(ValueError):
    """No eligible replay positions exist, rather than random rejection failure."""


class PackedTransport:
    """Lossless batched transfer with one bounded pinned staging allocation.

    Fields retain their exact storage dtype.8-byte alignment permits byte-pack
    views to be reinterpreted on either device. A dedicated transfer stream and
    event protect staging reuse; no Replay view survives an asynchronous copy.
    """
    def __init__(self,staging_bytes=16*1024*1024):
        self.staging_bytes=int(staging_bytes);self._pinned=None;self._pending=None;self._streams={}
        self.bytes_h2d=self.bytes_d2h=self.calls_h2d=self.calls_d2h=0
        self.wait_seconds=0.;self.pack_seconds=0.

    def _wait(self):
        if self._pending is not None:
            tick=time.perf_counter();self._pending.synchronize();self.wait_seconds+=time.perf_counter()-tick
            self._pending=None

    def _stage(self):
        self._wait()
        if self._pinned is None:self._pinned=torch.empty(self.staging_bytes,dtype=torch.uint8,pin_memory=True)
        return self._pinned

    def _stream(self,device):
        key=str(torch.device(device))
        if key not in self._streams:self._streams[key]=torch.cuda.Stream(device=device)
        return self._streams[key]

    def move(self,values,target):
        if not values:return values
        target=_device(target);source=next(iter(values.values())).device
        if source==target:return values
        if any(value.device!=source for value in values.values()):raise ValueError('Packed transfer requires one source device')
        if source.type!='cuda' and target.type!='cuda':return {key:value.to(target) for key,value in values.items()}
        tick=time.perf_counter();parts=[];specs=[];offset=0
        for key,value in values.items():
            raw=value.detach().contiguous().reshape(-1).view(torch.uint8)
            count=raw.numel();specs.append((key,value.dtype,tuple(value.shape),offset,count));parts.append(raw);offset+=count
            padding=(-offset)%8
            if padding:parts.append(torch.zeros(padding,dtype=torch.uint8,device=source));offset+=padding
        packed=torch.cat(parts) if parts else torch.empty(0,dtype=torch.uint8,device=source)
        self.pack_seconds+=time.perf_counter()-tick
        if source.type=='cuda' and target.type=='cuda':
            moved=packed.to(target)
        else:
            cuda=source if source.type=='cuda' else target
            stream=self._stream(cuda);current=torch.cuda.current_stream(cuda)
            moved=torch.empty(offset,dtype=torch.uint8,device=target)
            stream.wait_stream(current)
            for first in range(0,offset,self.staging_bytes):
                amount=min(self.staging_bytes,offset-first);stage=self._stage()[:amount]
                if source.type=='cpu':
                    stage.copy_(packed[first:first+amount])
                    with torch.cuda.stream(stream):
                        moved[first:first+amount].copy_(stage,non_blocking=True)
                        self._pending=torch.cuda.Event();self._pending.record(stream)
                else:
                    with torch.cuda.stream(stream):
                        stage.copy_(packed[first:first+amount],non_blocking=True)
                        self._pending=torch.cuda.Event();self._pending.record(stream)
                    self._wait();moved[first:first+amount].copy_(stage)
            if source.type=='cpu':
                if self._pending is not None:current.wait_event(self._pending)
                moved.record_stream(stream);self.bytes_h2d+=offset;self.calls_h2d+=1
            else:
                self.bytes_d2h+=offset;self.calls_d2h+=1
        return {key:moved[offset:offset+count].view(dtype).reshape(shape) for key,dtype,shape,offset,count in specs}

    def metadata(self):
        return {'staging_limit_bytes':self.staging_bytes,'pinned_bytes':0 if self._pinned is None else self._pinned.numel(),
            'h2d_bytes':self.bytes_h2d,'d2h_bytes':self.bytes_d2h,'h2d_calls':self.calls_h2d,'d2h_calls':self.calls_d2h,
            'staging_wait_seconds':self.wait_seconds,'pack_seconds':self.pack_seconds}


class Replay:
    def __init__(self,n,length,device,holdout_modulus=32,storage_device=None):
        self.n,self.L,self.device=n,length,_device(device)
        self.storage_device=_device(device if storage_device is None else storage_device)
        self.index_device=self.storage_device;self.transport=PackedTransport()
        self.holdout_modulus=holdout_modulus
        self.heldout=torch.zeros((n,length),device=self.storage_device,dtype=torch.bool)
        self.ptr=self.size=0
        self.total_collections=0
        self.sampling_fallbacks=0;self.last_sampling_fallback=None
        self.data={k:torch.zeros((n,length,d),device=self.storage_device,dtype=torch.float16 if k in HALF_FIELDS else torch.float32) for k,d in FLOAT_FIELDS.items()}
        self.data.update({k:torch.zeros((n,length,1),device=self.storage_device,dtype=d) for k,d in INT_FIELDS.items()})
        self.data.update({k:torch.zeros((n,length,1),device=self.storage_device,dtype=torch.bool) for k in BOOL_FIELDS})
        self.event_history=torch.zeros((n,length),device=self.storage_device,dtype=torch.bool)

    def add(self,**values):
        if set(values)!=set(self.data):
            raise ValueError(f'Replay schema mismatch missing={set(self.data)-set(values)} extra={set(values)-set(self.data)}')
        if self.storage_device!=torch.device(self.device):
            values=self.transport.move({key:value.reshape(self.n,self.data[key].shape[-1]).to(dtype=self.data[key].dtype) for key,value in values.items()},self.storage_device)
        for key,dst in self.data.items():
            dst[:,self.ptr]=values[key].reshape(self.n,dst.shape[-1])
        self.heldout[:,self.ptr]=self.validation_episodes(values['episode'])
        self.event_history[:,self.ptr]=False
        # Includes failures, successful completion, support loss and body contact.
        contact=values['next_contact_prob']
        event=values['terminated'].reshape(-1).bool()|values['success'].reshape(-1).bool()
        event|=(contact[:,:4].sum(-1)<.5)|(contact[:,4:6].amax(-1)>.5)
        ids=event.nonzero().flatten()
        if len(ids):
            k=min(100,self.size+1,self.L)
            slots=(self.ptr-torch.arange(k,device=self.index_device))%self.L
            same=self.data['episode'][ids[:,None],slots[None,:],0]==values['episode'].reshape(-1)[ids,None]
            self.event_history[ids[:,None],slots[None,:]]|=same
        self.ptr=(self.ptr+1)%self.L
        self.size=min(self.size+1,self.L)
        self.total_collections+=1

    def physical(self,t):
        return (self.ptr-self.size+t)%self.L

    def validation_episodes(self,episode):
        env=torch.arange(self.n,device=episode.device,dtype=torch.int64)
        return ((env*73856093+episode.reshape(-1).long()*19349663)%self.holdout_modulus)==0

    def _positions(self,batch,length,source=None,validation=False,whole=False):
        if self.size<length:raise ValueError('Replay lacks a complete window')
        selected_e=[];selected_t=[];remaining=batch
        for _ in range(16):
            count=max(remaining*(self.holdout_modulus*2 if validation else 2),32)
            e=torch.randint(self.n,(count,),device=self.device)
            t=torch.randint(self.size-length+1,(count,),device=self.device)
            if e.device!=self.index_device:
                indices=self.transport.move({'e':e,'t':t},self.index_device);e,t=indices['e'],indices['t']
            ok=self.heldout[e,self.physical(t)]==validation
            if whole and length>1:
                slots=self.physical(t[:,None]+torch.arange(length,device=self.index_device))
                ok&=(self.heldout[e[:,None],slots]==validation).all(-1)
            if source is not None:
                ok&=self.data['source'][e,self.physical(t+length-1),0]==source
            ids=ok.nonzero().flatten()[:remaining]
            selected_e.append(e[ids]);selected_t.append(t[ids]);remaining-=len(ids)
            if not remaining:break
        if remaining:
            e,t=self._exact_positions(remaining,length,source,validation,whole)
            selected_e.append(e);selected_t.append(t)
        return torch.cat(selected_e),torch.cat(selected_t)

    def _exact_positions(self,batch,length,source,validation,whole):
        """Rare fallback only. Keep the original uniform-with-replacement law.

        Enumerate metadata, never replay payloads or length-expanded windows.
        Prefix counts implement the existing all-heldout sequence predicate in
        O(n*size) memory, including when the physical ring wraps.
        """
        starts=torch.arange(self.size-length+1,device=self.index_device)
        if whole and length>1:
            slots=self.physical(torch.arange(self.size,device=self.index_device))
            bad=self.heldout[:,slots]!=validation
            prefix=torch.cat((torch.zeros((self.n,1),device=self.index_device,dtype=torch.int32),
                              bad.cumsum(1,dtype=torch.int32)),1)
            eligible=(prefix[:,length:]-prefix[:,:-length])==0
        else:
            eligible=self.heldout[:,self.physical(starts)]==validation
        if source is not None:
            eligible&=self.data['source'][:,self.physical(starts+length-1),0]==source
        flat=eligible.reshape(-1).nonzero().flatten()
        self.sampling_fallbacks+=1
        self.last_sampling_fallback={'requested':batch,'available_positions':len(flat),
            'length':length,'source':source,'validation':bool(validation),'whole':bool(whole)}
        if not len(flat):
            raise ReplayPartitionUnavailable('No eligible replay positions: '+str(self.last_sampling_fallback))
        pick=torch.randint(len(flat),(batch,),device=self.device).to(self.index_device)
        chosen=flat[pick];width=len(starts)
        return chosen//width,chosen%width

    def sample_states(self,batch,source=None,validation=False):
        e,t=self._positions(batch,1,source,validation)
        return e,self.physical(t)

    def _gather(self,e,slots):
        if e.device!=self.index_device or slots.device!=self.index_device:
            indices=self.transport.move({'e':e,'slots':slots},self.index_device);e,slots=indices['e'],indices['slots']
        ei=e if slots.ndim==1 else e[:,None]
        out={key:value[ei,slots] for key,value in self.data.items()}
        out['env']=e;out['trajectory_ids']=e
        out=self.transport.move(out,self.device)
        return {key:value.float() if key in FLOAT_FIELDS else value for key,value in out.items()}

    def event_mask(self,e,slots):
        if e.device!=self.index_device or slots.device!=self.index_device:
            indices=self.transport.move({'e':e,'slots':slots},self.index_device);e,slots=indices['e'],indices['slots']
        return self.event_history[e,slots]

    def metadata(self):
        return {'storage_device':str(self.storage_device),'target_device':str(self.device),'storage_bytes':self.storage_bytes,
            'transport':self.transport.metadata()}

    def seeds(self,batch,xi=None,source=None,burnin=32,validation=False):
        e,t=self.sample_states(batch,source,validation)
        out=self._gather(e,t)
        if burnin:
            logical=(t-(self.ptr-self.size))%self.L
            before=logical[:,None]-torch.arange(burnin,0,-1,device=self.index_device)[None,:]
            slots=self.physical(before.clamp_min(0))
            prefix=self._gather(e,slots)
            valid=(before>=0).to(self.device)&(prefix['episode'][...,0]==out['episode'][:,0,None])
            prefix['valid']=valid
            prefix['reset']=~valid
            prefix['reset'][:,0]=True
            out['prefix']=prefix
        return out

    def sequences(self,batch,loss_steps=64,burnin=32,source=None,validation=False):
        total=loss_steps+burnin
        e,first=self._positions(batch,total,source,validation,whole=True)
        slots=self.physical(first[:,None]+torch.arange(total,device=self.index_device))
        out=self._gather(e,slots)
        reset=torch.ones((batch,total),device=self.device,dtype=torch.bool)
        reset[:,1:]=out['episode'][:,1:,0]!=out['episode'][:,:-1,0]
        out['reset']=reset
        out['valid']=torch.ones_like(reset)
        out['loss_mask']=torch.arange(total,device=self.device)[None,:].expand(batch,-1)>=burnin
        return out

    def windows(self,batch,k,xi=None,event=False,validation=False):
        e,start=self._positions(batch,k,validation=validation)
        if event:
            for _ in range(8):
                bad=~self.event_history[e,self.physical(start)]
                if not bad.any():break
                ne,nt=self._positions(int(bad.sum()),k,validation=validation)
                e[bad]=ne;start[bad]=nt
        idx=self.physical(start[:,None]+torch.arange(k,device=self.index_device));ee=e[:,None]
        required=('s','pose','s_next','pose_next','a','a_prev','xi','relief','command',
                  'episode','done','next_contact_prob','next_applied_torque',
                  'next_foot_velocity','scene_ids','task_ids','next_joint_acceleration','next_log_contact_force',
                  'frame_index','depth_frame_index','depth_scene_ids','depth_pose')
        raw={name:self.data[name][ee,idx] for name in required}
        raw.update(_event_history=self.event_history[ee,idx],_validation=self.heldout[e,self.physical(start)],_trajectory_ids=e)
        raw=self.transport.move(raw,self.device)
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
             'joint_acceleration':raw['next_joint_acceleration'],
             'log_contact_force':raw['next_log_contact_force'],
             'frame_index':raw['frame_index'],'episode':raw['episode'],
             'depth_frame_index':raw['depth_frame_index'],'depth_scene_ids':raw['depth_scene_ids'],
             'depth_pose':raw['depth_pose'],
             'scene_ids':raw['scene_ids'][...,0],
             'task_ids':raw['task_ids'][...,0],
             'event_history':raw['_event_history'],
             'trajectory_ids':raw['_trajectory_ids'],'episode_ids':raw['episode'][:,0,0],
             'validation':raw['_validation']}
        return out

    @property
    def storage_bytes(self):
        return sum(x.numel()*x.element_size() for x in self.data.values())+self.event_history.numel()+self.heldout.numel()

class Archive:
    def __init__(self,capacity,k,device,storage_device=None):
        self.cap,self.device,self.ptr,self.size=capacity,_device(device),0,0
        self.storage_device=_device(device if storage_device is None else storage_device)
        self.index_device=self.storage_device;self.transport=PackedTransport()
        self.data={}
        self.admission_tick=torch.zeros(capacity,device=self.storage_device,dtype=torch.long)
        self.total_admitted=0

    def add(self,w,tick=0):
        if w.get('validation') is not None and bool(w['validation'].any()):
            raise ValueError('Held-out episodes must never enter the training archive')
        n=min(len(w['s']),self.cap)
        if not n:return
        if not self.data:
            self.data={key:torch.empty((self.cap,*value.shape[1:]),device=self.storage_device,dtype=value.dtype) for key,value in w.items()}
        if set(w)!=set(self.data):raise ValueError('Archive schema changed')
        idx=(self.ptr+torch.arange(n,device=self.index_device))%self.cap
        data=self.transport.move({key:value[:n] for key,value in w.items()},self.storage_device)
        for key in self.data:self.data[key][idx]=data[key]
        self.admission_tick[idx]=int(tick);self.total_admitted+=n
        self.ptr=(self.ptr+n)%self.cap;self.size=min(self.size+n,self.cap)

    def sample(self,n):
        if not self.size:raise ValueError('Empty Archive')
        idx=torch.randint(self.size,(n,),device=self.device)
        if idx.device!=self.index_device:idx=self.transport.move({'idx':idx},self.index_device)['idx']
        return self.transport.move({k:v[idx] for k,v in self.data.items()},self.device)

    @property
    def storage_bytes(self):return sum(v.numel()*v.element_size() for v in self.data.values())+self.admission_tick.numel()*self.admission_tick.element_size()

    def metadata(self):return {'storage_device':str(self.storage_device),'target_device':str(self.device),'storage_bytes':self.storage_bytes,'transport':self.transport.metadata()}

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
