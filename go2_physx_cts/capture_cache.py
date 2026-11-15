"""Bounded direct-index CPU clean-depth cache; no per-frame Python dictionary.

FP16 magnitude is optical depth; sign stores geometric validity (no extra mask).
Noise is regenerated from immutable keys. No trainable latent is cached.
"""
import time
import numpy as np
import torch

class CapturedImages:
    def __init__(self,capacity,device,num_envs=4096):
        self.device=torch.device(device);self.num_envs=num_envs
        self.slots_per_env=max(1,int(capacity)//num_envs);self.capacity=self.slots_per_env*num_envs
        self.data=torch.empty((self.capacity,1,48,64),dtype=torch.float16)
        self.tags=torch.zeros((self.capacity,11),dtype=torch.int64);self.tags[:,0]=-1
        self.hits=self.misses=0;self.stage_rows=2048;self._pending=[]
        self._pools={'h2d':[],'d2h':[]};self._cursor={'h2d':0,'d2h':0}
        self.stream=torch.cuda.Stream(device=self.device) if self.device.type=='cuda' else None
        self.transfer_stats={'h2d_bytes':0,'d2h_bytes':0,'host_wait_seconds':0.}

    @staticmethod
    def keys(pose,scene,frame,episode,trajectory):
        n=len(pose)
        def vector(x):return torch.as_tensor(x,device=pose.device).long().reshape(-1).expand(n)
        bits=pose.detach().float().contiguous().view(torch.int32).long()
        return torch.cat((torch.stack([vector(x) for x in (trajectory,episode,frame,scene)],-1),bits),-1).cpu()

    def _wait(self,item):
        if item.get('event') is not None:
            tick=time.monotonic();item['event'].synchronize();self.transfer_stats['host_wait_seconds']+=time.monotonic()-tick
            item['event']=None

    def _drain(self):
        for item,slots,tags in self._pending:
            self._wait(item)
            # Duplicate ring slots resolve to the last frame, tags and pixels together.
            reverse=slots.numpy()[::-1];_,ix=np.unique(reverse,return_index=True)
            take=torch.from_numpy((len(slots)-1-ix).copy());dest=slots[take]
            self.data.index_copy_(0,dest,item['data'][take]);self.tags.index_copy_(0,dest,tags[take])
        self._pending.clear()

    def _stage(self,kind):
        pool=self._pools[kind];i=self._cursor[kind]%2;self._cursor[kind]+=1
        while len(pool)<=i:pool.append({'data':torch.empty((self.stage_rows,1,48,64),dtype=torch.float16,pin_memory=True),'event':None})
        item=pool[i]
        if kind=='d2h' and item['event'] is not None:self._drain()
        else:self._wait(item)
        return item

    def _upload(self,host):
        if self.stream is None:return host.to(self.device)
        item=self._stage('h2d');item['data'][:len(host)].copy_(host)
        with torch.cuda.stream(self.stream):
            out=item['data'][:len(host)].to(self.device,non_blocking=True);event=torch.cuda.Event();event.record()
        item['event']=event;current=torch.cuda.current_stream(self.device);current.wait_event(event);out.record_stream(current)
        self.transfer_stats['h2d_bytes']+=host.numel()*2
        return out

    def _store(self,values,slots,tags):
        if self.stream is None:
            item={'data':values.cpu(),'event':None};self._pending.append((item,slots,tags));self._drain();return
        for start in range(0,len(slots),self.stage_rows):
            item=self._stage('d2h');piece=values[start:start+self.stage_rows]
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                item['data'][:len(piece)].copy_(piece,non_blocking=True);event=torch.cuda.Event();event.record();piece.record_stream(self.stream)
            item['event']=event;self._pending.append((item,slots[start:start+len(piece)],tags[start:start+len(piece)].clone()))
            self.transfer_stats['d2h_bytes']+=piece.numel()*2

    @torch.no_grad()
    def get(self,render,pose,scene,frame,episode,trajectory,capture_keys=None):
        self._drain();n=len(pose)
        keys=self.keys(pose,scene,frame,episode,trajectory) if capture_keys is None else capture_keys
        slots=keys[:,0].remainder(self.num_envs)*self.slots_per_env+keys[:,2].remainder(self.slots_per_env)
        hit=(self.tags[slots]==keys).all(-1)
        out=torch.empty((n,1,48,64),device=self.device,dtype=torch.float16)
        ids=hit.nonzero().flatten()
        for start in range(0,len(ids),self.stage_rows):
            rows=ids[start:start+self.stage_rows];out[rows.to(self.device)]=self._upload(self.data[slots[rows]])
        missing=(~hit).nonzero().flatten();self.hits+=len(ids);self.misses+=len(missing)
        if len(missing):
            gpu=missing.to(self.device);value=render(pose[gpu]).to(torch.float16)
            out[gpu]=value;self._store(value,slots[missing],keys[missing])
        return out

    @property
    def storage_bytes(self):return self.data.numel()*2+self.tags.numel()*8
    @property
    def staging_bytes(self):return sum(i['data'].numel()*2 for p in self._pools.values() for i in p)
