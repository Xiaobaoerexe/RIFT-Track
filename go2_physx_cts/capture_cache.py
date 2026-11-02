"""CPU storage for frozen historical images, with bounded asynchronous staging.

No future observation or trainable feature is cached here. Keys retain the
trajectory, episode, frame, scene and exact capture pose of the original cache.
"""
from collections import OrderedDict
import time
import torch


class CapturedImages:
    def __init__(self,capacity,device,block_size=4096):
        self.capacity=int(capacity);self.device=torch.device(device)
        self.block_size=min(block_size,self.capacity)
        self.entries=OrderedDict();self.blocks=[];self.hits=0;self.misses=0
        self.stage_rows=2048;self._pending=[];self._pools={'h2d':[],'d2h':[]};self._cursor={'h2d':0,'d2h':0}
        self.stream=torch.cuda.Stream(device=self.device) if self.device.type=='cuda' else None
        self.transfer_stats={'h2d_bytes':0,'d2h_bytes':0,'host_wait_seconds':0.,'h2d_ms':0.,'d2h_ms':0.}

    def _storage(self,slot):
        block,index=divmod(slot,self.block_size)
        while len(self.blocks)<=block:
            remaining=self.capacity-len(self.blocks)*self.block_size
            self.blocks.append(torch.empty((min(self.block_size,remaining),1,48,64),dtype=torch.float16))
        return self.blocks[block],index

    @staticmethod
    def keys(pose,scene,frame,episode,trajectory):
        n=len(pose)
        def vector(x):return torch.as_tensor(x,device=pose.device).long().reshape(-1).expand(n)
        vectors=list(map(vector,(trajectory,episode,frame,scene)))
        ids=torch.stack(vectors,-1).cpu().tolist()
        poses=pose.detach().float().contiguous().cpu().numpy()
        return [(*row,poses[i].tobytes()) for i,row in enumerate(ids)]

    def _wait(self,item,kind):
        event=item.get('end')
        if event is not None:
            tick=time.monotonic();event.synchronize()
            self.transfer_stats['host_wait_seconds']+=time.monotonic()-tick
            self.transfer_stats[kind+'_ms']+=item['begin'].elapsed_time(event)
            item['end']=None

    def _drain(self):
        # Complete older writes before any slot is read, evicted or reused.
        for item,writes in self._pending:
            self._wait(item,'d2h')
            for block,rows in writes.items():
                dst,src=zip(*rows)
                self.blocks[block].index_copy_(0,torch.tensor(dst),item['data'][list(src)])
        self._pending.clear()

    def _stage(self,kind):
        pool=self._pools[kind];index=self._cursor[kind]%2;self._cursor[kind]+=1
        while len(pool)<=index:
            pool.append({'data':torch.empty((self.stage_rows,1,48,64),dtype=torch.float16,pin_memory=True),'end':None})
        item=pool[index]
        if kind=='d2h' and item.get('end') is not None:self._drain()
        else:self._wait(item,kind)
        return item

    def _upload(self,host):
        if self.stream is None:return host.float()
        item=self._stage('h2d');n=len(host);item['data'][:n].copy_(host)
        with torch.cuda.stream(self.stream):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            value=item['data'][:n].to(self.device,non_blocking=True)
            end.record()
        item.update(begin=begin,end=end)
        current=torch.cuda.current_stream(self.device);current.wait_event(end);value.record_stream(current)
        self.transfer_stats['h2d_bytes']+=host.numel()*host.element_size()
        return value.float()

    def _store(self,stored,writes):
        # Flatten only metadata. GPU/CPU image work remains batch operations.
        rows=[(block,dst,src) for block,entries in writes.items() for dst,src in entries.items()]
        if self.stream is None:
            for block,entries in writes.items():
                dst,src=zip(*entries.items());self.blocks[block][list(dst)]=stored[list(src)].cpu()
            return
        for lo in range(0,len(rows),self.stage_rows):
            batch=rows[lo:lo+self.stage_rows];item=self._stage('d2h')
            values=stored[[row[2] for row in batch]]
            produced=torch.cuda.Event();produced.record()
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(produced)
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                item['data'][:len(batch)].copy_(values,non_blocking=True);end.record()
                values.record_stream(self.stream)
            item.update(begin=begin,end=end)
            mapping={}
            for j,(block,dst,_) in enumerate(batch):mapping.setdefault(block,[]).append((dst,j))
            self._pending.append((item,mapping))
            self.transfer_stats['d2h_bytes']+=values.numel()*values.element_size()

    @torch.no_grad()
    def get(self,render,pose,scene,frame,episode,trajectory,capture_keys=None):
        self._drain()
        n=len(pose)
        def vector(x):return torch.as_tensor(x,device=pose.device).long().reshape(-1).expand(n)
        scene,frame,episode,trajectory=map(vector,(scene,frame,episode,trajectory))
        keys=self.keys(pose,scene,frame,episode,trajectory) if capture_keys is None else capture_keys
        if len(keys)!=n:raise ValueError('Capture key batch mismatch')
        out=pose.new_empty((n,1,48,64),dtype=torch.float32)
        missing={};hits=[]
        for i,key in enumerate(keys):
            slot=self.entries.get(key)
            if slot is None:missing.setdefault(key,[]).append(i)
            else:self.entries.move_to_end(key);hits.append((i,slot))
        for lo in range(0,len(hits),self.stage_rows):
            batch=hits[lo:lo+self.stage_rows]
            host=torch.empty((len(batch),1,48,64),dtype=torch.float16)
            groups={}
            for j,(_,slot) in enumerate(batch):groups.setdefault(slot//self.block_size,[]).append((j,slot%self.block_size))
            for block,rows in groups.items():
                dst,src=zip(*rows);host[list(dst)]=self.blocks[block][list(src)]
            out[[row[0] for row in batch]]=self._upload(host)
        self.hits+=len(hits);self.misses+=len(missing)
        if missing:
            first=[rows[0] for rows in missing.values()]
            generated=render(pose[first],scene[first],frame[first],episode[first],trajectory[first],mode='inference')
            stored=generated.to(torch.float16)
            writes={};duplicate_dst=[];duplicate_src=[]
            for j,(key,rows) in enumerate(missing.items()):
                if len(self.entries)<self.capacity:slot=len(self.entries)
                else:_,slot=self.entries.popitem(last=False)
                self.entries[key]=slot;self._storage(slot)
                writes.setdefault(slot//self.block_size,{})[slot%self.block_size]=j
                duplicate_dst.extend(rows);duplicate_src.extend([j]*len(rows))
            out[duplicate_dst]=stored[duplicate_src].float()
            self._store(stored,writes)
        return out

    @property
    def storage_bytes(self):return sum(x.numel()*x.element_size() for x in self.blocks)

    @property
    def staging_bytes(self):
        return sum(item['data'].numel()*item['data'].element_size() for pool in self._pools.values() for item in pool)
