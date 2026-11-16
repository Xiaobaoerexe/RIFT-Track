"""Bounded multi-GPU inference for the frozen keyed depth-noise network.

This does not move trainable CNNs, optimizer state, Replay or imagined latents.
Each image retains its key and output row. At most one chunk per device is in
flight; no reliance on peer-to-peer support (consumer4090s may lack it).
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import torch


class _Worker:
    def __init__(self,device,noise,function,chunk_size,mixed_precision,compile_enabled):
        self.device=torch.device(device);self.chunk_size=chunk_size
        self.mixed_precision=mixed_precision
        with torch.cuda.device(self.device):
            self.noise=copy.deepcopy(noise).to(self.device).eval().requires_grad_(False)
            self.counter=torch.arange(4*54*70,device=self.device,dtype=torch.int64)[None]
            self.stream=torch.cuda.Stream(device=self.device)
            self.function=torch.compile(function,dynamic=False,fullgraph=True) if compile_enabled else function
            # Initialization uses the current stream; publish only complete replicas.
            self.stream.wait_stream(torch.cuda.current_stream(self.device))

    @torch.no_grad()
    def run(self,signed,keys,ready):
        with torch.cuda.device(self.device),torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            signed=signed.to(self.device,non_blocking=True)
            keys=keys.to(self.device,non_blocking=True)
            z=signed[:,0].abs();valid=(signed[:,0]>0).float();count=len(z)
            if count<self.chunk_size:
                pad=self.chunk_size-count
                z=torch.cat((z,z[:1].expand(pad,-1,-1)))
                valid=torch.cat((valid,valid[:1].expand(pad,-1,-1)))
                keys=torch.cat((keys,keys[:1].expand(pad)))
            result=self.function(self.noise,z,valid,keys,self.counter,self.mixed_precision)[:count].half()
        # A worker cannot reuse its bounded scratch before its result completes.
        # Synchronize here; transfers back are issued on the caller's own stream.
        self.stream.synchronize()
        return result


class FrozenNoisePool:
    def __init__(self,devices,noise,function,chunk_size,mixed_precision,compile_enabled):
        self.chunk_size=chunk_size
        self.workers=[_Worker(d,noise,function,chunk_size,mixed_precision,compile_enabled) for d in devices]
        self.executor=ThreadPoolExecutor(max_workers=len(self.workers),thread_name_prefix='as2-frozen-noise')

    @torch.no_grad()
    def render(self,render_clean,pose,keys):
        outputs=[];width=self.chunk_size*len(self.workers)
        for start in range(0,len(pose),width):
            # Exact geometry remains on the PhysX device and is evaluated once.
            signed=render_clean(pose[start:start+width]).half().float()
            row_keys=keys[start:start+width]
            ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(pose.device))
            pending=[]
            for index,worker in enumerate(self.workers):
                lo=index*self.chunk_size
                if lo>=len(signed):break
                sl=slice(lo,lo+self.chunk_size)
                pending.append(self.executor.submit(worker.run,signed[sl],row_keys[sl],ready))
            # Preserve query order and bound cross-device staging memory. Blocking
            # result copies are intentional: no dangling asynchronous cache input.
            outputs.extend(job.result().to(pose.device) for job in pending)
        return torch.cat(outputs,0)

    def close(self):
        self.executor.shutdown(wait=True)
