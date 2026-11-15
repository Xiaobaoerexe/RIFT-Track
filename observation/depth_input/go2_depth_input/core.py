"""Frozen sensor-noise keys and fields; geometric generator retired.

Functions retained verbatim from supplied depth_input0.1.0.
"""
import math
import torch
from torch.nn import functional as F


def mix32(x):
    x=x & 0xffffffff
    x=((x^(x>>16))*0x45d9f3b)&0xffffffff
    x=((x^(x>>16))*0x45d9f3b)&0xffffffff
    return x^(x>>16)


def noise_keys(trajectory_ids,frame_index,seed=317,episode_ids=0):
    """Explicit per-observation keys. Identical keys reproduce forward/backward noise."""
    return mix32(trajectory_ids.long()*0x9e3779b1+frame_index*0x85ebca6b+episode_ids*0xc2b2ae35+seed)


def fields_from_keys(keys,counter=None):
    # Stateless Gaussian samples, independent of chunk boundaries and global RNG.
    if counter is None:
        counter=torch.arange(4*54*70,device=keys.device,dtype=torch.int64)[None]
    # Signed spelling is identical modulo 2**32 and avoids an Inductor/Triton
    # index-expression int32 literal overflow for 0x9e3779b1.
    base=keys[:,None]+counter*(-1640531535)
    a=mix32(base);b=mix32(base+0x68bc21eb)
    u=((a>>8).float()+.5)/16777216.;v=((b>>8).float()+.5)/16777216.
    normals=(torch.sqrt(-2*torch.log(u))*torch.cos(2*math.pi*v)).reshape(-1,4,54,70)
    fields=torch.stack((normals[:,0,:48,:64],(F.avg_pool2d(normals[:,1:2,:50,:66],3,1)*3)[:,0],(F.avg_pool2d(normals[:,2:3],7,1)*7)[:,0]),1)
    return fields,normals[:,3,:48,:64]
