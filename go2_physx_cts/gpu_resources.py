"""Select idle visible 4090s without changing global training batch sizes.

PhysX and all trainable parameters stay on one device. Additional devices run
replicas of the frozen sensor-noise network, not independent experiments.
"""
import csv
import io
import os
import subprocess


def _query(fields,kind='gpu'):
    result=subprocess.run(['nvidia-smi',f'--query-{kind}='+','.join(fields),
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    return [dict(zip(fields,(v.strip() for v in row)))
            for row in csv.reader(io.StringIO(result.stdout)) if row]


def resolve_devices(cfg):
    """Run before PhysX starts; never claim a GPU owned by another compute job.

    UUID matching respects CUDA_VISIBLE_DEVICES and CUDA's device enumeration.
    Fail explicitly if inventory cannot be verified instead of guessing ordinals.
    This is a startup snapshot, not a cluster reservation service.
    """
    import torch
    if cfg.gpu_mode=='single':
        cfg.noise_devices=()
        return {'mode':'single','primary':cfg.device,'noise_devices':[cfg.device],
                'auto_selection':False}
    devices=_query(('uuid','name','memory.total','memory.free'))
    busy={row['gpu_uuid'] for row in _query(('gpu_uuid','pid'),'compute-apps')
          if row['pid']!=str(os.getpid())}
    visible=[]
    for index in range(torch.cuda.device_count()):
        props=torch.cuda.get_device_properties(index)
        uuid=str(getattr(props,'uuid',''))
        matches=[row for row in devices if row['uuid'].removeprefix('GPU-')==uuid.removeprefix('GPU-')]
        if len(matches)!=1:
            raise RuntimeError(f'Cannot map CUDA device {index} to a verified NVIDIA UUID; use gpu_mode=single with explicit visibility')
        row=matches[0]
        visible.append({'device':f'cuda:{index}','uuid':row['uuid'],'name':props.name,
            'total_mib':float(row['memory.total']),'free_mib':float(row['memory.free']),
            'foreign_compute':row['uuid'] in busy})
    idle=[d for d in visible if '4090' in d['name'] and not d['foreign_compute']]
    primary=[d for d in idle if d['free_mib']>=21*1024]
    if not primary:
        raise RuntimeError('No idle visible RTX4090 with at least21GiB free; existing jobs will not be interrupted')
    chosen=max(primary,key=lambda d:d['free_mib'])
    workers=[chosen]+[d for d in idle if d!=chosen and d['free_mib']>=4*1024]
    execution_before={k:getattr(cfg,k) for k in ('replay_storage','depth_cache_gib','torch_threads')}
    # Reuse the measured24GiB execution choices, not the48GiB GPU-Replay layout.
    # Replay sampling, physical batch sizes and optimizer steps do not change.
    if chosen['total_mib']<32*1024:
        cfg.replay_storage='cpu'
        cfg.depth_cache_gib=min(cfg.depth_cache_gib,2.)
        cfg.torch_threads=min(cfg.torch_threads,4)
    cfg.device=chosen['device'];cfg.noise_devices=tuple(d['device'] for d in workers)
    torch.cuda.set_device(cfg.device)
    return {'mode':'auto','primary':cfg.device,'noise_devices':list(cfg.noise_devices),
        'visible_devices':visible,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'execution_before':execution_before,
        'execution_resolved':{k:getattr(cfg,k) for k in execution_before},
        'distribution':'single PhysX/learner; parallel frozen noise inference',
        'global_batch_changed':False,'multi_gpu_runtime_verified':False}
