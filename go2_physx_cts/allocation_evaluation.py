"""Fixed-case planning for task allocation; execution stays in RoboGauge."""
import hashlib
from pathlib import Path
from .task_allocation import SEEDS,PAYLOADS,TERRAIN_WEIGHTS,fingerprint

def plan_cases(iteration,manifest,control_protocol):
    from .commands import speed_cap
    plans=[];missing=[]
    for task,weights in enumerate(TERRAIN_WEIGHTS):
        for terrain in weights:
            level=5 if task==6 else 0
            maps=[m for m in manifest.get('maps',[]) if m.get('task_id')==task and m.get('terrain_id')==terrain and m.get('level')==level]
            maps.sort(key=lambda m:(m.get('label',''),m.get('layout',0)))
            if not maps:
                missing.append({'task_id':task,'terrain_id':terrain,'reason':'missing_exact_map'});continue
            for branch in ('teacher','student'):
                for i,seed in enumerate(SEEDS):
                    for j,payload in enumerate(PAYLOADS):
                        m=maps[(2*i+j)%len(maps)]
                        if task==6 and 'recovery_qpos' not in m:
                            missing.append({'task_id':task,'terrain_id':terrain,'reason':'missing_recovery_pose'});continue
                        if task==0:command=((0.,0.,0.),(.5,0.,0.),(0.,.3,0.),(0.,0.,.8),(speed_cap(0,iteration),0.,0.))[i]
                        elif task==1:command=(1.5 if i==0 else speed_cap(1,iteration),0.,0.)
                        else:command=(1. if task in (5,8,9) else .5,0.,0.)
                        record={'task_id':task,'terrain_id':terrain,'branch':branch,'level':level,'seed':seed,'payload_kg':payload,
                                'command':list(command),'map_sha256':hashlib.sha256(Path(m['xml']).read_bytes()).hexdigest(),
                                'fixture':m.get('label','default'),'recovery_qpos':m.get('recovery_qpos'),
                                'spawn':m.get('spawn'),'goal':m.get('goal'),'protocol':control_protocol}
                        record['case_id']=fingerprint(record)
                        plans.append((record,m))
    return plans,missing
