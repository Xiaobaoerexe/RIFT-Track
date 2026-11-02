"""Launch the pinned official runtime from the independent training runtime."""
import hashlib,json,math,subprocess,sys
from pathlib import Path

def evaluate_checkpoint(checkpoint,output,iteration,official_python):
    if not official_python:
        raise ValueError('An explicit RoboGauge Python3.11/MuJoCo3.2.3 runtime is required')
    root=Path(__file__).resolve().parents[1];out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    from .commands import speed_cap
    current=float(speed_cap(0,iteration));speeds=f'{current},{min(current+.5,3.)}'
    fast=float(speed_cap(1,iteration));fast_speeds=f'{fast},{min(fast+.5,5.)}'
    active=[0] if iteration<5000 else list(range(5)) if iteration<8000 else list(range(10))
    manifest=Path(checkpoint).parent/'robogauge_maps/manifest.json'
    commands=[
        [sys.executable,'-m','go2_physx_cts.robogauge_export','--checkpoint',str(checkpoint),'--output',str(out/'exports'),'--iteration',str(iteration)],
        [official_python,'-m','go2_physx_cts.robogauge_bridge','--exports',str(out/'exports'),'--output',str(out/'official'),'--iterations',str(iteration),'--forward-speeds',speeds,'--tasks',','.join(map(str,active)),'--video'],
    ]
    if manifest.exists():
        maps=json.loads(manifest.read_text())
        allocation_maps=manifest.parent/'allocation/manifest.json'
        if allocation_maps.exists():maps['maps']+=json.loads(allocation_maps.read_text())['maps']
        combined=out/'task_maps.json';combined.write_text(json.dumps(maps,indent=2))
        commands[1]+=['--task-maps',str(combined)]
    if iteration>=5000:commands[1]+=['--high-speed-probes',fast_speeds]
    if iteration>=20000:commands[1]+=['--allocation-cohort']
    for i,cmd in enumerate(commands):
        proc=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=7200)
        (out/f'process_{i}.log').write_text(proc.stdout+'\n'+proc.stderr)
        if proc.returncode:raise RuntimeError(f'RoboGauge process failed; see {out}/process_{i}.log')
    result=json.loads((out/'official/results.json').read_text())
    # Transfer measurements are not a native capability gate or an automatic
    # rollback signal. A native/transfer decision requires explicit review.
    result['automatic_training_decision']=False
    teacher=[r for r in result.get('rows',[]) if r['branch']=='teacher' and r.get('cohort_role')=='fixed_protocol']
    cases=[(r['task_id'],r['map_sha256'],r.get('fixture_label'),r['command'],r['payload_kg'],r['seed']) for r in teacher]
    valid=bool(teacher) and not result.get('skipped') and all(r.get('error') is None and r.get('xy_rmse') is not None for r in teacher)
    scores=[float(r['survived'])-float(r['xy_rmse'])-.5*float(r['yaw_rmse']) for r in teacher if r.get('xy_rmse') is not None]
    result['teacher_fixed_eval']={'kind':'teacher_transfer_fixed_eval','valid':valid,
        'cohort_id':hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
        'success_rate':sum(bool(r['pass']) for r in teacher)/max(1,len(teacher)),
        'return':sum(scores)/max(1,len(scores)),'iteration':iteration,
        'metric':'diagnostic score: survival - xy_RMSE - 0.5*yaw_RMSE; NOT training reward return',
        'depth_contract':'actual MuJoCo geometry with frozen noise at checkpoint camera cadence'}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    return result
