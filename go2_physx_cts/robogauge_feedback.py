"""Launch the pinned official runtime from the independent training runtime."""
import json,subprocess,sys
from pathlib import Path

def evaluate_checkpoint(checkpoint,output,iteration,official_python):
    if not official_python:
        raise ValueError('An explicit RoboGauge Python3.11/MuJoCo3.2.3 runtime is required')
    root=Path(__file__).resolve().parents[1];out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    from .commands import RANGE_ITERATIONS,RANGE_STAGES
    stage=sum(iteration>=v for v in RANGE_ITERATIONS)
    speeds=','.join(str(RANGE_STAGES[i][0][1]) for i in (stage,min(stage+1,len(RANGE_STAGES)-1)))
    commands=[
        [sys.executable,'-m','go2_physx_cts.robogauge_export','--checkpoint',str(checkpoint),'--output',str(out/'exports'),'--iteration',str(iteration)],
        [official_python,'-m','go2_physx_cts.robogauge_bridge','--exports',str(out/'exports'),'--output',str(out/'official'),'--iterations',str(iteration),'--forward-speeds',speeds],
    ]
    for i,cmd in enumerate(commands):
        proc=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=1800)
        (out/f'process_{i}.log').write_text(proc.stdout+'\n'+proc.stderr)
        if proc.returncode:raise RuntimeError(f'RoboGauge process failed; see {out}/process_{i}.log')
    result=json.loads((out/'official/results.json').read_text())
    # Transfer measurements are not a native capability gate or an automatic
    # rollback signal. A native/transfer decision requires explicit review.
    result['automatic_training_decision']=False
    (out/'result.json').write_text(json.dumps(result,indent=2))
    return result
