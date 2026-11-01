"""Qualify a newly trained approximate observer, not an equivalent code rewrite.

An architecture change changes its function and derivatives. We report old-model
VJP agreement separately; correctness checks use the new function's numerical
derivatives, analytic geometry, local response and frozen/recompute contracts.
This does not certify visibility/hole discontinuities as exactly differentiable.
"""
import argparse,json,math
from pathlib import Path
import torch
from .source import sha256
from .train import atomic_save

def main():
    p=argparse.ArgumentParser()
    for name in ('weights','images','gradient','boxes','benchmark','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();image=json.loads(Path(a.images).read_text());grad=json.loads(Path(a.gradient).read_text())
    boxes=json.loads(Path(a.boxes).read_text());bench=json.loads(Path(a.benchmark).read_text());digest=sha256(a.weights)
    assert image['weights_sha256']==grad['weights_sha256']==boxes['weights_sha256']==digest
    assert bench['depth']['direct_model']['checkpoint_sha256']==digest
    failures=[];limits={'near':.05,'middle':.10,'far':.16}
    def maximum(value,limit,name):
        if value is None or not math.isfinite(value) or value>limit:failures.append(f'{name}: {value} > {limit}')
    # Valid depth accuracy and mask coverage are separate quantities. Invalid
    # sensor readings are not assigned a fictitious metric depth to pass a gate.
    for name,metric in [('RTX',image['images']['candidate_rtx']['all']),('new_collision_boxes',boxes['candidate']['all'])]:
        for band,limit in limits.items():maximum(metric['common_valid_range_error'][band]['mae_m'],limit,name+'/'+band)
        maximum(1-metric['validity_f1'],.02,name+'/mask_error_1_minus_F1')
        maximum(metric['edge_mae_m'],.08,name+'/edge_mean_m')
    for name,metric in boxes['candidate'].items():
        for band,limit in limits.items():maximum(metric['common_valid_range_error'][band]['mae_m'],limit,'box_terrain/'+name+'/'+band)
        maximum(1-metric['validity_f1'],.03,'box_terrain/'+name+'/mask_error')
    for name,metric in image['images']['teacher_geometry'].items():
        if not name.startswith('terrain:'):continue
        for band,limit in limits.items():
            if metric['common_valid_range_error'][band]['pixels']:
                maximum(metric['common_valid_range_error'][band]['mae_m'],limit,'compatibility/'+name+'/'+band)
    maximum(grad['summary']['stable_relative_l2'],.05,'new_function_numerical_VJP')
    maximum(1-grad['summary']['finite_difference_stable_fraction'],.5,'finite_difference_unstable_fraction')
    maximum(grad['analytic_plane']['vjp_relative_l2'],.05,'analytic_plane_VJP')
    maximum(grad['analytic_plane']['mae_m'],.05,'analytic_plane_depth')
    for name,limit in {'near':.03,'middle':.05,'far':.08}.items():
        maximum(grad['summary']['range_response_mae'][name],limit,'local_pose_response/'+name)
    eq=image['recompute_equivalence'];maximum(eq['output_max_abs'],1e-6,'recompute_image');maximum(eq['vjp_relative_l2'],1e-5,'recompute_gradient')
    if not image['frozen_parameters']:failures.append('Unfrozen observer parameters')
    if not bench['passed'] or not bench['restore_passed']:failures.append('Full PhysX update/restore validation failed')
    if bench['config']['actor_experts']!=3 or bench['config']['imag_batch']!=8192 or bench['config']['teacher_fraction']!=.75:
        failures.append('Formal workload changed')
    required={'gap','indoor_obstacles','obstacle_up','obstacle_down'}
    if not required.issubset(boxes['candidate']):failures.append('Missing new-task physical geometry')
    ck=torch.load(a.weights,map_location='cpu',weights_only=False)
    record={'profile':'learned_approximation_v3','passed':not failures,'failures':failures,'weights_sha256':digest,
        'depth_mae_limits_m':limits,'depth_bands_m':{'near':[.1,2.],'middle':[2.,4.],'far':[4.,6.]},
        'metrics':'Depth errors on jointly valid pixels, with F1 separately constrained; all-label errors and p90/p95 remain in the reports.',
        'source_gradient_equivalence_passed':image['passed'],'source_gradient_equivalence_required':False,
        'source_gradient_comparison':{k:v['all'] for k,v in image['pose_vjp'].items()},
        'gradient_scope':'New-function numerical derivatives on independently finite-difference-stable samples, analytic plane, finite local pose response; no claim of exact hard visibility/hole derivatives.',
        'finite_difference_stable_fraction':grad['summary']['finite_difference_stable_fraction'],
        'source_equivalence_caveat':'The old observer VJP equivalence gate did not pass. This is a new learned approximation validated under user-approved depth tolerances, not an equivalent replacement of every derivative.',
        'evidence':{k:{'path':v,'sha256':sha256(v)} for k,v in vars(a).items() if k not in ('output','weights')}}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);report=out/'ACCEPTANCE.json';report.write_text(json.dumps(record,indent=2))
    if failures:
        print(json.dumps(record));raise SystemExit(2)
    ck={k:v for k,v in ck.items() if k not in ('optimizer',)}
    ck['acceptance']={'passed':True,'accepted_precisions':['fp32'],'profile':record['profile'],
        'report':str(report),'report_sha256':sha256(report),'source_gradient_equivalence_passed':False,
        'gradient_scope':record['gradient_scope'],'depth_mae_limits_m':limits}
    atomic_save(ck,out/'accepted.pt');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
