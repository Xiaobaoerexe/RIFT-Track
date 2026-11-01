"""Independent image, layout coverage and full-pose VJP acceptance.

RTX error and frozen-teacher compatibility are different evidence. New terrain
banks without RTX labels are explicitly reported as such. The fixed 5% VJP
gate compares the old model's surrogate gradient; it does not certify exact
derivatives through real visibility, collisions or binary holes.
"""
import argparse,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .api import FastDepthGenerator,FastSceneCache
from .data import Corpus
from .source import ieee_fp32,tangent_pose,soft_image,sha256,package_fingerprints
from .train import atomic_save,cotangent

THRESHOLDS={'common_valid_mae_m':.02,'common_valid_p95_m':.10,
            'validity_f1_min':.98,'edge_mae_m':.05,'pose_vjp_relative_l2_max':.05,
            'rtx_relative_degradation_max':.10}
FAMILIES={0:'nominal',1:'transition',2:'fallen',3:'full_orientation'}
NEW_TERRAINS={'indoor_obstacles','obstacle_up','obstacle_down'}


def terrain_name(name):
    return {'obstacles_up':'obstacle_up','obstacles_down':'obstacle_down'}.get(name,name)


def edge_mask(gt,valid):
    mask=torch.zeros_like(valid)
    for dim in (-1,-2):
        a=valid.narrow(dim,0,valid.shape[dim]-1);b=valid.narrow(dim,1,valid.shape[dim]-1)
        edge=(gt.diff(dim=dim).abs()>.05)&a&b | (a^b)
        mask.narrow(dim,0,valid.shape[dim]-1).logical_or_(edge)
        mask.narrow(dim,1,valid.shape[dim]-1).logical_or_(edge)
    return mask


class Images:
    def __init__(self):
        self.errors=[];self.edge_sum=0.;self.edge_count=0;self.tp=self.fp=self.fn=0
        self.label_sum=0.;self.label_count=0;self.frames=0
    def add(self,z,hit,gt,valid):
        z,hit,gt,valid=(x.detach().cpu() for x in (z,hit,gt,valid))
        e=(z-gt).abs();common=hit&valid;edge=common&edge_mask(gt,valid)
        self.errors.append(e[common]);self.edge_sum+=float(e[edge].sum());self.edge_count+=int(edge.sum())
        self.label_sum+=float(e[valid].sum());self.label_count+=int(valid.sum())
        self.tp+=int(common.sum());self.fp+=int((hit&~valid).sum());self.fn+=int((~hit&valid).sum());self.frames+=len(z)
    def result(self):
        e=torch.cat(self.errors) if self.errors else torch.empty(0)
        denominator=2*self.tp+self.fp+self.fn
        return {'frames':self.frames,'common_valid_pixels':len(e),'label_valid_pixels':self.label_count,
            'common_valid_mae_m':float(e.mean()) if len(e) else 0.,
            'common_valid_p95_m':float(e.quantile(.95)) if len(e) else 0.,
            'validity_f1':2*self.tp/denominator if denominator else 1.,
            'edge_mae_m':self.edge_sum/max(self.edge_count,1),'edge_pixels':self.edge_count,
            'label_valid_mae_m':self.label_sum/max(self.label_count,1)}


class VJPs:
    def __init__(self):self.diff=torch.zeros(6,dtype=torch.float64);self.ref=self.diff.clone();self.dot=self.diff.clone();self.count=0
    def add(self,candidate,reference):
        a,b=candidate.detach().double().cpu(),reference.detach().double().cpu()
        self.diff+=(a-b).square().sum(0);self.ref+=b.square().sum(0);self.dot+=(a*b).sum(0);self.count+=len(a)
    def result(self):
        ratio=lambda sl:float((self.diff[sl].sum()/self.ref[sl].sum().clamp_min(1e-20)).sqrt())
        return {'frames_times_probes':self.count,'translation_relative_l2':ratio(slice(0,3)),
                'rotation_relative_l2':ratio(slice(3,6)),
                'axis_relative_l2':(self.diff/self.ref.clamp_min(1e-20)).sqrt().tolist(),
                'reference_axis_rms':(self.ref/max(self.count,1)).sqrt().tolist()}


def image_failures(metric,prefix):
    failures=[]
    for name in ('common_valid_mae_m','common_valid_p95_m','edge_mae_m'):
        if not math.isfinite(metric[name]) or metric[name]>THRESHOLDS[name]:failures.append(f'{prefix}: {name}={metric[name]:.6g}')
    if metric['validity_f1']<THRESHOLDS['validity_f1_min']:failures.append(f'{prefix}: validity_f1={metric["validity_f1"]:.6g}')
    return failures


def vjp_failures(metric,prefix):
    return [f'{prefix}: {key}={metric[key]:.6g}' for key in ('translation_relative_l2','rotation_relative_l2')
            if not math.isfinite(metric[key]) or metric[key]>THRESHOLDS['pose_vjp_relative_l2_max']]


def grouped_names(corpus,d,i):
    m=corpus.meta[int(d['scene'][i])]
    return ('all','terrain:'+terrain_name(m['name']),'pose:'+FAMILIES[int(d['mode'][i])],
            'layout:'+terrain_name(m['name'])+':'+m['layout_identity'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--depth-package',required=True)
    p.add_argument('--data',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--precision',choices=('fp32','bf16'),default='fp32')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--views-per-scene',type=int,default=16)
    p.add_argument('--gradient-views-per-scene',type=int,default=4);p.add_argument('--probes',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(8)
    if min(args.batch,args.views_per_scene,args.gradient_views_per_scene,args.probes)<=0:raise ValueError('Positive counts required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    engine=FastDepthGenerator(args.weights,args.depth_package,device=args.device,chunk_size=args.batch,
                              compile_mode='off',precision=args.precision,require_accepted=False)
    source=engine.source;source.chunk_size=args.batch
    corpus=Corpus(args.data,'test',args.device).prepare(source)
    # Check the training identities recorded with the checkpoint against test.
    checkpoint=torch.load(args.weights,map_location='cpu',weights_only=False)
    if checkpoint.get('source_code_sha256') and checkpoint['source_code_sha256']!=package_fingerprints(args.depth_package):
        raise ValueError('Pinned source observation code changed after candidate training')
    train_meta=checkpoint.get('training_layouts')
    if train_meta is not None:
        known={(x['name'],str(x['layout_identity'])) for x in train_meta}
        overlap=known&{(m['name'],m['layout_identity']) for m in corpus.meta}
        if overlap:raise ValueError(f'Train/test layout overlap: {overlap}')
    with torch.no_grad(),ieee_fp32(args.device):
        embedding=engine.core.model.encode_scenes(corpus.features)
    cache=FastSceneCache(embedding,embedding.new_zeros((len(embedding),3)),len(embedding))
    from go2_depth_input.api import SceneCache
    source_cache=SceneCache(corpus.source_cache,cache.origins,len(embedding))
    reports={name:{} for name in ('labels','teacher_geometry','teacher_noisy','old_rtx','candidate_rtx')}
    gradients={name:{} for name in ('geometry','noisy')}
    def decode(pose,rotation,ids):
        with torch.autocast(device_type=engine.device.type,dtype=torch.bfloat16,enabled=args.precision=='bf16'):
            return engine.core.model.decode(embedding[ids],pose,rotation)
    indices=corpus.heldout_indices(args.views_per_scene)
    with ieee_fp32(args.device):
        for number,idx in enumerate(indices.split(args.batch)):
            d=corpus.get(idx);ids=d['scene'];keys=idx.to(engine.device).long()+17423
            with torch.no_grad():
                z,l=decode(d['pos'],d['rot'],ids)
                tz,tl=source.core.geometry_cached(corpus.source_cache,ids,d['pos'],d['rot'])
                noisy=engine.render(cache,d['pos'],d['rot'],ids,keys,mode='inference')[:,0]
                tn=source.render(source_cache,d['pos'],d['rot'],ids,keys,mode='inference')[:,0]
            label=d['depth'].float()/1000;valid=label>0
            for i in range(len(ids)):
                names=grouped_names(corpus,d,i);sl=slice(i,i+1)
                for name in names:
                    reports['labels'].setdefault(name,Images()).add(z[sl],l[sl]>0,label[sl],valid[sl])
                    reports['teacher_geometry'].setdefault(name,Images()).add(z[sl],l[sl]>0,tz[sl],tl[sl]>0)
                    reports['teacher_noisy'].setdefault(name,Images()).add(noisy[sl],noisy[sl]>0,tn[sl],tn[sl]>0)
                    if corpus.meta[int(ids[i])]['supervision_source']=='rtx':
                        reports['old_rtx'].setdefault(name,Images()).add(tz[sl],tl[sl]>0,label[sl],valid[sl])
                        reports['candidate_rtx'].setdefault(name,Images()).add(z[sl],l[sl]>0,label[sl],valid[sl])
            if number%50==0:print(json.dumps({'phase':'images','frames':min((number+1)*args.batch,len(indices)),'total':len(indices)}),flush=True)
        generator=torch.Generator().manual_seed(9917)
        gradient_indices=corpus.heldout_indices(args.gradient_views_per_scene)
        recompute_check=None;frozen_parameters=all(not q.requires_grad for q in engine.core.parameters())
        for number,idx in enumerate(gradient_indices.split(args.batch)):
            d=corpus.get(idx);ids=d['scene'];keys=idx.to(engine.device).long()+17423;n=len(ids)
            for chain in ('geometry','noisy'):
                for probe_index in range(args.probes):
                    probe=cotangent((n,48,64),engine.device,generator)
                    delta=torch.zeros(n,6,device=engine.device,requires_grad=True)
                    pp,rr=tangent_pose(d['pos'],d['rot'],delta)
                    if chain=='geometry':
                        tz,tl=source.core.geometry_cached(corpus.source_cache,ids,pp,rr);target=soft_image(tz,tl)
                    else:target=source.render(source_cache,pp,rr,ids,keys,mode='autograd')[:,0]
                    reference=torch.autograd.grad((target*probe).sum(),delta)[0]
                    delta2=torch.zeros_like(delta,requires_grad=True);pp,rr=tangent_pose(d['pos'],d['rot'],delta2)
                    if chain=='geometry':
                        z,l=decode(pp,rr,ids);image=soft_image(z,l)
                    else:image=engine.render(cache,pp,rr,ids,keys,mode='autograd')[:,0]
                    candidate=torch.autograd.grad((image*probe).sum(),delta2)[0]
                    for i in range(n):
                        for name in grouped_names(corpus,d,i):
                            gradients[chain].setdefault(name,VJPs()).add(candidate[i:i+1],reference[i:i+1])
                    if chain=='noisy' and recompute_check is None:
                        delta3=torch.zeros_like(delta,requires_grad=True);pp,rr=tangent_pose(d['pos'],d['rot'],delta3)
                        saved=engine.render(cache,pp,rr,ids,keys,mode='recompute')[:,0]
                        recomputed=torch.autograd.grad((saved*probe).sum(),delta3)[0]
                        recompute_check={'output_max_abs':float((saved-image.detach()).abs().max()),
                            'vjp_max_abs':float((recomputed-candidate).abs().max()),
                            'vjp_relative_l2':float((recomputed-candidate).norm()/candidate.norm().clamp_min(1e-10))}
            if number%10==0:print(json.dumps({'phase':'pose_vjp','frames':min((number+1)*args.batch,len(gradient_indices)),'total':len(gradient_indices)}),flush=True)
    reports={chain:{group:value.result() for group,value in groups.items()} for chain,groups in reports.items()}
    gradients={chain:{group:value.result() for group,value in groups.items()} for chain,groups in gradients.items()}
    failures=[]
    # Gates apply to each layout as well as global/terrain/pose summaries, so a
    # good flat-ground average cannot hide a broken new terrain or fallen pose.
    # Fixed compatibility tolerances compare against the pinned observer. RTX
    # label errors use the measured source baseline below: the reference itself
    # is not exact, so an absolute 2cm RTX gate could reject an exact copy of it.
    for chain in ('teacher_geometry','teacher_noisy'):
        for group,metric in reports[chain].items():failures+=image_failures(metric,f'{chain}/{group}')
    for chain,groups in gradients.items():
        for group,metric in groups.items():failures+=vjp_failures(metric,f'{chain}/{group}')
    rtx_degradation={}
    for group,old in reports['old_rtx'].items():
        new=reports['candidate_rtx'][group]
        limit=old['label_valid_mae_m']*(1+THRESHOLDS['rtx_relative_degradation_max'])+1e-6
        rtx_degradation[group]={'old_mae_m':old['label_valid_mae_m'],'candidate_mae_m':new['label_valid_mae_m'],'allowed_mae_m':limit}
        if new['label_valid_mae_m']>limit:failures.append(f'RTX regression/{group}: {new["label_valid_mae_m"]:.6g} > {limit:.6g}')
        f1_floor=old['validity_f1']-.01
        edge_limit=max(THRESHOLDS['edge_mae_m'],old['edge_mae_m']*1.1)
        rtx_degradation[group].update(old_f1=old['validity_f1'],candidate_f1=new['validity_f1'],
                                      allowed_f1_min=f1_floor,old_edge_mae_m=old['edge_mae_m'],
                                      candidate_edge_mae_m=new['edge_mae_m'],allowed_edge_mae_m=edge_limit)
        if new['validity_f1']<f1_floor:failures.append(f'RTX mask regression/{group}')
        if new['edge_mae_m']>edge_limit:failures.append(f'RTX edge regression/{group}')
    terrains={terrain_name(m['name']) for m in corpus.meta}
    missing=sorted(NEW_TERRAINS-terrains)
    if missing:failures.append(f'Missing new-terrain holdout: {missing}')
    for family in FAMILIES.values():
        if 'pose:'+family not in gradients['noisy']:failures.append(f'Missing pose derivative holdout: {family}')
    if not reports['old_rtx']:failures.append('No RTX ground-truth holdout')
    if not train_meta:failures.append('Checkpoint has no training-layout identities for leakage audit')
    if not frozen_parameters:failures.append('Observation/noise parameters are not frozen')
    if recompute_check is None or recompute_check['output_max_abs']>1e-6 or recompute_check['vjp_relative_l2']>1e-5:
        failures.append('Recompute forward/backward does not match ordinary autograd')
    for row in corpus.meta:
        if row['supervision_source']=='frozen_teacher':
            bank=next(b for b in corpus.banks if b['path']==row['bank'])
            if bank['manifest'].get('source_sha256')!=checkpoint['source_weights_sha256']:
                failures.append(f'Teacher bank source hash mismatch: {row["bank"]}')
    result={'passed':not failures,'precision':args.precision,'weights_sha256':sha256(args.weights),
            'source_weights_sha256':checkpoint['source_weights_sha256'],'thresholds':THRESHOLDS,
            'holdout':corpus.provenance(),'images':reports,'pose_vjp':gradients,
            'rtx_regression':rtx_degradation,'recompute_equivalence':recompute_check,'frozen_parameters':frozen_parameters,
            'absolute_rtx_missing_terrains':sorted({terrain_name(m['name']) for m in corpus.meta if m['supervision_source']!='rtx'}-
                {terrain_name(m['name']) for m in corpus.meta if m['supervision_source']=='rtx'}),
            'gradient_meaning':'Agreement with pinned old geometry/noise surrogate at fixed keys; binary visibility/hole derivatives are not exact physical derivatives.',
            'failures':failures,'elapsed_seconds':time.perf_counter()-began}
    def finite_json(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value,list):return [finite_json(v) for v in value]
        return value
    report=out/f'acceptance.{args.precision}.json';report.write_text(json.dumps(finite_json(result),indent=2,allow_nan=False))
    if result['passed']:
        existing=checkpoint.get('acceptance',{})
        profiles=dict(existing.get('profiles',{}));profiles[args.precision]={'passed':True,'report':str(report),'report_sha256':sha256(report)}
        checkpoint['acceptance']={'passed':True,'accepted_precisions':sorted(profiles),'profiles':profiles,
                                  'caveat':result['gradient_meaning'],'absolute_rtx_missing_terrains':result['absolute_rtx_missing_terrains']}
        atomic_save(checkpoint,out/'accepted.pt')
    print(json.dumps({'passed':result['passed'],'failures':len(failures),'first_failures':failures[:20],'report':str(report),
                      'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
