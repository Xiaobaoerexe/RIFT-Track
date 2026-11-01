"""Offline direct-depth fitting with frozen-teacher pose-VJP supervision.

Run the first 100 updates before choosing a longer run budget. No locomotion
updates are performed by this program, and no training checkpoint is accepted
for deployment without the separate test-split evaluator.
"""
import argparse,json,os,signal,statistics,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import DirectDepth,ModelConfig
from .source import import_package,sha256,ieee_fp32,tangent_pose,soft_image,package_fingerprints
from .data import Corpus,assert_disjoint


def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)


def image_objective(z,logit,millimetres):
    gt=millimetres.float()/1000.;valid=gt>0
    denom=valid.sum().clamp_min(1)
    metric=(F.huber_loss(z,gt,delta=.05,reduction='none')*valid).sum()/denom
    bce=F.binary_cross_entropy_with_logits(logit,valid.float(),reduction='none')
    # Balance valid and invalid pixels without giving empty classes infinite weight.
    mask=.5*((bce*valid).sum()/valid.sum().clamp_min(1)+(bce*~valid).sum()/(~valid).sum().clamp_min(1))
    edge=z.new_zeros(())
    for dim in (-1,-2):
        dz=z.diff(dim=dim);dg=gt.diff(dim=dim)
        vm=valid.narrow(dim,0,valid.shape[dim]-1)&valid.narrow(dim,1,valid.shape[dim]-1)
        weight=1+4*(dg.abs()>.05)
        edge+=(F.huber_loss(dz,dg,delta=.05,reduction='none')*vm*weight).sum()/vm.sum().clamp_min(1)
    return 20*metric+mask+10*edge,{'metric_huber':metric.detach(),'validity_bce':mask.detach(),'edge_huber':edge.detach()}


def cotangent(shape,device,generator):
    # Low-frequency and pixel-wise probes expose global and local derivative
    # errors. They are fixed when teacher and student are compared.
    n,h,w=shape
    low=torch.randn((n,1,6,8),generator=generator).to(device)
    low=F.interpolate(low,size=(h,w),mode='bilinear',align_corners=False)[:,0]
    high=torch.randn(shape,generator=generator).to(device)
    c=low+.25*high
    return c/c.square().sum((-1,-2),keepdim=True).sqrt().clamp_min(1e-8)


def pose_supervision(model,source,corpus,d,generator,n=4):
    """Direct Sobolev VJP plus local finite-pose image response, geometry only."""
    n=min(n,len(d['pos']));p=d['pos'][:n];r=d['rot'][:n];ids=d['scene'][:n]
    unique,inv=torch.unique(ids,return_inverse=True)
    encoded=model.encode_scenes(corpus.features[unique])[inv]
    delta=torch.zeros((n,6),device=p.device,requires_grad=True)
    pp,rr=tangent_pose(p,r,delta)
    tz,tl=source.core.geometry_cached(corpus.source_cache,ids,pp,rr)
    probe=cotangent(tz.shape,p.device,generator)
    target=torch.autograd.grad((soft_image(tz,tl)*probe).sum(),delta)[0].detach()
    delta2=torch.zeros_like(delta,requires_grad=True)
    pp,rr=tangent_pose(p,r,delta2)
    sz,sl=model.decode(encoded,pp,rr)
    student=torch.autograd.grad((soft_image(sz,sl)*probe).sum(),delta2,create_graph=True)[0]
    scales=target.square().mean(0).clamp_min(1e-8).detach()
    vjp=((student-target).square()/scales).mean()
    # Preserve source validity confidence as well as hard labels. Matching only
    # binary masks can saturate logits and erase their pose derivatives.
    teacher_fit=F.smooth_l1_loss(sz,tz.detach(),beta=.05)+F.binary_cross_entropy_with_logits(sl,tl.detach().sigmoid())
    # An independent central perturbation trains response over a finite local
    # neighbourhood; it does not replace the differential VJP check above.
    direction=torch.randn((n,6),generator=generator).to(p.device)
    direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-8)*.005
    plus=tangent_pose(p,r,direction);minus=tangent_pose(p,r,-direction)
    with torch.no_grad():
        ap,lp=source.core.geometry_cached(corpus.source_cache,ids,*plus)
        am,lm=source.core.geometry_cached(corpus.source_cache,ids,*minus)
        response=soft_image(ap,lp)-soft_image(am,lm)
    bp,kp=model.decode(encoded,*plus);bm,km=model.decode(encoded,*minus)
    paired=F.smooth_l1_loss(soft_image(bp,kp)-soft_image(bm,km),response,beta=.01)
    return vjp,paired,teacher_fit


@torch.no_grad()
def validate(model,corpus,batch=64,per_scene=8):
    error=valid_count=tp=fp=fn=0
    for idx in corpus.heldout_indices(per_scene).split(batch):
        d=corpus.get(idx);z,l=model(d['features'],d['local_scene'],d['pos'],d['rot'])
        gt=d['depth'].float()/1000;valid=gt>0;pred=l>0
        error+=float(((z-gt).abs()*valid).sum());valid_count+=int(valid.sum())
        tp+=int((pred&valid).sum());fp+=int((pred&~valid).sum());fn+=int((~pred&valid).sum())
    return {'mae_on_label_valid_m':error/max(valid_count,1),'validity_f1':2*tp/max(2*tp+fp+fn,1)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',action='append',required=True)
    p.add_argument('--depth-package',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=100);p.add_argument('--batch',type=int,default=192)
    p.add_argument('--device',default='cuda:0');p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--vjp-every',type=int,default=4);p.add_argument('--vjp-batch',type=int,default=4)
    p.add_argument('--vjp-weight',type=float,default=.1);p.add_argument('--resume')
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=824)
    args=p.parse_args()
    if min(args.steps,args.batch,args.eval_every,args.vjp_every,args.vjp_batch)<=0:raise ValueError('Positive counts required')
    torch.set_num_threads(8);torch.manual_seed(args.seed)
    device=torch.device(args.device);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    package=import_package(args.depth_package)
    source=package.DepthInputGenerator(device=args.device,compile_mode='off',precision='fp32')
    train=Corpus(args.data,'train',device).prepare(source);val=Corpus(args.data,'val',device).prepare(source)
    assert_disjoint(train,val)
    model=DirectDepth().to(device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.7,patience=4,min_lr=1e-6)
    generator=torch.Generator().manual_seed(args.seed)
    start=0;best=float('inf');stop=False;durations=[]
    source_sha=sha256(Path(args.depth_package)/'go2_depth_input/assets/weights.pt')
    source_code=package_fingerprints(args.depth_package)
    for corpus in (train,val):
        for bank in corpus.banks:
            manifest=bank['manifest']
            if manifest.get('supervision_source')=='frozen_teacher' and manifest.get('source_sha256')!=source_sha:
                raise ValueError(f'Teacher bank source hash mismatch: {bank["path"]}')
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        if ck['source_weights_sha256']!=source_sha:raise ValueError('Source package changed since checkpoint')
        if ck['model_config']!=model.configuration():raise ValueError('Resume model config mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        generator.set_state(ck['rng'].cpu());start=ck['step'];best=ck['best']
    config=dict(vars(args),train=train.provenance(),val=val.provenance(),model=model.configuration(),
                parameters=sum(x.numel() for x in model.parameters()),source_sha256=source_sha,
                source_code_sha256=source_code,
                precision='IEEE FP32; no TF32',supervision='RTX and explicitly identified frozen-teacher banks; metric + mask + edge + finite pose response + direct VJP')
    (out/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config),flush=True)
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    began=time.perf_counter();step=start
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    with ieee_fp32(device):
        for step in range(start+1,args.steps+1):
            sync();t=time.perf_counter();d=train.batch(args.batch,generator)
            model.train();opt.zero_grad(set_to_none=True)
            z,l=model(d['features'],d['local_scene'],d['pos'],d['rot'])
            loss,parts=image_objective(z,l,d['depth'])
            if step%args.vjp_every==0:
                vjp,paired,teacher_fit=pose_supervision(model,source,train,d,generator,args.vjp_batch)
                loss=loss+args.vjp_weight*vjp+10*paired+.1*teacher_fit
                parts.update(vjp=vjp.detach(),paired_response=paired.detach(),teacher_fit=teacher_fit.detach())
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}: {parts}')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            sync();durations.append(time.perf_counter()-t)
            if step==start+1 or step%25==0:
                record={'step':step,'loss':float(loss.detach()),'components':{k:float(v) for k,v in parts.items()},
                    'gradient_norm':float(grad),'lr':opt.param_groups[0]['lr'],'seconds_per_update_mean':statistics.mean(durations[-100:]),
                    'seconds_per_update_p50':statistics.median(durations[-100:]),'elapsed_seconds':time.perf_counter()-began,
                    'peak_cuda_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None}
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
            if step%args.eval_every==0 or step==args.steps or stop:
                model.eval();score=validate(model,val);metric=score['mae_on_label_valid_m'];improved=metric<best
                best=min(best,metric);scheduler.step(metric)
                ck={'schema':'direct_depth_v1','model':model.state_dict(),'model_config':model.configuration(),
                    'source_weights_sha256':source_sha,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                    'source_code_sha256':source_code,
                    'training_layouts':[{'name':m['name'],'layout_identity':m['layout_identity']} for m in train.meta],
                    'rng':generator.get_state(),'step':step,'best':best,'config':config,'validation':score,
                    'acceptance':{'passed':False,'reason':'Independent holdout image and VJP acceptance has not passed'}}
                atomic_save(ck,out/'latest.pt')
                if improved:atomic_save(ck,out/'best.pt')
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps({'step':step,**score})+'\n')
            if stop:break
    summary={'step':step,'updates_measured':len(durations),'mean_seconds_per_update':statistics.mean(durations) if durations else None,
             'last100_mean_seconds':statistics.mean(durations[-100:]) if durations else None,
             'total_seconds_including_validation':time.perf_counter()-began,'accepted':False,'interrupted':stop}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
