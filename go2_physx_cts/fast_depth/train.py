"""Offline distillation of source image correction with its geometry held fixed."""
import argparse,json,os,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import SurfaceRefiner,ModelConfig,surface_features,refiner_output,decode
from .source import import_package,sha256,package_fingerprints,ieee_fp32
from .data import Corpus,assert_disjoint,collision_box_labels

def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def cotangent(shape,device,generator):
    n,h,w=shape
    x=F.interpolate(torch.randn(n,1,6,8,generator=generator).to(device),size=(h,w),mode='bilinear',align_corners=False)[:,0]
    x=x+.25*torch.randn(shape,generator=generator).to(device)
    return x/x.square().sum((-1,-2),keepdim=True).sqrt().clamp_min(1e-8)

@torch.no_grad()
def prepare_features(corpus,source,per_scene,batch=32):
    data={k:[] for k in ('image','base','hit','teacher','gt','source')}
    for idx in corpus.heldout_indices(per_scene).split(batch):
        d=corpus.get(idx);image,base,hit=surface_features(source.core,corpus.source_cache,d['scene'],d['pos'],d['rot'])
        target=refiner_output(source.core.geometry,image)
        if not data['image']:
            a,b=decode(target,base,hit);c,e=source.core.geometry_cached(corpus.source_cache,d['scene'],d['pos'],d['rot'])
            assert (a-c).abs().max()<2e-5 and (b-e).abs().max()<2e-5,'Source feature split is not equivalent'
        data['image'].append(image.half());data['base'].append(base);data['hit'].append(hit)
        labels,box_source=collision_box_labels(corpus,d)
        data['teacher'].append(target.half());data['gt'].append(labels)
        data['source'].append(box_source|torch.tensor([corpus.meta[int(i)]['supervision_source']=='rtx' for i in d['scene']],device=image.device))
    return {k:torch.cat(v).detach() for k,v in data.items()}

def take(data,ids):return {k:v[ids] for k,v in data.items()}

def objective(output,d):
    target=d['teacher'].float();z,l=decode(output,d['base'],d['hit']);tz,tl=decode(target,d['base'],d['hit'])
    valid=tl>0;weight=torch.where(tz<2.,3.,torch.where(tz<4.,1.5,1.))
    metric=((z-tz).abs()*weight*valid).sum()/valid.sum().clamp_min(1)
    correction=F.smooth_l1_loss(output[:,0].tanh(),target[:,0].tanh(),beta=.02)
    mask=F.mse_loss(l.sigmoid(),tl.sigmoid())+.01*F.smooth_l1_loss(output[:,1],target[:,1],beta=.2)
    edge=F.l1_loss(z.diff(dim=-1),tz.diff(dim=-1))+F.l1_loss(z.diff(dim=-2),tz.diff(dim=-2))
    gt=d['gt'].float()/1000.;rtx=(gt>0)&d['source'][:,None,None]
    rtx_weight=torch.where(gt<2.,3.,torch.where(gt<4.,1.5,1.))
    rtx_loss=((z-gt).abs()*rtx*rtx_weight).sum()/rtx.sum().clamp_min(1)
    return metric+correction*.25+mask+edge*.25+.3*rtx_loss

@torch.no_grad()
def validate(model,data,batch=64):
    errs={name:[] for name in ('near','middle','far')};total=0.;count=0
    for ids in torch.arange(len(data['image']),device=data['image'].device).split(batch):
        d=take(data,ids);z,l=decode(model(d['image'].float()),d['base'],d['hit'])
        tz,tl=decode(d['teacher'].float(),d['base'],d['hit']);valid=(tl>0)&(l>0)
        for name,lo,hi in (('near',0,2),('middle',2,4),('far',4,6.01)):
            errs[name].append((z-tz).abs()[valid&(tz>=lo)&(tz<hi)].cpu())
        total+=float(((z-tz).abs()*(tl>0)).sum());count+=int((tl>0).sum())
    return {'mae_m':total/max(1,count),**{k:float(torch.cat(v).mean()) for k,v in errs.items()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',action='append',required=True);p.add_argument('--depth-package',required=True)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch',type=int,default=96)
    p.add_argument('--views-per-scene',type=int,default=32);p.add_argument('--width',type=int,default=8)
    p.add_argument('--lr',type=float,default=5e-4);p.add_argument('--resume');p.add_argument('--device',default='cuda:0')
    p.add_argument('--gradient-every',type=int,default=8);args=p.parse_args();torch.set_num_threads(8);torch.manual_seed(825)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);package=import_package(args.depth_package)
    source=package.DepthInputGenerator(device=args.device,compile_mode='off',precision='fp32')
    train=Corpus(args.data,'train',args.device).prepare(source);val=Corpus(args.data,'val',args.device).prepare(source);assert_disjoint(train,val)
    model=SurfaceRefiner(ModelConfig(args.width)).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    source_hash=sha256(Path(args.depth_package)/'go2_depth_input/assets/weights.pt');code_hash=package_fingerprints(args.depth_package)
    config=dict(vars(args),train=train.provenance(),val=val.provenance(),model=model.configuration(),parameters=sum(p.numel() for p in model.parameters()),
        extra_supervision='Production-box samples receive exact ray/AABB+surrounding-ground labels in addition to source distillation')
    (out/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config),flush=True)
    with ieee_fp32(args.device):
        training=prepare_features(train,source,args.views_per_scene);validation=prepare_features(val,source,16)
        print(json.dumps({'features_ready':len(training['image']),'validation':len(validation['image'])}),flush=True)
        best=float('inf');start=0
        if args.resume:
            ck=torch.load(args.resume,map_location=args.device,weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step'];best=ck['best']
            for group in opt.param_groups:group['lr']=args.lr
            # The resumed supervision recipe may contain newly added physical
            # labels, so select its own best validation checkpoint explicitly.
            best=float('inf')
        began=time.perf_counter()
        for step in range(start+1,args.steps+1):
            ids=torch.randint(len(training['image']),(args.batch,),device=args.device);d=take(training,ids)
            opt.zero_grad(set_to_none=True);loss=objective(model(d['image'].float()),d)
            derivative=loss.new_zeros(())
            if args.gradient_every and step%args.gradient_every==0:
                # Match the CNN's input sensitivity while the exact ray/terrain
                # derivative remains shared. This is not a detached pose shortcut.
                x=d['image'][:4].float().detach().requires_grad_();probe=torch.randn(4,48,64,device=args.device)
                t=refiner_output(source.core.geometry,x)
                target=torch.autograd.grad((t[:,0].tanh()*probe).sum()+.05*(t[:,1]*probe).sum(),x)[0].detach()
                s=model(x);got=torch.autograd.grad((s[:,0].tanh()*probe).sum()+.05*(s[:,1]*probe).sum(),x,create_graph=True)[0]
                derivative=(got-target).square().sum()/target.square().sum().clamp_min(1e-6)
                loss=loss+.02*derivative
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite supervised loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%500==0 or step==args.steps:
                model.eval();score=validate(model,validation);improved=score['mae_m']<best;best=min(best,score['mae_m'])
                ck={'schema':'surface_depth_v2','model':model.state_dict(),'model_config':model.configuration(),'optimizer':opt.state_dict(),
                    'step':step,'best':best,'source_weights_sha256':source_hash,'source_code_sha256':code_hash,'config':config,
                    'training_layouts':[{'name':m['name'],'layout_identity':m['layout_identity']} for m in train.meta],
                    'acceptance':{'passed':False},'validation':score}
                atomic_save(ck,out/'latest.pt')
                if improved:atomic_save(ck,out/'best.pt')
                record={'step':step,'loss':float(loss),'derivative_loss':float(derivative),'validation':score,'seconds':time.perf_counter()-began}
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True);model.train()
        (out/'summary.json').write_text(json.dumps({'step':step,'seconds':time.perf_counter()-began,'best_mae_m':best},indent=2))

if __name__=='__main__':main()
