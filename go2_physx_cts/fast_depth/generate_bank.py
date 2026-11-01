"""Generate explicitly teacher-labelled banks for new maps and full rotations.

This does not turn the frozen model into RTX ground truth. It supplies a
layout-disjoint compatibility/pose-derivative holdout for unlabelled terrains.
"""
import argparse,json,math,time
from pathlib import Path
import torch
from .source import import_package,sha256,ieee_fp32


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--maps',required=True);p.add_argument('--depth-package',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--train-views',type=int,default=512);p.add_argument('--holdout-views',type=int,default=128)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--seed',type=int,default=824)
    args=p.parse_args();torch.set_num_threads(8)
    package=import_package(args.depth_package)
    from go2_depth_input.vendor.camera import sample_poses,rpy_matrix,optical_rotation,height_at,contract
    source=package.DepthInputGenerator(device=args.device,chunk_size=args.batch,compile_mode='off',precision='fp32')
    bundle=torch.load(args.maps,map_location='cpu',weights_only=False)
    maps=bundle['maps'].float();metadata=bundle['metadata']
    if maps.shape!=(len(metadata),512,512):raise ValueError('Expected maps[S,512,512] and matching metadata')
    for split in ('train','val','test'):
        selection=[i for i,m in enumerate(metadata) if m['split']==split]
        if not selection:continue
        out=Path(args.output)/(split+'_direct_teacher')
        if out.exists():raise FileExistsError(f'Refusing to overwrite existing bank: {out}')
        out.mkdir(parents=True)
        chosen=maps[selection];meta=[dict(metadata[i],supervision_source='frozen_teacher',layout_id=str(metadata[i].get('layout_id',selection[j]))) for j,i in enumerate(selection)]
        torch.save(chosen.half(),out/'maps.pt')
        manifest={'schema':'direct_depth_bank_v1','split':split,'scenes':meta,
                  'supervision_source':'frozen_teacher','source_sha256':sha256(Path(args.depth_package)/'go2_depth_input/assets/weights.pt'),
                  'map_bundle_sha256':sha256(args.maps),'camera':contract(),
                  'pose_families':{'0':'nominal','1':'transition','2':'fallen','3':'full_orientation'},
                  'caveat':'Teacher compatibility labels, not new RTX or real-world geometry evidence.'}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        cache=source.prepare_scenes(chosen);gen=torch.Generator(device=args.device).manual_seed(args.seed+len(split))
        count=args.train_views if split=='train' else args.holdout_views
        if count<4:raise ValueError('At least 4 views per scene required')
        began=time.perf_counter()
        for j in range(len(chosen)):
            ids=torch.full((count,),j,dtype=torch.long,device=args.device)
            pos,rot,mode=sample_poses(cache.tensors[0],ids,gen)
            # Explicitly cover full rotation, including roll/pitch beyond upright
            # gait. These are geometric visibility samples, not recovery dynamics.
            q=count//4
            rpy=(torch.rand(q,3,device=args.device,generator=gen)-.5)*2*math.pi
            rot[-q:]=optical_rotation(rpy_matrix(rpy));mode[-q:]=3
            # Include near-spawn and feature-directed views supplied by the
            # production arena bundle; random yaw keeps all view directions.
            spawn=meta[j].get('body_spawn_local')
            if spawn is not None:
                xy=pos[:count//2,:2]
                xy.copy_(xy*.35+xy.new_tensor(spawn[:2]))
                floor=height_at(cache.tensors[0],ids[:count//2],xy)
                pos[:count//2,2]=floor+.45+torch.rand(count//2,device=args.device,generator=gen)*.35
            depths=[]
            with torch.no_grad(),ieee_fp32(args.device):
                for start in range(0,count,args.batch):
                    stop=min(start+args.batch,count)
                    z,logit=source.core.geometry_cached(cache.tensors,ids[start:stop],pos[start:stop],rot[start:stop])
                    # 0..6000 mm fits signed int16, including PyTorch versions
                    # that cannot serialize uint16 storage yet.
                    depths.append(torch.where(logit>0,(z*1000).round(),0).to(torch.int16).cpu())
            torch.save({'depth':torch.cat(depths),'pos':pos.cpu(),'rot':rot.cpu(),
                        'scene':ids.cpu().to(torch.int16),'mode':mode.cpu().to(torch.int8)},out/f'views_{j:04d}.pt')
            print(json.dumps({'split':split,'scene':j+1,'scenes':len(chosen),'seconds':time.perf_counter()-began}),flush=True)
        (out/'COMPLETE').write_text('teacher compatibility supervision; RTX not available\n')


if __name__=='__main__':main()
