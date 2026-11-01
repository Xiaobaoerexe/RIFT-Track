"""Layout-disjoint supervision banks; RTX and frozen-teacher labels stay distinct."""
import json
from pathlib import Path
import torch


class Corpus:
    def __init__(self, roots, split, device='cuda:0'):
        self.device = torch.device(device)
        maps, rows, pairs, self.meta, self.banks = [], [], [], [], []
        base = 0
        for root in roots:
            root = Path(root)
            banks = [root] if (root/'manifest.json').exists() and root.name.startswith(split+'_') else sorted(root.glob(split+'_*'))
            for bank in banks:
                if not (bank/'COMPLETE').exists():
                    raise RuntimeError(f'Incomplete bank: {bank}')
                manifest = json.loads((bank/'manifest.json').read_text())
                bank_maps = torch.load(bank/'maps.pt', map_location='cpu', weights_only=True).float()
                scenes = manifest['scenes']
                if len(scenes) != len(bank_maps):
                    raise ValueError(f'Metadata/map mismatch: {bank}')
                source = manifest.get('supervision_source', 'rtx')
                for i, scene in enumerate(scenes):
                    item = dict(scene, bank=str(bank), supervision_source=scene.get('supervision_source', source))
                    item['layout_identity'] = str(scene.get('layout_id', scene.get('seed',f'{bank.name}:{i}')))
                    self.meta.append(item)
                self.banks.append({'path':str(bank), 'manifest':manifest})
                maps.append(bank_maps)
                for path in sorted(bank.glob('views_*.pt')):
                    d = torch.load(path, map_location='cpu', weights_only=True)
                    n = len(d['scene']); layout = d.get('pair_layout')
                    rows.append({'depth':d['depth'].to(torch.int16), 'pos':d['pos'].float(),
                                 'rot':d['rot'].float(), 'scene':d['scene'].long()+base,
                                 'mode':d['mode'].long()})
                    if layout is None:
                        pairs.append(torch.zeros(n, 3, dtype=torch.long))
                    else:
                        at = torch.arange(n)
                        pairs.append(torch.stack(((at % layout['batch'] < layout['anchor']).long(),
                            torch.full((n,),layout['plus_start']), torch.full((n,),layout['minus_start'])), -1))
                base += len(bank_maps)
        if not maps or not rows:
            raise FileNotFoundError(f'No completed {split} supervision banks in {roots}')
        self.maps = torch.cat(maps)
        self.data = {k:torch.cat([r[k] for r in rows]) for k in rows[0]}
        self.pairs = torch.cat(pairs)
        self.count = len(self.data['scene'])
        self.scene_indices = [torch.where(self.data['scene']==i)[0] for i in range(base)]
        if any(len(v)==0 for v in self.scene_indices):
            raise ValueError('Every scene must contain supervision')
        self.source_cache = None

    def prepare(self, source):
        """Only immutable source terrain features are cached."""
        chunks = []
        for maps in self.maps.split(16):
            cache = source.prepare_scenes(maps)
            chunks.append(cache.tensors[:3])
        maps, local, context = (torch.cat([c[i] for c in chunks]) for i in range(3))
        flat = lambda x:x.permute(0,2,3,1).contiguous().reshape(-1,x.shape[1])
        self.source_cache = (maps,local,context,flat(maps[:,None]),flat(local),flat(context))
        return self

    def get(self, indices):
        indices = indices.cpu()
        d = {k:v[indices].to(self.device) for k,v in self.data.items()}
        unique, inverse = torch.unique(d['scene'], sorted=True, return_inverse=True)
        d['unique_scene'], d['local_scene'] = unique, inverse
        return d

    def batch(self, views, generator, scenes=8):
        nscenes = min(scenes,len(self.maps),views)
        sid = torch.randperm(len(self.maps),generator=generator)[:nscenes]
        indices = []
        for j,s in enumerate(sid.tolist()):
            n = views//nscenes+(j<views%nscenes)
            pool = self.scene_indices[s]
            indices.append(pool[torch.randint(len(pool),(n,),generator=generator)])
        return self.get(torch.cat(indices))

    def heldout_indices(self, per_scene=16):
        """Deterministic coverage of each layout and every available orientation family."""
        selected = []
        for pool in self.scene_indices:
            families = torch.unique(self.data['mode'][pool]).tolist()
            for mode in families:
                ids = pool[self.data['mode'][pool]==mode]
                count = min(len(ids),max(1,per_scene//len(families)))
                selected.append(ids[torch.linspace(0,len(ids)-1,count).long()])
        return torch.cat(selected)

    def provenance(self):
        return {'frames':self.count,'scenes':len(self.maps),'banks':[b['path'] for b in self.banks],
                'sources':sorted(set(m['supervision_source'] for m in self.meta)),
                'terrains':sorted(set(m['name'] for m in self.meta))}


def assert_disjoint(train, holdout):
    """Check explicit identities, rather than treating identical flat maps as leakage."""
    a={(m['name'],m['layout_identity']) for m in train.meta}
    b={(m['name'],m['layout_identity']) for m in holdout.meta}
    overlap=a&b
    if overlap:
        raise ValueError(f'Train/holdout layout identity overlap: {sorted(overlap)[:8]}')


def collision_box_labels(corpus,d):
    """Independent metric labels from the production AABBs, where supplied.

    These are analytic collision-mesh labels, not rendered RTX or teacher data.
    Existing RTX samples pass through without replacement.
    """
    from go2_depth_input.vendor.camera import rays
    ray=rays(d['pos'].device).reshape(-1,3)
    directions=torch.einsum('bij,rj->bri',d['rot'],ray)
    labels=d['depth'].clone();exact=torch.zeros(len(labels),dtype=torch.bool,device=labels.device)
    for i,sid in enumerate(d['scene'].tolist()):
        m=corpus.meta[sid]
        if 'boxes' not in m:continue
        boxes=d['pos'].new_tensor(m['boxes']);p=d['pos'][i]+d['pos'].new_tensor(m['centre']);direction=directions[i]
        safe=torch.where(direction.abs()>1e-8,direction,torch.full_like(direction,1e-8))
        a=(boxes[:,None,0,:]-p)/safe;b=(boxes[:,None,1,:]-p)/safe
        entry=torch.minimum(a,b).amax(-1);leave=torch.maximum(a,b).amin(-1)
        hit=(leave>=entry)&(entry>=.1)&(entry<=6.)
        depth=torch.where(hit,entry,torch.full_like(entry,100.)).amin(0)
        plane=-p[2]/safe[:,2];xy=p[None,:2]+direction[:,:2]*plane[:,None]
        lo=boxes[:,0,:2].amin(0);hi=boxes[:,1,:2].amax(0)
        hit=((xy<lo)|(xy>hi)).any(-1)&(plane>=.1)&(plane<=6.)
        depth=torch.minimum(depth,torch.where(hit,plane,torch.full_like(plane,100.)))
        labels[i]=torch.where(depth<=6.,(depth*1000).round(),0.).reshape(48,64).to(labels.dtype);exact[i]=True
    return labels,exact
