"""Real event coverage using uniform, recent-event and bounded success history.

The archive stores episode windows, not duplicated root/burn-in pairs. Images,
encoder latents and GRU states are never cached here. All observation poses and
noise keys are retained so the *current* encoder reconstructs sequence memory.
"""
import torch
from .buffer import FLOAT_FIELDS, INT_FIELDS, BOOL_FIELDS, PackedTransport
from .task import COMPLETED

PREFIX_FIELDS = ('obs','depth_pose','depth_frame_index','depth_scene_ids','episode')


def _prefix_minimal(seed):
    if 'prefix' in seed:
        seed['prefix']={key:seed['prefix'][key] for key in
            (*PREFIX_FIELDS,'valid','reset','env','trajectory_ids')}
    return seed


@torch.no_grad()
def _seeds_at(replay, envs, slots, burnin):
    out=replay._gather(envs,slots)
    if burnin:
        logical=(slots-(replay.ptr-replay.size))%replay.L
        before=logical[:,None]-torch.arange(burnin,0,-1,device=replay.index_device)
        prefix=replay._gather(envs,replay.physical(before.clamp_min(0)))
        valid=(before>=0).to(replay.device)&(prefix['episode'][...,0]==out['episode'][:,0,None])
        prefix['valid']=valid;prefix['reset']=~valid;prefix['reset'][:,0]=True
        out['prefix']=prefix
    return _prefix_minimal(out)


class EventSeedArchive:
    """CPU success-only windows, independently bounded per source and event task.

    Four FIFO partitions (teacher/student x crossing/recovery) prevent frequent
    teacher crossings from evicting every student recovery. Failures never enter.
    max_admissions_per_observe limits transfer work when success becomes common.
    """
    schema='as2_real_event_archive_v1'

    def __init__(self,capacity_episodes=256,history_steps=100,burnin=16,
                 max_admissions_per_observe=4):
        if capacity_episodes<4 or capacity_episodes%4 or min(history_steps,burnin)<0:
            raise ValueError('Event archive needs four equal nonempty partitions')
        self.capacity=int(capacity_episodes);self.per_group=self.capacity//4
        self.history_steps=int(history_steps);self.burnin=int(burnin)
        self.span=self.history_steps+1+self.burnin
        self.max_admissions=int(max_admissions_per_observe)
        if self.max_admissions<1:raise ValueError('Archive admissions must be positive')
        self.ptr=torch.zeros(4,dtype=torch.long);self.size=torch.zeros(4,dtype=torch.long)
        self.float_slices={};self.int_slices={};cursor=0
        for key,dim in FLOAT_FIELDS.items():self.float_slices[key]=slice(cursor,cursor+dim);cursor+=dim
        self.float_dim=cursor
        for i,key in enumerate((*INT_FIELDS,*BOOL_FIELDS)):self.int_slices[key]=slice(i,i+1)
        self.int_dim=len(self.int_slices)
        self.floats=self.ints=self.valid=self.envs=None
        self.total_admitted=0;self.total_success_candidates=0;self.last_observed=-1
        self.last_sample_stats={}
        self.transport=PackedTransport()

    def _allocate(self):
        if self.floats is not None:return
        # Ordinary CPU tensors; a fixed-size staging buffer handles transfers.
        self.floats=torch.empty((self.capacity,self.span,self.float_dim),dtype=torch.float32)
        self.ints=torch.empty((self.capacity,self.span,self.int_dim),dtype=torch.int64)
        self.valid=torch.zeros((self.capacity,self.span),dtype=torch.bool)
        self.envs=torch.zeros(self.capacity,dtype=torch.long)

    @property
    def storage_bytes(self):
        return sum(x.numel()*x.element_size() for x in (self.floats,self.ints,self.valid,self.envs)
                   if x is not None)

    @torch.no_grad()
    def observe(self,replay,done=None):
        """Call immediately after Replay.add; actual success can precede done."""
        if replay.size==0 or replay.total_collections==self.last_observed:return 0
        self.last_observed=replay.total_collections;slot=(replay.ptr-1)%replay.L
        success=replay.data['success'][:,slot,0]&~replay.heldout[:,slot]
        task=replay.data['task_ids'][:,slot,0].long()
        source=replay.data['source'][:,slot,0].long()
        success&=((task==3)|(task==4))&((source==0)|(source==1))
        candidates=success.nonzero().flatten()
        if not len(candidates):return 0
        self.total_success_candidates+=len(candidates)
        group=source*2+(task-3)
        chosen=[]
        def index_draw(value):
            return replay.transport.move({'index':value},replay.index_device)['index']
        # Four group operations, never one Python operation per trajectory step.
        for g in range(4):
            ids=candidates[group[candidates]==g]
            if len(ids):chosen.append(ids[index_draw(torch.randint(len(ids),(1,),device=replay.device))])
        selected=torch.cat(chosen)
        if len(selected)>self.max_admissions:
            selected=selected[index_draw(torch.randperm(len(selected),device=replay.device)[:self.max_admissions])]
        elif len(selected)<self.max_admissions:
            remaining=candidates[~torch.isin(candidates,selected)]
            count=min(self.max_admissions-len(selected),len(remaining))
            if count:selected=torch.cat((selected,remaining[index_draw(torch.randperm(len(remaining),device=replay.device)[:count])]))
        logical=replay.size-1-torch.arange(self.span-1,-1,-1,device=replay.index_device)
        slots=replay.physical(logical.clamp_min(0))
        data={key:value[selected[:,None],slots[None,:]] for key,value in replay.data.items()}
        same=data['episode'][...,0]==data['episode'][:,-1,0,None]
        valid=same&(logical>=0)[None,:]
        # No in-flight Replay views escape: both transfers finish before return.
        transfer=self.transport.move({'floats':torch.cat([data[key].float() for key in self.float_slices],-1),
            'ints':torch.cat([data[key].long() for key in self.int_slices],-1),
            'groups':group[selected],'envs':selected,'valid':valid},'cpu')
        packed_f,packed_i=transfer['floats'],transfer['ints']
        groups,selected_cpu,valid=transfer['groups'],transfer['envs'],transfer['valid']
        self._allocate()
        for g in range(4):
            rows=(groups==g).nonzero().flatten()
            if not len(rows):continue
            rows=rows[-self.per_group:]
            dst=g*self.per_group+(self.ptr[g]+torch.arange(len(rows)))%self.per_group
            self.floats[dst]=packed_f[rows];self.ints[dst]=packed_i[rows]
            self.valid[dst]=valid[rows];self.envs[dst]=selected_cpu[rows]
            self.ptr[g]=(self.ptr[g]+len(rows))%self.per_group
            self.size[g]=min(self.per_group,int(self.size[g])+len(rows))
        self.total_admitted+=len(selected)
        return len(selected)

    def _roots(self,source):
        if self.valid is None:return torch.empty((0,2),dtype=torch.long)
        groups=(source*2,source*2+1)
        rows=torch.cat([g*self.per_group+torch.arange(int(self.size[g])) for g in groups])
        rootmask=self.valid[rows].clone();rootmask[:,:self.burnin]=False
        found=rootmask.nonzero()
        if len(found):found[:,0]=rows[found[:,0]]
        return found

    def _unpack(self,floats,ints):
        out={key:floats[...,part] for key,part in self.float_slices.items()}
        for key,part in self.int_slices.items():
            dtype=INT_FIELDS[key] if key in INT_FIELDS else torch.bool
            out[key]=ints[...,part].to(dtype=dtype)
        return out

    @torch.no_grad()
    def sample(self,batch,source,burnin,device):
        if burnin>self.burnin:raise ValueError('Requested burn-in exceeds archived exact history')
        roots=self._roots(source)
        if batch<=0 or not len(roots):return None
        chosen=roots[torch.randint(len(roots),(batch,))];rows,t=chosen.unbind(-1)
        transfer={'floats':self.floats[rows,t],'ints':self.ints[rows,t],'envs':self.envs[rows]}
        if burnin:
            previous=t[:,None]-torch.arange(burnin,0,-1)[None,:]
            positions=previous.clamp_min(0)
            # Only five capture/observation fields enter the GRU prefix. Pack
            # them once each to avoid dozens of tiny PCIe transfers.
            obs=self.floats[rows[:,None],positions,self.float_slices['obs']]
            poses=self.floats[rows[:,None],positions,self.float_slices['depth_pose']]
            transfer['prefix_floats']=torch.cat((obs,poses),-1)
            keys=('depth_frame_index','depth_scene_ids','episode')
            transfer['prefix_ints']=torch.cat([self.ints[rows[:,None],positions,self.int_slices[k]] for k in keys],-1)
            valid=self.valid[rows[:,None],positions]&(previous>=0)
            valid&=(self.ints[rows[:,None],positions,self.int_slices['episode']].squeeze(-1)
                    ==self.ints[rows,t,self.int_slices['episode']])
            transfer['prefix_valid']=valid
        transfer=self.transport.move(transfer,device)
        out=self._unpack(transfer['floats'],transfer['ints']);envs=transfer['envs']
        out['env']=envs;out['trajectory_ids']=envs
        if burnin:
            fp,ip,valid=transfer['prefix_floats'],transfer['prefix_ints'],transfer['prefix_valid']
            reset=~valid;reset[:,0]=True
            prefix={'obs':fp[...,:450],'depth_pose':fp[...,450:],
                **{k:ip[...,i:i+1].to(INT_FIELDS[k]) for i,k in enumerate(keys)},
                'valid':valid,'reset':reset,'env':envs,'trajectory_ids':envs}
            out['prefix']=prefix
        return out

    def state_dict(self):
        return {'schema':self.schema,'capacity':self.capacity,'history_steps':self.history_steps,
            'burnin':self.burnin,'max_admissions':self.max_admissions,
            **{key:getattr(self,key) for key in ('ptr','size','floats','ints','valid','envs',
                'total_admitted','total_success_candidates','last_observed')}}

    def load_state_dict(self,state):
        if (state.get('schema')!=self.schema or state['capacity']!=self.capacity
                or state['history_steps']!=self.history_steps or state['burnin']!=self.burnin):
            raise ValueError('Event archive schema/configuration mismatch')
        for key in ('ptr','size','floats','ints','valid','envs','total_admitted','total_success_candidates','last_observed'):
            value=state[key];setattr(self,key,value.cpu() if torch.is_tensor(value) else value)


def _concat(parts):
    return {key:(_concat([part[key] for part in parts]) if isinstance(parts[0][key],dict)
                 else torch.cat([part[key] for part in parts],0)) for key in parts[0]}


@torch.no_grad()
def coverage_from_seeds(seed):
    tasks=seed['task_ids'].reshape(-1).long()
    success=seed['success'].reshape(-1).bool();failure=seed['terminated'].reshape(-1).bool()
    completed=seed['task_context'][:,COMPLETED]>=.5
    states=torch.stack((torch.bincount(tasks,minlength=5),
        torch.bincount(tasks[success],minlength=5),torch.bincount(tasks[failure],minlength=5),
        torch.bincount(tasks[completed],minlength=5)),1).cpu().tolist()
    return {str(i):dict(zip(('states','success','failure','already_completed'),row)) for i,row in enumerate(states)}


@torch.no_grad()
def sample_event_seeds(replay,archive,batch,source,burnin):
    """50/25/25 recent uniform/event/success-history; empty buckets use uniform."""
    if source not in (0,1) or batch<1:raise ValueError('An own-policy source and positive batch are required')
    quarter=batch//4;parts=[];near_count=0;archive_count=0
    if quarter and archive is not None:
        stored=archive.sample(quarter,source,burnin,replay.device)
        if stored is not None:parts.append(stored);archive_count=len(stored['s'])
    remaining=quarter;envs=[];slots=[]
    for _ in range(4):
        if not remaining:break
        e,t=replay.sample_states(max(32,remaining*4),source=source)
        selected=replay.event_mask(e,t).nonzero().flatten()[:remaining]
        if len(selected):envs.append(e[selected]);slots.append(t[selected]);remaining-=len(selected)
    if envs:
        near_count=quarter-remaining
        parts.append(_seeds_at(replay,torch.cat(envs),torch.cat(slots),burnin))
    uniform=batch-near_count-archive_count
    if uniform:parts.append(_prefix_minimal(replay.seeds(uniform,source=source,burnin=burnin)))
    seed=_concat(parts)
    if archive is not None:
        archive.last_sample_stats={'source':source,'uniform':uniform,'near_event':near_count,
            'long_success_history':archive_count,'tasks':coverage_from_seeds(seed),
            'archive_episodes_by_group':archive.size.tolist(),'archive_cpu_bytes':archive.storage_bytes}
    return seed
