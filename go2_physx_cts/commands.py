"""Functional port of Go2RLLab's pinned mixed-task command sampler.

Source: wty-yy/go2_rl_lab 6c95d9feef4814c25eb6536e61bb38e1d1d718d4,
cmd_utils/rl_gym_cmd.py and tasks/mixed/env_cfg.py:CommandsCfg.
The caller supplies learner iterations (num_steps_per_iter=1), not physics
substeps. Simple mode preserves the pinned sampler; events/moe add the approved per-terrain grids.

All command memory is a packed floating tensor suitable for replay. Random
choices may be supplied explicitly; no state tensor is mutated or detached.
"""
from itertools import product
import math
import torch

SOURCE_COMMIT = '6c95d9feef4814c25eb6536e61bb38e1d1d718d4'
CMD_STATE_DIM = 21
GRID_BIN,TASK_ID,TERRAIN_ID,COMPLETED = 17,18,19,20
TERRAIN_NAMES=("wave","slope_up","slope_down","rough_slope","stairs_up","stairs_down","obstacles","flat","wheels","small_squares","gap","obstacle_up","obstacle_down")
TASK_TERRAINS=(tuple(range(10)),(7,),(7,),(10,11,12),(7,))
FINAL_TASK_COUNTS=(820,819,819,819,819)

def phase(iteration):return 0 if iteration<2500 else 1 if iteration<5000 else 2

def task_assignments(n,iteration,device,variant="simple"):
    active=1 if variant=="simple" or iteration<5000 else 5
    counts=[n//active+(i<n%active) for i in range(active)]
    return torch.repeat_interleave(torch.arange(active,device=device),torch.tensor(counts,device=device))

def speed_cap(task,iteration):
    return .5 if iteration<5000 else 1. if iteration<10000 else 2. if iteration<20000 else 4. if iteration<30000 else 6. if iteration<35000 else 8.

REQUEST, COMMAND = slice(0, 3), slice(3, 6)
TIMER, ACCUMULATION, EPISODE_ELAPSED = 6, slice(7, 9), 9
LAST_LIMIT, COMMAND_COUNTER, MAX_DISTANCE = 10, 11, 12
ORIGIN, RANGE_STAGE, LIMIT_STAGE = slice(13, 15), 15, 16
COMMAND_NAMES = ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')
RANGE_ITERATIONS = (5000, 10000, 20000, 30000, 35000)
RANGE_STAGES = (
    ((-.5, .5), (-.5, .5), (-1., 1.)),
    ((-1., 1.), (-1., 1.), (-2., 2.)),
    ((-2., 2.), (-1., 1.), (-3., 3.)),
    ((-3., 4.), (-1., 1.), (-3., 3.)),
    ((-4., 6.), (-1., 1.), (-3., 3.)),
    ((-4.5, 8.), (-1., 1.), (-3., 3.)),
)
TERRAIN_RANGES = {
    'flat': ((-4.5, 8.), (-1., 1.), (-3., 3.)),
    'wave': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'slope_up': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'slope_down': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'rough_slope': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'stairs_up': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'stairs_down': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'obstacles': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'wheels': ((-2., 1.5), (-1., 1.), (-1.5, 1.5)),
    'small_squares': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'gap': ((.2, 2.), (0., 0.), (0., 0.)),
    'obstacle_up': ((.2, 2.), (0., 0.), (0., 0.)),
    'obstacle_down': ((.2, 2.), (0., 0.), (0., 0.)),
}


def special_speed_contract(episode_seconds=8.,target_radius=.8,start_allowance=.5):
    """Project reachability bound, not an upstream curriculum formula.

    The most distant accepted region is level9 gap: spawn=6-(.2+.07*9)-1.4,
    entry=8.4-radius. Reserve32 stable control ticks and startup allowance.
    At8seconds this requires >=.6m/s; the initial1m/s envelope contains a
    real frontier cell. This only guarantees kinematic reachability, not jumps.
    """
    episode_seconds=float(episode_seconds)
    distance=8.4-float(target_radius)-(6.-(.2+.07*9)-1.4)
    stable_seconds=32*.02;moving_time=episode_seconds-stable_seconds-float(start_allowance)
    if not math.isfinite(moving_time) or moving_time<=0:raise ValueError('Special episode leaves no time for approach and stable completion')
    lower=math.ceil((distance/moving_time-1e-10)*10.)/10.
    minimum_extent=math.ceil((lower-1e-10)/.5)*.5
    if lower>2. or minimum_extent>2.:raise ValueError('Special episode is unreachable within approved2m/s command cap')
    return {'episode_seconds':episode_seconds,'max_distance_m':distance,'stable_seconds':stable_seconds,
        'start_allowance_seconds':float(start_allowance),'minimum_speed_mps':lower,'minimum_extent_mps':minimum_extent,
        'initial_extent_mps':max(1.,minimum_extent),'maximum_speed_mps':2.}


def zero_command_prob(iteration):
    """Official linear zero-command ramp: 0 at iteration0, .1 at1500."""
    if isinstance(iteration, torch.Tensor):
        return (iteration / 1500.).clamp(0., 1.) * .1
    return min(max(float(iteration) / 1500., 0.), 1.) * .1


def coupled_clip(command):
    """Apply official priority rules after terrain clipping, without in-place ops."""
    x, y, yaw = command.unbind(-1)
    fast = x.abs() > 2.
    y = torch.where(fast, y.clamp(-.5, .5), y)
    yaw = torch.where(fast, yaw.clamp(-1., 1.), yaw)
    turning = yaw.abs() > 2.
    x = torch.where(turning, x.clamp(-2., 2.), x)
    y = torch.where(turning, y.clamp(-1., 1.), y)
    return torch.stack((x, y, yaw), -1)


class GridTerrainLookup:
    """Official half-open rectangular grid lookup; outside IDs are -1.

    Metadata axes match Isaac terrain grids: rows along x, columns along y.
    ``origin_xy`` is the lower grid corner, excluding the surrounding border;
    ``column_ids`` maps each column to an index in ``names``. The PhysX mesh
    adapter must supply these from its actual generator, not infer from robots.
    """
    def __init__(self, origin_xy, tile_size, rows, cols, column_ids, names, device='cpu'):
        self.origin_xy = torch.as_tensor(origin_xy, device=device, dtype=torch.float32)
        self.tile_size = torch.as_tensor(tile_size, device=device, dtype=torch.float32)
        self.rows, self.cols, self.names = int(rows), int(cols), tuple(names)
        self.column_ids = torch.as_tensor(column_ids, device=device, dtype=torch.long)
        if (self.column_ids.shape != (self.cols,) or self.origin_xy.shape != (2,)
                or self.tile_size.shape != (2,) or min(self.rows, self.cols) < 1
                or not bool((self.tile_size > 0).all())
                or not bool(((self.column_ids >= 0) & (self.column_ids < len(self.names))).all())):
            raise ValueError('Invalid terrain grid metadata.')

    def query(self, position_xy):
        cell = torch.floor((position_xy[..., :2] - self.origin_xy.to(position_xy))
                           / self.tile_size.to(position_xy)).long()
        valid = ((cell[..., 0] >= 0) & (cell[..., 0] < self.rows)
                 & (cell[..., 1] >= 0) & (cell[..., 1] < self.cols))
        ids = self.column_ids.to(position_xy.device)[cell[..., 1].clamp(0, self.cols - 1)]
        return torch.where(valid, ids, torch.full_like(ids, -1))



class VelocityGrid:
    """Rapid weighted-cell sampling, separated by task AND physical terrain.

    Source f5143ef940e934849c00284e34caf164d6ce7b6e. The approved adaptation
    uses own-policy commanded progress and hard special-task success every1k.
    Recovery is excluded. Neighbor weights and progress envelopes checkpoint.
    """
    source_commit='f5143ef940e934849c00284e34caf164d6ce7b6e'
    def __init__(self,device='cpu',special_episode_seconds=8.):
        self.special_contract=special_speed_contract(special_episode_seconds)
        axes=(torch.arange(-9,17,device=device)*.5,torch.arange(-5,6,device=device)*.2,torch.arange(-30,31,device=device)*.1)
        self.shape=tuple(len(x) for x in axes);self.grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        self.width=self.grid.new_tensor([.5,.2,.1]);self.extent=self.grid.new_tensor([.5,.2,.1]).expand(5,13,3).clone()
        self.extent[3,:,0]=self.special_contract["initial_extent_mps"]
        initial=(self.grid.abs()<=self.width+1e-6).all(-1)
        self.weights=initial.float().expand(5,13,-1).clone()
        special=(self.grid[:,0]>=self.special_contract["minimum_speed_mps"]-.25)&(self.grid[:,0]<=self.special_contract["initial_extent_mps"]+.25)&(self.grid[:,1:]==0).all(-1)
        self.weights[3]=special.float()[None]
        self.sum=torch.zeros_like(self.weights);self.count=torch.zeros_like(self.weights)
        self.last_update=0;self.last_statistics={};self._cache_key=None
    def bin_indices(self,command):
        i=(command/self.width).round().long()+command.new_tensor([9,5,30],dtype=torch.long)
        i=torch.minimum(torch.maximum(i,torch.zeros_like(i)),i.new_tensor([25,10,60]))
        return (i[:,0]*11+i[:,1])*61+i[:,2]
    def state_dict(self):
        return {k:getattr(self,k).clone() for k in ('weights','extent','sum','count')}|{'last_update':self.last_update,'statistics':self.last_statistics,'special_contract':dict(self.special_contract)}
    def load_state_dict(self,s):
        if s.get('special_contract',self.special_contract)!=self.special_contract:raise ValueError('Special task feasibility contract changed across resume')
        for k in ('weights','extent','sum','count'):getattr(self,k).copy_(s[k])
        self.last_update=int(s['last_update']);self.last_statistics=s.get('statistics',{});self._cache_key=None
    @torch.no_grad()
    def record(self,task,terrain,bins,progress,own):
        valid=own&(task>=0)&(task<4)&(terrain>=0)&(terrain<13)&(bins>=0)&(bins<len(self.grid))&torch.isfinite(progress)
        index=(task[valid]*13+terrain[valid])*len(self.grid)+bins[valid]
        self.sum.view(-1).scatter_add_(0,index,progress[valid].clamp(0,1))
        self.count.view(-1).scatter_add_(0,index,torch.ones_like(progress[valid]))
    @torch.no_grad()
    def update(self,iteration):
        if iteration<self.last_update+1000:return {}
        results={};ranges=self.grid.new_tensor([TERRAIN_RANGES[n] for n in TERRAIN_NAMES])
        for task in range(4):
            terrains={}
            for terrain in TASK_TERRAINS[task]:
                count=self.count[task,terrain];sums=self.sum[task,terrain]
                if not bool((count>0).any()):continue
                # Measure the sampled frontier, rather than letting standing
                # dominate whether the next speed band is learnable.
                frontier=self.grid[:,0].abs()>=(self.extent[task,terrain,0]-.25).clamp_min(.25)
                considered=count*frontier
                if not bool((considered>0).any()):continue
                score=(sums*frontier).sum()/considered.sum();old=self.extent[task,terrain].clone()
                if score>.75:self.extent[task,terrain]+=self.width
                elif score<.5:self.extent[task,terrain]=torch.maximum(self.extent[task,terrain]-self.width,self.width)
                cap=ranges[terrain].abs().amax(-1);cap[0]=min(float(cap[0]),speed_cap(task,iteration))
                self.extent[task,terrain]=torch.minimum(self.extent[task,terrain],torch.maximum(cap,self.width))
                if task==3:self.extent[task,terrain,0].clamp_(min=self.special_contract["minimum_extent_mps"],max=2.)
                mean=sums/count.clamp_min(1);good=((count>0)&(mean>.75)).reshape(self.shape)
                neighbor=torch.nn.functional.max_pool3d(good.float()[None,None],3,stride=1,padding=1)[0,0].flatten()
                self.weights[task,terrain].add_(neighbor*.2).clamp_(max=1.)
                terrains[TERRAIN_NAMES[terrain]]={'progress':float(score),'old_extent':old.tolist(),'extent':self.extent[task,terrain].tolist(),'samples':int(count.sum())}
            if terrains:results[str(task)]={'terrains':terrains,'macro_progress':sum(v['progress'] for v in terrains.values())/len(terrains)}
        self.sum.zero_();self.count.zero_();self.last_update=int(iteration);self.last_statistics=results;self._cache_key=None
        return results
    def sample(self,task,terrain,iteration,generator=None):
        stage=sum(iteration>=v for v in RANGE_ITERATIONS);key=(stage,self.last_update)
        if key!=self._cache_key:
            caps=self.grid.new_tensor(RANGE_STAGES[stage]);local=self.grid.new_tensor([TERRAIN_RANGES[n] for n in TERRAIN_NAMES])
            lower=torch.maximum(local[:,:,0],caps[:,0]);upper=torch.minimum(local[:,:,1],caps[:,1])
            if iteration>=5000 and self.special_contract["minimum_speed_mps"]>speed_cap(3,iteration):
                raise ValueError("Special deadline is unreachable under this iteration speed cap")
            extent=self.extent.reshape(-1,3);lo=lower.repeat(5,1);hi=upper.repeat(5,1)
            lo=torch.maximum(lo,-extent);hi=torch.minimum(hi,extent)
            # Pure forward channel before special success; lower speed remains
            # available at early curriculum, rather than unconditionally1m/s.
            lo[13:26,0]=torch.minimum(hi[13:26,0],torch.full_like(hi[13:26,0],.5))
            lo[3*13:4*13,0]=self.special_contract["minimum_speed_mps"]
            lo[3*13:4*13,1:]=0.;hi[3*13:4*13,1:]=0.
            g=self.grid[None];enabled=((g>=lo[:,None]-self.width*.5-1e-6)&(g<=hi[:,None]+self.width*.5+1e-6)).all(-1)
            weights=self.weights.flatten(0,1)*enabled
            weights=torch.where((weights.sum(-1)==0)[:,None],enabled.float(),weights)
            self._cdf=weights.cumsum(-1)/weights.sum(-1,keepdim=True).clamp_min(1e-9)
            self._lo,self._hi=lo,hi;self._cache_key=key
        terrain=terrain.clamp(0,12);key=task.clamp(0,4)*13+terrain;n=len(task)
        u=torch.rand(65,n,device=task.device,generator=generator)
        index=torch.searchsorted(self._cdf.contiguous(),u.contiguous()).clamp_max(len(self.grid)-1)[key,torch.arange(n,device=task.device)]
        sample=self.grid[index]+(torch.rand(n,3,device=task.device,generator=generator)-.5)*self.width
        sample=torch.maximum(torch.minimum(sample,self._hi[key]),self._lo[key])
        # At least30% of samples per task use only vx, with random rotating rank.
        membership=torch.nn.functional.one_hot(task,5);rank=(membership.cumsum(0)*membership).sum(-1)-1;totals=membership.sum(0).clamp_min(1)
        offset=(torch.rand(5,device=task.device,generator=generator)*totals).long()
        pure=((rank-offset[task])%totals[task])<torch.ceil(.3*totals[task]).long()
        sample=torch.stack((sample[:,0],torch.where(pure,0.,sample[:,1]),torch.where(pure,0.,sample[:,2])),-1)
        return coupled_clip(sample),index

class OfficialCommands:
    """One functional command transition for both real and imagined rollouts.

    ``initial``/``reset`` resample without advancing episode time. ``advance``
    consumes one dt, resamples expired timers, clips requests at the current
    terrain, and updates displacement. Pass the post-physics XY position.
    ``random`` is optional [N,8] independent U[0,1) values; unused draws keep
    shapes static. Compared with upstream this changes RNG consumption, not
    sampling probabilities. The caller checkpoints its generator/global RNG.
    """
    resampling_time = 5.0
    command_area_length = 8.0
    num_steps_per_iter = 1
    limit_vel_prob = .2
    limit_vel_hold_prob = 0.
    limit_ang_vel_at_zero_command_prob = .2
    zero_command_prob = staticmethod(zero_command_prob)

    def __init__(self, dt=.02, episode_seconds=20., terrain_lookup=None,variant="simple",special_episode_seconds=8.):
        if variant not in ("simple","events","moe"):raise ValueError("Unknown variant")
        self.variant=variant;self.grid=None;self.special_episode_seconds=float(special_episode_seconds)
        self.special_contract=special_speed_contract(special_episode_seconds) if variant!="simple" else None
        self.dt, self.episode_seconds = float(dt), float(episode_seconds)
        if not math.isfinite(self.dt) or not math.isfinite(self.episode_seconds) or self.dt <= 0 or self.episode_seconds <= 0:
            raise ValueError('Positive finite control dt and episode duration required.')
        self.max_episode_steps = math.ceil(self.episode_seconds / self.dt)
        self.terrain_lookup = terrain_lookup
        self.terrain_names = () if terrain_lookup is None else tuple(terrain_lookup.names)
        if set(self.terrain_names) - set(TERRAIN_RANGES):
            raise ValueError('Missing official command ranges for terrain names.')

    @staticmethod
    def command(state):
        return state[:, COMMAND]

    @staticmethod
    def timer(state):
        return state[:, TIMER:TIMER + 1]

    def _ranges(self, state, position_xy, stage):
        global_ranges = state.new_tensor(RANGE_STAGES)[stage.long()]
        if self.terrain_lookup is None:
            local=global_ranges
        else:
            ids = self.terrain_lookup.query(position_xy)
            table = state.new_tensor([TERRAIN_RANGES['flat']] + [TERRAIN_RANGES[name] for name in self.terrain_names])
            terrain_ranges = table[ids + 1]
            local = torch.clamp(global_ranges, min=terrain_ranges[..., :1], max=terrain_ranges[..., 1:])
        if self.variant!="simple":
            post=(state[:,TASK_ID]==3)&(state[:,COMPLETED]>.5)
            post_ranges=state.new_tensor(((-2.,2.),(-1.,1.),(-1.5,1.5)))
            completed_ranges=torch.clamp(global_ranges,min=post_ranges[:,:1],max=post_ranges[:,1:])
            local=torch.where(post[:,None,None],completed_ranges,local)
        return global_ranges, local

    def initial(self, position_xy, origins_xy, iteration=0, *, generator=None, random=None,task_id=None,terrain_id=None):
        state = position_xy.new_zeros((len(position_xy), CMD_STATE_DIM))
        return self.reset(state, torch.ones(len(state), device=state.device, dtype=torch.bool),
                          position_xy, origins_xy, iteration, generator=generator, random=random,task_id=task_id,terrain_id=terrain_id)

    def reset(self, state, mask, position_xy, origins_xy, iteration, *, generator=None, random=None,task_id=None,terrain_id=None):
        """Reset selected episodes; nonreset rows retain memory and raw requests.

        ``mask`` is [N] bool. A reset triggers the global iteration stage update
        just as the source manager's _resample does. All rows are re-clipped.
        """
        fresh = torch.cat((torch.zeros_like(state[:, :ORIGIN.start]), origins_xy[:, :2],
                           state[:, RANGE_STAGE:]), -1)
        base = torch.where(mask[:, None], fresh, state)
        task=base[:,TASK_ID] if task_id is None else torch.where(mask,task_id.to(base.dtype),base[:,TASK_ID])
        terrain=base[:,TERRAIN_ID] if terrain_id is None else torch.where(mask,terrain_id.to(base.dtype),base[:,TERRAIN_ID])
        completion=torch.where(mask,torch.zeros_like(base[:,COMPLETED]),base[:,COMPLETED])
        base=torch.cat((base[:,:TASK_ID],task[:,None],terrain[:,None],completion[:,None]),-1)
        return self._resample(base, mask, position_xy, iteration, generator=generator, random=random)

    def advance(self, state, position_xy, iteration, *, generator=None, random=None,completed=None,task_ids=None,terrain_ids=None):
        if completed is not None:
            fresh=(completed>.5)&(state[:,COMPLETED]<.5)
            completion=torch.maximum(state[:,COMPLETED],completed.to(state.dtype))
            timer=torch.where(fresh,torch.zeros_like(state[:,TIMER]),state[:,TIMER])
            state=torch.cat((state[:,:TIMER],timer[:,None],state[:,TIMER+1:COMPLETED],completion[:,None],state[:,COMPLETED+1:]),-1)
        if terrain_ids is not None or self.terrain_lookup is not None:
            current=self.terrain_lookup.query(position_xy) if terrain_ids is None else terrain_ids
            terrain=torch.where(current>=0,current.to(state.dtype),state[:,TERRAIN_ID])
            state=torch.cat((state[:,:TERRAIN_ID],terrain[:,None],state[:,TERRAIN_ID+1:]),-1)
        elapsed = state[:, EPISODE_ELAPSED] + self.dt
        timer = state[:, TIMER] - self.dt
        updated = torch.cat((state[:, :TIMER], timer[:, None], state[:, ACCUMULATION],
                             elapsed[:, None], state[:, LAST_LIMIT:]), -1)
        result = self._resample(updated, timer <= 0., position_xy, iteration, generator=generator, random=random)
        distance = torch.linalg.vector_norm(position_xy[:, :2] - result[:, ORIGIN], dim=-1)
        maximum = torch.maximum(result[:, MAX_DISTANCE], distance)
        return torch.cat((result[:, :MAX_DISTANCE], maximum[:, None], result[:, ORIGIN.start:]), -1)

    def _resample(self, state, due, position_xy, iteration, *, generator=None, random=None):
        if state.ndim != 2 or state.shape[1] != CMD_STATE_DIM:
            raise ValueError('Expected packed command state [N,21].')
        # Lazy global curriculum activation matches the official manager:
        # stages change at the first resampling event at/after the iteration.
        clock = torch.as_tensor(iteration, device=state.device)
        current_stage = (clock[..., None] >= clock.new_tensor(RANGE_ITERATIONS)).sum(-1)
        current_limit_stage = (clock >= 10000).long()
        stage = torch.where(due.any(), current_stage, state[:, RANGE_STAGE]).expand(len(state))
        limit_stage = torch.where(due.any(), current_limit_stage, state[:, LIMIT_STAGE]).expand(len(state))
        global_ranges, ranges = self._ranges(state, position_xy, stage)
        u = torch.rand((len(state), 8), device=state.device, dtype=state.dtype, generator=generator) if random is None else random
        if u.shape != (len(state), 8):
            raise ValueError('Command random draws must have shape [N,8].')
        remaining = (.625 * self.command_area_length
                     - torch.linalg.vector_norm(state[:, ACCUMULATION], dim=-1) * self.resampling_time).clamp_min(0.)
        remaining_time = self.max_episode_steps * self.dt - state[:, EPISODE_ELAPSED] + self.dt
        lower = (remaining / (remaining_time + 1e-9)).clamp_min(0.)
        negative = (-lower[:, None] - ranges[:, :2, 0]).relu()
        positive = (ranges[:, :2, 1] - lower[:, None]).relu()
        draw = u[:, :2] * (negative + positive + 1e-6)
        xy = torch.where(draw < negative, ranges[:, :2, 0] + draw,
                         ranges[:, :2, 1] - positive + draw - negative)
        yaw = ranges[:, 2, 0] + u[:, 2] * (ranges[:, 2, 1] - ranges[:, 2, 0])
        sampled = torch.cat((xy, yaw[:, None]), -1)
        limited = u[:, 3] < self.limit_vel_prob
        early_combos = state.new_tensor(tuple(product((-1, 1), (-1, 1), (-1, 0, 1))))
        late_combos = state.new_tensor(tuple(product((-1, 1), (0,), (-1, 0, 1))))
        code = torch.where(limit_stage[:, None].bool(), late_combos[(u[:, 4] * 6).long().clamp_max(5)],
                           early_combos[(u[:, 4] * 12).long().clamp_max(11)])
        boundary = torch.where(code < 0, global_ranges[..., 0], global_ranges[..., 1])
        boundary = torch.where(code == 0, torch.zeros_like(boundary), boundary)
        # Upstream inverts the newly sampled value when the preceding command
        # was limited. It does not invert the preceding saved boundary command.
        limit_value = torch.where(state[:, LAST_LIMIT:LAST_LIMIT + 1].bool(), -sampled, boundary)
        sampled = torch.where(limited[:, None], limit_value, sampled)
        max_lin = global_ranges[:, :2].abs().flatten(1).amax(-1)
        zero_time = (self.episode_seconds - state[:, EPISODE_ELAPSED]
                     - remaining / (.8 * max_lin + 1e-9)).clamp(0., self.resampling_time)
        is_zero = ((u[:, 3] >= self.limit_vel_prob)
                   & (u[:, 3] < self.limit_vel_prob + zero_command_prob(iteration)) & (zero_time > 0.))
        zero_yaw = torch.where(u[:, 6] < .5, ranges[:, 2, 0], ranges[:, 2, 1])
        zero_yaw = torch.where(u[:, 5] < self.limit_ang_vel_at_zero_command_prob, zero_yaw, torch.zeros_like(zero_yaw))
        zero_command = torch.stack((torch.zeros_like(zero_yaw), torch.zeros_like(zero_yaw), zero_yaw), -1)
        sampled = torch.where(is_zero[:, None], zero_command, sampled)
        request = torch.where(due[:, None], sampled, state[:, REQUEST])
        if self.variant!='simple':
            if self.grid is None:self.grid=VelocityGrid(state.device,self.special_episode_seconds)
            task=state[:,TASK_ID].long();terrain=state[:,TERRAIN_ID].long()
            sampled,bins=self.grid.sample(task,terrain,iteration,generator)
            # Ordinary standing and pure-yaw proportions follow the source sampler.
            ordinary=(task==0);sampled=torch.where((is_zero&ordinary)[:,None],zero_command,sampled)
            special=(task==3)&(state[:,COMPLETED]<.5)
            post=(task==3)&~special
            sampled=torch.where(post[:,None],request,sampled)
            sampled=torch.where(special[:,None],torch.stack((sampled[:,0].clamp(self.special_contract['minimum_speed_mps'],2.),torch.zeros_like(sampled[:,0]),torch.zeros_like(sampled[:,0])),-1),sampled)
            request=torch.where(due[:,None],sampled,state[:,REQUEST])
            gridbin=torch.where(due,self.grid.bin_indices(request),state[:,GRID_BIN])
        else:gridbin=state[:,GRID_BIN]
        actual = coupled_clip(torch.clamp(request, min=ranges[..., 0], max=ranges[..., 1]))
        new_timer = torch.where(is_zero, zero_time, torch.full_like(zero_time, self.resampling_time))
        if self.variant!="simple":
            new_timer=torch.where(state[:,TASK_ID]==2,torch.ones_like(new_timer),new_timer)
            approach=(state[:,TASK_ID]==3)&(state[:,COMPLETED]<.5)
            new_timer=torch.where(approach,torch.full_like(new_timer,self.episode_seconds),new_timer)
        timer = torch.where(due, new_timer, state[:, TIMER])
        accumulation = state[:, ACCUMULATION] + torch.where(due[:, None], actual[:, :2], torch.zeros_like(actual[:, :2]))
        last_limit = torch.where(due, limited.to(state.dtype), state[:, LAST_LIMIT])
        counter = state[:, COMMAND_COUNTER] + due.to(state.dtype)
        return torch.cat((request, actual, timer[:, None], accumulation,
                          state[:, EPISODE_ELAPSED:EPISODE_ELAPSED + 1], last_limit[:, None], counter[:, None],
                          state[:, MAX_DISTANCE:ORIGIN.stop], stage[:, None], limit_stage[:, None],gridbin[:,None],state[:,TASK_ID:]), -1)

    def state_dict(self):return None if self.grid is None else self.grid.state_dict()
    def load_state_dict(self,s):
        if s is not None:
            self.grid=VelocityGrid(s['weights'].device,self.special_episode_seconds);self.grid.load_state_dict(s)
