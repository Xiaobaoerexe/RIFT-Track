"""Functional port of Go2RLLab's pinned mixed-task command sampler.

Source: wty-yy/go2_rl_lab 6c95d9feef4814c25eb6536e61bb38e1d1d718d4,
cmd_utils/rl_gym_cmd.py and tasks/mixed/env_cfg.py:CommandsCfg.
The caller supplies learner iterations (num_steps_per_iter=1), not physics
substeps. This module contains no capability gates or grid curriculum.

All command memory is a packed floating tensor suitable for replay. Random
choices may be supplied explicitly; no state tensor is mutated or detached.
"""
from itertools import product
import math
import torch

SOURCE_COMMIT = '6c95d9feef4814c25eb6536e61bb38e1d1d718d4'
CMD_STATE_DIM = 17
REQUEST, COMMAND = slice(0, 3), slice(3, 6)
TIMER, ACCUMULATION, EPISODE_ELAPSED = 6, slice(7, 9), 9
LAST_LIMIT, COMMAND_COUNTER, MAX_DISTANCE = 10, 11, 12
ORIGIN, RANGE_STAGE, LIMIT_STAGE = slice(13, 15), 15, 16
COMMAND_NAMES = ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')
RANGE_ITERATIONS = (5000, 15000, 25000, 35000, 45000)
RANGE_STAGES = (
    ((-.5, .5), (-.5, .5), (-1., 1.)),
    ((-1., 1.), (-1., 1.), (-2., 2.)),
    ((-2., 2.), (-1., 1.), (-3., 3.)),
    ((-3., 3.), (-1., 1.), (-3., 3.)),
    ((-4., 4.), (-1., 1.), (-3., 3.)),
    ((-4.5, 4.2), (-1., 1.), (-3., 3.)),
)
TERRAIN_RANGES = {
    'flat': ((-4.5, 4.2), (-1., 1.), (-3., 3.)),
    'wave': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'slope_up': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'slope_down': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'rough_slope': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'stairs_up': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'stairs_down': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'obstacles': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'wheels': ((-2., 1.5), (-1., 1.), (-1.5, 1.5)),
    'small_squares': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
}


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

    def __init__(self, dt=.02, episode_seconds=25., terrain_lookup=None):
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
            return global_ranges, global_ranges
        ids = self.terrain_lookup.query(position_xy)
        table = state.new_tensor([TERRAIN_RANGES['flat']] + [TERRAIN_RANGES[name] for name in self.terrain_names])
        terrain_ranges = table[ids + 1]
        local = torch.clamp(global_ranges, min=terrain_ranges[..., :1], max=terrain_ranges[..., 1:])
        return global_ranges, local

    def initial(self, position_xy, origins_xy, iteration=0, *, generator=None, random=None):
        state = position_xy.new_zeros((len(position_xy), CMD_STATE_DIM))
        return self.reset(state, torch.ones(len(state), device=state.device, dtype=torch.bool),
                          position_xy, origins_xy, iteration, generator=generator, random=random)

    def reset(self, state, mask, position_xy, origins_xy, iteration, *, generator=None, random=None):
        """Reset selected episodes; nonreset rows retain memory and raw requests.

        ``mask`` is [N] bool. A reset triggers the global iteration stage update
        just as the source manager's _resample does. All rows are re-clipped.
        """
        fresh = torch.cat((torch.zeros_like(state[:, :ORIGIN.start]), origins_xy[:, :2],
                           state[:, RANGE_STAGE:]), -1)
        base = torch.where(mask[:, None], fresh, state)
        return self._resample(base, mask, position_xy, iteration, generator=generator, random=random)

    def advance(self, state, position_xy, iteration, *, generator=None, random=None):
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
            raise ValueError('Expected packed command state [N,17].')
        # Lazy global curriculum activation matches the official manager:
        # stages change at the first resampling event at/after the iteration.
        clock = torch.as_tensor(iteration, device=state.device)
        current_stage = (clock[..., None] >= clock.new_tensor(RANGE_ITERATIONS)).sum(-1)
        current_limit_stage = (clock >= 25000).long()
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
        actual = coupled_clip(torch.clamp(request, min=ranges[..., 0], max=ranges[..., 1]))
        new_timer = torch.where(is_zero, zero_time, torch.full_like(zero_time, self.resampling_time))
        timer = torch.where(due, new_timer, state[:, TIMER])
        accumulation = state[:, ACCUMULATION] + torch.where(due[:, None], actual[:, :2], torch.zeros_like(actual[:, :2]))
        last_limit = torch.where(due, limited.to(state.dtype), state[:, LAST_LIMIT])
        counter = state[:, COMMAND_COUNTER] + due.to(state.dtype)
        return torch.cat((request, actual, timer[:, None], accumulation,
                          state[:, EPISODE_ELAPSED:EPISODE_ELAPSED + 1], last_limit[:, None], counter[:, None],
                          state[:, MAX_DISTANCE:ORIGIN.stop], stage[:, None], limit_stage[:, None]), -1)
