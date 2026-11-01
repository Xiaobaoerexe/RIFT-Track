"""Approved depth multitask training contract; one active implementation."""
from dataclasses import dataclass, fields

@dataclass
class TrainCfg:
    stage: str = 'multitask'
    num_envs: int = 4096
    buffer_len: int = 1000
    collect_steps: int = 1
    iterations: int = 120000
    warmup_iterations: int = 1000
    critic_warmup_iterations: int = 500
    actor_ramp_iterations: int = 1000
    wm_hidden: tuple = (1024, 1024, 1024)
    wm_window: int = 8
    wm_batch: int = 2048
    wm_prioritized_frac: float = .25
    wm_archive_frac: float = .25
    archive_size: int = 100000
    archive_keep_prob: float = .02
    event_history_steps: int = 100
    imag_batch: int = 8192
    teacher_fraction: float = .75
    horizon: int = 32
    actor_microbatch: int = 8192
    actor_hidden: tuple = (512, 256, 128)
    critic_hidden: tuple = (256, 128)
    actor_experts: int = 6
    proprio_experts: int = 8
    task_count: int = 10
    task_context_dim: int = 21
    cts_updates: int = 2
    cts_batch: int = 128
    cts_sequence_steps: int = 64
    cts_burnin: int = 16
    cts_lr: float = 3e-4
    critic_epochs: int = 1
    critic_minibatches: int = 5
    critic_microbatch: int = 4096
    lr: float = 2e-4
    wm_lr: float = 2e-4
    teacher_lr: float = 1e-4
    lr_trial_factor: float = 1.3
    lr_trial_interval: int = 1000
    lr_fixed_until: int = 5000
    wm_lr_bounds: tuple = (2e-5, 5e-4)
    actor_lr_bounds: tuple = (4e-6, 2e-3)
    critic_lr_bounds: tuple = (4e-6, 2e-3)
    cts_lr_bounds: tuple = (1e-5, 3e-4)
    grad_clip: float = 1.
    grad_clip_elem: float = .1
    gamma: float = .99
    lam: float = .95
    action_noise: float = .1
    explore_frac: float = .25
    explore_noise: float = .3
    explore_theta: float = 2.
    driver_policy: str = ''
    driver_frac: float = .1
    provider_fade_iterations: int = 5000
    collapse_falls: float = 3.
    log_interval: int = 50
    save_interval: int = 1000
    seed: int = 17
    dt: float = .02
    physics_dt: float = .005
    episode_seconds: float = 20.
    payload_max_kg: float = 1.2
    reward_calibration: str = ''
    evaluation_interval: int = 5000
    evaluation_python: str = ''
    depth_package: str = '/home/abc/Documents/go2_supermimic_runtime/depth_input'
    depth_chunk_size: int = 32
    depth_period: int = 10
    mixed_precision: bool = True
    # Original frozen depth BF16 failed the independent pose-VJP acceptance.
    # Neural policy/dynamics computations remain BF16; a direct-depth candidate
    # can select BF16 only after its own precision-specific acceptance passes.
    depth_precision: str = 'fp32'
    fast_depth_checkpoint: str = ''
    compile: bool = True
    device: str = 'cuda:0'

    def wm_updates_for(self, iteration):
        return 2 if iteration < 2500 else 4

    @classmethod
    def from_record(cls, record):
        unknown = set(record) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError('Obsolete or unknown configuration fields: ' + ', '.join(sorted(unknown)))
        return cls(**record)

    def validate(self, production=True):
        if self.stage != 'multitask':
            raise ValueError('Only the approved depth multitask architecture is supported')
        if (self.collect_steps, self.horizon, self.wm_window) != (1, 32, 8):
            raise ValueError('Collect/H/K contract must remain 1/32/8')
        if production and (self.num_envs, self.buffer_len, self.imag_batch, self.wm_batch) != (4096,1000,8192,2048):
            raise ValueError('Formal scale must be 4096/1000/8192/2048')
        if production and self.iterations != 120000:
            raise ValueError('Approved formal budget is 120000 iterations')
        if production and (self.cts_updates,self.cts_batch,self.cts_sequence_steps,self.cts_burnin) != (2,128,64,16):
            raise ValueError('Student sequence contract must be 2 x 128 x (16 burn-in + 64 loss)')
        if self.actor_experts != 6 or self.proprio_experts != 8 or self.task_count != 10:
            raise ValueError('Network expert/task counts changed')
        if self.actor_microbatch <= 0 or self.actor_microbatch > self.imag_batch:
            raise ValueError('Invalid actor microbatch; total cohort remains8192')
        if production and self.actor_microbatch != self.imag_batch:
            raise ValueError('V2 production uses one fused8192 imagination cohort')
        if self.teacher_fraction != .75:
            raise ValueError('Imagination split must remain6144teacher/2048student, equal branch means')
        if not 0 <= self.driver_frac < 1 or (self.driver_frac and not self.driver_policy):
            raise ValueError('A declared official collection provider must exist')
        if not 0 <= self.payload_max_kg <= 1.2:
            raise ValueError('Payload must remain0..1.2kg and fixed within each episode')
        if (self.critic_epochs,self.critic_minibatches) != (1,5):
            raise ValueError('Critic uses exactly one full pass divided into five optimizer steps')
        if self.depth_period != 10:
            raise ValueError('Approved visual cadence is5Hz at50Hz control')
        if self.depth_precision not in ('fp32','bf16'):
            raise ValueError('Unsupported depth precision')
        if production and not self.reward_calibration:
            raise ValueError('Formal launch requires measured reward budgets')
        if production and not self.evaluation_python:
            raise ValueError('Formal launch requires the pinned official evaluation runtime')
