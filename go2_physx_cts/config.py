"""Three AS2 perception experiments, pinned to blind baseline 2061be6."""
from dataclasses import dataclass, fields

@dataclass
class TrainCfg:
    robot: str = 'as2'
    variant: str = 'simple'
    stage: str = 'terrain'
    num_envs: int = 4096
    buffer_len: int = 1000
    replay_storage: str = 'device'
    collect_steps: int = 1
    iterations: int = 120000
    warmup_iterations: int = 1000
    critic_warmup_iterations: int = 500
    actor_ramp_iterations: int = 1000
    policy_start: int = 1500
    terrain_start_iteration: int = 2500
    all_terrain_iteration: int = 5000
    wm_hidden: tuple = (1024, 1024, 1024)
    wm_window: int = 8
    wm_batch: int = 2048
    wm_visual_windows: int = 512
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
    critic_hidden: tuple = (512, 256, 128)
    actor_experts: int = 3
    proprio_experts: int = 8
    task_count: int = 1
    task_context_dim: int = 62
    cts_updates: int = 2
    cts_batch: int = 160
    cts_sequence_steps: int = 64
    cts_burnin: int = 16
    cts_lr: float = 3e-4
    cts_balance: float = .01
    cts_action_weight: float = .1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_microbatch: int = 16384
    critic_huber_delta: float = 1.
    lr: float = 2e-4
    teacher_lr: float = 2e-4
    wm_lr: float = 2e-4
    lr_adjust_factor: float = 1.3
    lr_fixed_until: int = 5000
    wm_fixed_until: int = 40000
    cts_fixed_until: int = 40000
    wm_lr_bounds: tuple = (5e-5, 2e-4)
    actor_lr_bounds: tuple = (5e-5, 2e-4)
    critic_lr_bounds: tuple = (5e-5, 2e-4)
    cts_lr_bounds: tuple = (1e-4, 3e-4)
    lr_holdout_modulus: int = 32
    grad_clip: float = 1.
    grad_clip_elem: float = .1
    gamma: float = .99
    lam: float = .95
    action_noise: float = .1
    explore_frac: float = .25
    explore_noise: float = .3
    explore_theta: float = 2.
    log_interval: int = 50
    save_interval: int = 1000
    seed: int = 17
    dt: float = .02
    physics_dt: float = .005
    episode_seconds: float = 20.
    payload_max_kg: float = 2.
    reward_calibration: str = ''
    evaluation_interval: int = 5000
    evaluation_python: str = ''
    depth_package: str = 'observation/depth_input'
    depth_chunk_size: int = 1024
    encoding_chunk_size: int = 2048
    torch_threads: int = 8
    depth_period: int = 1
    imagination_depth_period: int = 5
    depth_cache_gib: float = 26.
    latent_actor_stop_gradient: bool = False
    base_height_target: float = .36
    mixed_precision: bool = True
    depth_precision: str = 'fp32'
    compile: bool = True
    device: str = 'cuda:0'
    gpu_mode: str = 'single'
    noise_devices: tuple = ()

    def wm_updates_for(self, iteration):
        return 2 if iteration < self.terrain_start_iteration else 4

    @classmethod
    def from_record(cls, record):
        unknown = set(record) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError('Retired/unknown configuration fields: ' + ', '.join(sorted(unknown)))
        return cls(**record)

    def validate(self, production=True):
        if self.gpu_mode not in ('single','auto'):
            raise ValueError('gpu_mode must be single or auto')
        if self.robot != 'as2' or self.variant not in ('simple', 'events', 'moe'):
            raise ValueError('Only the three AS2 experiments are supported')
        if self.replay_storage not in ('device','cpu'):
            raise ValueError('Replay storage must be device or cpu')
        expected = (1, 3) if self.variant == 'simple' else (4, 3 if self.variant == 'moe' else 1)
        if (self.task_count, self.actor_experts) != expected:
            raise ValueError('Variant/task/actor mismatch')
        if (self.collect_steps, self.horizon, self.wm_window) != (1, 32, 8):
            raise ValueError('Source collect/H/K contract is 1/32/8')
        if not 0 < self.wm_visual_windows <= self.wm_batch:
            raise ValueError('Visual supervision must sample valid whole WM windows')
        if self.depth_precision not in ('fp32','bf16') or min(self.encoding_chunk_size,self.torch_threads)<=0:
            raise ValueError('Invalid noise precision or execution batching')
        if production and (self.num_envs, self.buffer_len, self.imag_batch, self.wm_batch) != (4096, 1000, 8192, 2048):
            raise ValueError('Production scale must be 4096/1000/8192/2048')
        if production and self.iterations != 120000:
            raise ValueError('Production budget is 120000 iterations')
        if production and (self.cts_updates,self.cts_batch,self.cts_sequence_steps,self.cts_burnin) != (2,160,64,16):
            raise ValueError('Visual CTS adaptation is 2 x 160 x (16 burn + 64 loss)')
        if (self.critic_epochs,self.critic_minibatches) != (5,4):
            raise ValueError('Keep the AS2 five passes / four minibatches')
        if not 0 < self.actor_microbatch <= self.imag_batch or self.teacher_fraction != .75:
            raise ValueError('Invalid equivalent imagination partition')
        if self.payload_max_kg != 2. or self.depth_period != 1 or self.base_height_target != .36:
            raise ValueError('AS2 contract requires 0..2kg payload, 50Hz depth and 0.36m height')
        if self.imagination_depth_period != 5:
            raise ValueError('Imagined policy visual input refreshes every 5 control steps (10Hz)')
        if self.task_context_dim != 62 or self.depth_cache_gib <= 0:
            raise ValueError('Invalid completion memory or depth cache capacity')
