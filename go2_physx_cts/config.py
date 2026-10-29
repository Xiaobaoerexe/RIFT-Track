"""Explicit source-aligned defaults; task and CTS adaptations are named."""
from dataclasses import dataclass

@dataclass
class TrainCfg:
    stage: str = "flat"
    num_envs: int = 4096
    buffer_len: int = 1000
    collect_steps: int = 1
    iterations: int = 40000
    warmup_iterations: int = 300
    wm_hidden: tuple = (1024, 1024, 1024)
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 2
    wm_prioritized_frac: float = .25
    wm_archive_frac: float = .25
    archive_size: int = 100000
    archive_keep_prob: float = .02
    imag_batch: int = 8192
    teacher_fraction: float = .75
    horizon: int = 32
    actor_microbatch: int = 2048
    actor_hidden: tuple = (512, 256, 128)
    critic_hidden: tuple = (512, 256, 128)
    cts_updates: int = 20
    cts_batch: int = 1024
    cts_lr: float = .001
    cts_balance: float = .01
    critic_epochs: int = 5
    critic_minibatches: int = 4
    lr: float = 2e-4
    actor_lr_scale: float = 1.
    lr_final_frac: float = .03
    lr_decay_iterations: int = 30000
    actor_ramp_iterations: int = 1000
    critic_warmup_iterations: int = 500
    grad_clip: float = 1.
    grad_clip_elem: float = .1
    gamma: float = .99
    lam: float = .95
    action_noise: float = .1
    explore_frac: float = .25
    explore_noise: float = .3
    explore_theta: float = 2.
    driver_policy: str = ""
    driver_frac: float = .1
    expert_policy: str = ""
    expert_frac: float = 0.
    allow_missing_flat_expert: bool = False
    collapse_falls: float = 3.
    log_interval: int = 50
    save_interval: int = 1000
    seed: int = 17
    dt: float = .02
    physics_dt: float = .005
    episode_seconds: float = 20.
    terrain_curriculum_iterations: int = 12000
    command_max: tuple = (.5, .3, .8)
    payload_max_kg: float = 0.
    evaluation_interval: int = 5000
    evaluation_xml: str = ''  # Legacy checkpoint field, ignored by official evaluation.
    evaluation_python: str = ''
    evaluation_profile: str = 'checkpoint_control'
    adaptive_iterations: bool = True
    compile: bool = True
    device: str = "cuda:0"

    @classmethod
    def terrain(cls, **kwargs):
        return cls(stage="terrain", wm_updates=4, actor_lr_scale=.3,
                   warmup_iterations=1000, expert_frac=.1, **kwargs)

    def validate(self, production=True):
        if self.stage not in ("flat", "terrain"):
            raise ValueError("Unknown stage")
        if self.collect_steps != 1 or self.horizon != 32 or self.wm_window != 8:
            raise ValueError("Source-aligned collect/H/K contract changed")
        if production and (self.num_envs, self.buffer_len, self.imag_batch, self.wm_batch) != (4096,1000,8192,2048):
            raise ValueError("Production scale must be4096/1000/8192/2048")
        if self.wm_updates != (2 if self.stage == "flat" else 4):
            raise ValueError("WM updates must match stage recipe")
        if not 0 < self.teacher_fraction < 1:
            raise ValueError("Both CTS branches required")
        if not 0 <= self.driver_frac + self.expert_frac < 1:
            raise ValueError("Invalid data source fractions")
        if self.driver_frac and not self.driver_policy:
            raise ValueError("Continuous mixed-terrain expert missing; no silent fallback")
        if self.expert_frac and not self.expert_policy:
            raise ValueError("Flat expert missing; set fraction0 and explicitly acknowledge degraded recipe")
        if self.stage == "terrain" and not self.expert_frac and not self.allow_missing_flat_expert:
            raise ValueError("Terrain recipe lacks its10% flat skill expert")
        if self.payload_max_kg < 0 or self.payload_max_kg > 1:
            raise ValueError("Only fixed nonnegative0..1kg payload is supported")
