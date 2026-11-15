"""Robot-specific contracts. Pinned AS2 PhysX contract from blind baseline2061be6."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RobotProfile:
    name: str
    initial_height: float
    left_hip: float
    reward_height: float
    calf_length: float | None
    velocity_limits: tuple

PROFILES = {
    # User-approved 0.36m stance target; asset dynamics remain pinned to AS2.
    # Fixed foot bodies are mandatory: no Go2 geometry fallback is permitted.
    'as2': RobotProfile('as2', .4, -.1, .36, None, (24., 24., 16.)),
}

def profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f'Unknown robot: {name}') from None

def as2_motor_values(device):
    import torch
    def t(values): return torch.tensor(values * 4, device=device)
    # Unitree unitree_rl_mjlab As2 constants; fixed-torque PD, not Go2 DC curve.
    inertia, kp, kd, effort = t([.026,.026,.038]), t([40.,40.,60.]), t([2.,2.,3.]), t([50.,50.,75.])
    # Project action normalization, explicitly not a manufacturer calibration.
    return inertia, kp, kd, effort, .25 * effort / kp
