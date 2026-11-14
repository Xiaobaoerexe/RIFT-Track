"""Unitree As2 articulation configuration.

The actuator gains are conservative smoke-test values. They are not the final
identified As2 actuator model and must be replaced after dynamics calibration.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AS2_USD_PATH = PROJECT_ROOT / "assets/robots/as2/as2/as2.usda"

AS2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(AS2_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.48),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            joint_effort_limit={".*_(hip|thigh)_joint": 60.0, ".*_calf_joint": 90.0},
            joint_velocity_limit=24.0,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
        )
    },
)
