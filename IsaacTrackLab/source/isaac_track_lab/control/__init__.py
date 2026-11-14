"""High-level command contracts; no locomotion implementation is implied."""

from .velocity import VelocityCommand, VelocityController
from .simulation import PlanarVelocitySurrogate, Pose2D
from .go2_locomotion import Go2LocomotionPolicy, Go2PolicyOutput, POLICY_JOINT_NAMES
from .visual_follow import BBoxVelocityPolicy, VisualFollowConfig
from .mode import TrackingControlMode, compose_tracking_command

__all__ = [
    "BBoxVelocityPolicy",
    "PlanarVelocitySurrogate",
    "Pose2D",
    "Go2LocomotionPolicy",
    "Go2PolicyOutput",
    "POLICY_JOINT_NAMES",
    "VelocityCommand",
    "VelocityController",
    "VisualFollowConfig",
    "TrackingControlMode",
    "compose_tracking_command",
]
