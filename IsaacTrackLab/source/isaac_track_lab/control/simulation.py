"""Planar high-level command surrogate for visual-task simulation.

This deliberately models the official SportClient command boundary, not As2
joint-level locomotion or sim-to-real dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from .velocity import VelocityCommand


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float


class PlanarVelocitySurrogate:
    def __init__(
        self,
        initial_pose: Pose2D,
        *,
        maximum_forward_mps: float = 0.8,
        maximum_lateral_mps: float = 0.5,
        maximum_yaw_rps: float = 1.0,
        linear_acceleration_mps2: float = 1.0,
        yaw_acceleration_rps2: float = 2.0,
    ) -> None:
        self.pose = initial_pose
        self.command = VelocityCommand(0.0, 0.0, 0.0)
        self.velocity = VelocityCommand(0.0, 0.0, 0.0)
        self.maximum_forward_mps = maximum_forward_mps
        self.maximum_lateral_mps = maximum_lateral_mps
        self.maximum_yaw_rps = maximum_yaw_rps
        self.linear_acceleration_mps2 = linear_acceleration_mps2
        self.yaw_acceleration_rps2 = yaw_acceleration_rps2

    def reset(self) -> None:
        self.command = VelocityCommand(0.0, 0.0, 0.0)
        self.velocity = VelocityCommand(0.0, 0.0, 0.0)

    def apply(self, command: VelocityCommand) -> None:
        self.command = VelocityCommand(
            max(-self.maximum_forward_mps, min(self.maximum_forward_mps, command.vx_mps)),
            max(-self.maximum_lateral_mps, min(self.maximum_lateral_mps, command.vy_mps)),
            max(-self.maximum_yaw_rps, min(self.maximum_yaw_rps, command.yaw_rate_rps)),
        )

    def step(self, dt_s: float) -> Pose2D:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.velocity = VelocityCommand(
            self._approach(
                self.velocity.vx_mps,
                self.command.vx_mps,
                self.linear_acceleration_mps2 * dt_s,
            ),
            self._approach(
                self.velocity.vy_mps,
                self.command.vy_mps,
                self.linear_acceleration_mps2 * dt_s,
            ),
            self._approach(
                self.velocity.yaw_rate_rps,
                self.command.yaw_rate_rps,
                self.yaw_acceleration_rps2 * dt_s,
            ),
        )
        yaw = self.pose.yaw_rad
        world_vx = cos(yaw) * self.velocity.vx_mps - sin(yaw) * self.velocity.vy_mps
        world_vy = sin(yaw) * self.velocity.vx_mps + cos(yaw) * self.velocity.vy_mps
        self.pose = Pose2D(
            self.pose.x_m + world_vx * dt_s,
            self.pose.y_m + world_vy * dt_s,
            self.pose.yaw_rad + self.velocity.yaw_rate_rps * dt_s,
        )
        return self.pose

    @staticmethod
    def _approach(current: float, target: float, maximum_delta: float) -> float:
        delta = max(-maximum_delta, min(maximum_delta, target - current))
        return current + delta
