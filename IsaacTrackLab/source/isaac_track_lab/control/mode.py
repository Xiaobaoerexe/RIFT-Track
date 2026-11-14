"""Explicit control-mode boundary between deployable actor data and simulator teachers."""

from __future__ import annotations

from enum import StrEnum
from math import atan2, cos, sin

from .velocity import VelocityCommand


class TrackingControlMode(StrEnum):
    """Control modes supported by the 305 m regression environment."""

    PURE_E2E = "pure_e2e"
    SHOWCASE_HYBRID = "showcase_hybrid"

    @property
    def allows_privileged_recovery(self) -> bool:
        return self is TrackingControlMode.SHOWCASE_HYBRID


def compose_tracking_command(
    *,
    mode: TrackingControlMode,
    visual_command: VelocityCommand,
    robot_heading_rad: float,
    scheduled_target_bearing_rad: float | None,
    target_visible: bool,
    route_yaw_gain: float = 1.8,
    maximum_yaw_rps: float = 0.9,
    visual_weight: float = 0.35,
) -> VelocityCommand:
    """Compose a command without letting privileged data enter ``pure_e2e``.

    ``scheduled_target_bearing_rad`` is simulator/route privileged state.  It
    is required by the showcase mode and rejected by construction in the pure
    actor path, which returns the visual command unchanged.
    """

    if mode is TrackingControlMode.PURE_E2E:
        return visual_command
    if scheduled_target_bearing_rad is None:
        raise ValueError("showcase_hybrid requires scheduled target bearing")
    heading_error = atan2(
        sin(scheduled_target_bearing_rad - robot_heading_rad),
        cos(scheduled_target_bearing_rad - robot_heading_rad),
    )
    route_yaw = max(
        -maximum_yaw_rps,
        min(maximum_yaw_rps, route_yaw_gain * heading_error),
    )
    image_weight = visual_weight if target_visible else 0.0
    yaw = (
        image_weight * visual_command.yaw_rate_rps
        + (1.0 - image_weight) * route_yaw
    )
    return VelocityCommand(
        vx_mps=visual_command.vx_mps,
        vy_mps=visual_command.vy_mps,
        yaw_rate_rps=max(-maximum_yaw_rps, min(maximum_yaw_rps, yaw)),
    )
