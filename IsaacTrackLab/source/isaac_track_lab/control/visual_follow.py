"""RGB-bounding-box to high-level body velocity baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .velocity import VelocityCommand


@dataclass(frozen=True)
class VisualFollowConfig:
    desired_area_ratio: float
    area_deadband_ratio: float = 0.002
    forward_gain: float = 8.0
    forward_feedforward_mps: float = 0.0
    yaw_gain: float = 2.0
    maximum_forward_mps: float = 0.5
    maximum_reverse_mps: float = 0.15
    maximum_yaw_rps: float = 0.8
    search_forward_mps: float = 0.12
    search_yaw_rps: float = 0.35


class BBoxVelocityPolicy:
    def __init__(self, config: VisualFollowConfig) -> None:
        if config.desired_area_ratio <= 0.0:
            raise ValueError("desired_area_ratio must be positive")
        self.config = config
        self.last_yaw_direction = 1.0

    def command(
        self,
        bbox_xyxy: Sequence[float] | None,
        image_width: int,
        image_height: int,
    ) -> VelocityCommand:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if bbox_xyxy is None:
            return VelocityCommand(
                self.config.search_forward_mps,
                0.0,
                self.last_yaw_direction * self.config.search_yaw_rps,
            )
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be non-empty")
        center_x = (x1 + x2) * 0.5 / image_width
        area_ratio = (x2 - x1) * (y2 - y1) / (image_width * image_height)
        area_error = self.config.desired_area_ratio - area_ratio
        if abs(area_error) <= self.config.area_deadband_ratio:
            area_error = 0.0
        forward = max(
            -self.config.maximum_reverse_mps,
            min(
                self.config.maximum_forward_mps,
                self.config.forward_feedforward_mps
                + self.config.forward_gain * area_error,
            ),
        )
        yaw = max(
            -self.config.maximum_yaw_rps,
            min(self.config.maximum_yaw_rps, self.config.yaw_gain * (0.5 - center_x)),
        )
        if abs(yaw) > 1e-6:
            self.last_yaw_direction = 1.0 if yaw > 0.0 else -1.0
        return VelocityCommand(forward, 0.0, yaw)
