"""Terrain-following target motion for static outdoor training scenes."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, pi
from typing import Any, Sequence

from .terrain_routes import terrain_mesh_height


@dataclass(frozen=True)
class TargetMotionPose:
    position_m: tuple[float, float, float]
    yaw_deg: float
    speed_mps: float
    route_distance_m: float
    completed: bool


class TerrainTargetFollower:
    """Stateful target root motion with terrain grounding and turn anticipation.

    This class controls only the character root.  It deliberately does not
    claim to provide skeletal foot IK; that requires a rig-specific retargeting
    contract and must be validated separately for each character asset.
    """

    def __init__(
        self,
        map_config: dict[str, Any],
        waypoints: Sequence[Sequence[float]],
        *,
        nominal_speed_mps: float = 1.25,
        minimum_speed_mps: float = 1.0,
        maximum_speed_mps: float = 1.5,
        heading_lookahead_m: float = 2.5,
        maximum_yaw_rate_deg_s: float = 120.0,
        ground_clearance_m: float = 0.0,
        terrain_grid_spacing_m: float = 1.0,
        terrain_buffer_m: float = 30.0,
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("at least two waypoints are required")
        if not 0.0 < minimum_speed_mps <= nominal_speed_mps <= maximum_speed_mps:
            raise ValueError("target speed contract is invalid")
        if heading_lookahead_m <= 0.0 or maximum_yaw_rate_deg_s <= 0.0:
            raise ValueError("lookahead and yaw-rate limits must be positive")
        self.map_config = map_config
        self.waypoints = tuple((float(p[0]), float(p[1])) for p in waypoints)
        self.nominal_speed_mps = float(nominal_speed_mps)
        self.minimum_speed_mps = float(minimum_speed_mps)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.heading_lookahead_m = float(heading_lookahead_m)
        self.maximum_yaw_rate_deg_s = float(maximum_yaw_rate_deg_s)
        self.ground_clearance_m = float(ground_clearance_m)
        self.terrain_grid_spacing_m = float(terrain_grid_spacing_m)
        self.terrain_buffer_m = float(terrain_buffer_m)
        self._segment_lengths = tuple(
            hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(self.waypoints, self.waypoints[1:])
        )
        if any(length <= 1e-9 for length in self._segment_lengths):
            raise ValueError("adjacent route points must be distinct")
        cumulative = [0.0]
        for length in self._segment_lengths:
            cumulative.append(cumulative[-1] + length)
        self._cumulative = tuple(cumulative)
        self.total_length_m = self._cumulative[-1]
        self.route_distance_m = 0.0
        self.yaw_deg = self._desired_yaw_deg(0.0)

    def reset(self, route_distance_m: float = 0.0) -> TargetMotionPose:
        if not 0.0 <= route_distance_m <= self.total_length_m:
            raise ValueError("reset distance lies outside the route")
        self.route_distance_m = float(route_distance_m)
        self.yaw_deg = self._desired_yaw_deg(self.route_distance_m)
        return self._pose(self.nominal_speed_mps if route_distance_m < self.total_length_m else 0.0)

    def step(self, dt_s: float) -> TargetMotionPose:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.route_distance_m >= self.total_length_m:
            return self._pose(0.0)
        speed = self._speed_at(self.route_distance_m)
        self.route_distance_m = min(
            self.total_length_m, self.route_distance_m + speed * dt_s
        )
        desired = self._desired_yaw_deg(self.route_distance_m)
        delta = (desired - self.yaw_deg + 180.0) % 360.0 - 180.0
        maximum_delta = self.maximum_yaw_rate_deg_s * dt_s
        self.yaw_deg += max(-maximum_delta, min(maximum_delta, delta))
        completed = self.route_distance_m >= self.total_length_m
        return self._pose(0.0 if completed else speed)

    def hold(self) -> TargetMotionPose:
        """Return the current grounded pose without advancing route progress."""

        return self._pose(0.0)

    def _pose(self, speed_mps: float) -> TargetMotionPose:
        x_m, y_m = self._xy_at(self.route_distance_m)
        z_m = terrain_mesh_height(
            self.map_config, x_m, y_m,
            grid_spacing_m=self.terrain_grid_spacing_m,
            terrain_buffer_m=self.terrain_buffer_m,
        ) + self.ground_clearance_m
        return TargetMotionPose(
            position_m=(x_m, y_m, z_m),
            yaw_deg=self.yaw_deg,
            speed_mps=float(speed_mps),
            route_distance_m=self.route_distance_m,
            completed=self.route_distance_m >= self.total_length_m,
        )

    def _xy_at(self, distance_m: float) -> tuple[float, float]:
        distance_m = max(0.0, min(self.total_length_m, float(distance_m)))
        for index, length in enumerate(self._segment_lengths):
            if distance_m <= self._cumulative[index + 1] or index == len(self._segment_lengths) - 1:
                ratio = (distance_m - self._cumulative[index]) / length
                left, right = self.waypoints[index], self.waypoints[index + 1]
                return (
                    left[0] + ratio * (right[0] - left[0]),
                    left[1] + ratio * (right[1] - left[1]),
                )
        return self.waypoints[-1]

    def _desired_yaw_deg(self, distance_m: float) -> float:
        before = self._xy_at(max(0.0, distance_m - self.heading_lookahead_m))
        after = self._xy_at(min(self.total_length_m, distance_m + self.heading_lookahead_m))
        return degrees(atan2(after[1] - before[1], after[0] - before[0]))

    def _speed_at(self, distance_m: float) -> float:
        before = self._xy_at(max(0.0, distance_m - self.heading_lookahead_m))
        center = self._xy_at(distance_m)
        after = self._xy_at(min(self.total_length_m, distance_m + self.heading_lookahead_m))
        incoming = atan2(center[1] - before[1], center[0] - before[0])
        outgoing = atan2(after[1] - center[1], after[0] - center[0])
        turn = abs((outgoing - incoming + pi) % (2.0 * pi) - pi)
        turn_fraction = min(1.0, turn / (0.5 * pi))
        speed = self.nominal_speed_mps - (
            self.nominal_speed_mps - self.minimum_speed_mps
        ) * turn_fraction
        return max(self.minimum_speed_mps, min(self.maximum_speed_mps, speed))
