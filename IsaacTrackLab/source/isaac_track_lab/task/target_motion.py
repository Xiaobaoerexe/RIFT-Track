"""Terrain-following target motion for static outdoor training scenes."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, sin
from typing import Any, Sequence

from .terrain_routes import terrain_mesh_height, traversable_feature_surface_height


@dataclass(frozen=True)
class TargetMotionPose:
    position_m: tuple[float, float, float]
    yaw_deg: float
    speed_mps: float
    route_distance_m: float
    completed: bool
    locomotion_mode: str
    traversable_feature_id: str | None
    walkable_surface_z_m: float
    surface_clearance_m: float
    vertical_speed_mps: float


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
        traversal_speed_mps: float = 0.65,
        step_anticipation_m: float = 0.45,
        maximum_step_up_speed_mps: float = 0.75,
        maximum_step_down_speed_mps: float = 0.55,
        terrain_grid_spacing_m: float = 1.0,
        terrain_buffer_m: float = 30.0,
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("at least two waypoints are required")
        if not 0.0 < minimum_speed_mps <= nominal_speed_mps <= maximum_speed_mps:
            raise ValueError("target speed contract is invalid")
        if heading_lookahead_m <= 0.0 or maximum_yaw_rate_deg_s <= 0.0:
            raise ValueError("lookahead and yaw-rate limits must be positive")
        if not 0.0 < traversal_speed_mps <= maximum_speed_mps:
            raise ValueError("traversal speed must be positive and bounded")
        if step_anticipation_m <= 0.0:
            raise ValueError("step anticipation must be positive")
        if maximum_step_up_speed_mps <= 0.0 or maximum_step_down_speed_mps <= 0.0:
            raise ValueError("vertical speed limits must be positive")
        self.map_config = map_config
        self.waypoints = tuple((float(p[0]), float(p[1])) for p in waypoints)
        self.nominal_speed_mps = float(nominal_speed_mps)
        self.minimum_speed_mps = float(minimum_speed_mps)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.heading_lookahead_m = float(heading_lookahead_m)
        self.maximum_yaw_rate_deg_s = float(maximum_yaw_rate_deg_s)
        self.ground_clearance_m = float(ground_clearance_m)
        self.traversal_speed_mps = float(traversal_speed_mps)
        self.step_anticipation_m = float(step_anticipation_m)
        self.maximum_step_up_speed_mps = float(maximum_step_up_speed_mps)
        self.maximum_step_down_speed_mps = float(maximum_step_down_speed_mps)
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
        self._root_z_m = 0.0
        self._vertical_speed_mps = 0.0

    def reset(self, route_distance_m: float = 0.0) -> TargetMotionPose:
        if not 0.0 <= route_distance_m <= self.total_length_m:
            raise ValueError("reset distance lies outside the route")
        self.route_distance_m = float(route_distance_m)
        self.yaw_deg = self._desired_yaw_deg(self.route_distance_m)
        surface, _, ahead_surface, ahead_feature = self._surface_context(
            self.route_distance_m
        )
        self._root_z_m = max(
            surface,
            ahead_surface if ahead_feature is not None else surface,
        ) + self.ground_clearance_m
        self._vertical_speed_mps = 0.0
        return self._pose(self.nominal_speed_mps if route_distance_m < self.total_length_m else 0.0)

    def step(self, dt_s: float) -> TargetMotionPose:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.route_distance_m >= self.total_length_m:
            return self._pose(0.0)
        speed = self._speed_at(self.route_distance_m)
        previous_z_m = self._root_z_m
        self.route_distance_m = min(
            self.total_length_m, self.route_distance_m + speed * dt_s
        )
        desired = self._desired_yaw_deg(self.route_distance_m)
        delta = (desired - self.yaw_deg + 180.0) % 360.0 - 180.0
        maximum_delta = self.maximum_yaw_rate_deg_s * dt_s
        self.yaw_deg += max(-maximum_delta, min(maximum_delta, delta))
        surface, _, ahead_surface, ahead_feature = self._surface_context(
            self.route_distance_m
        )
        desired_z_m = surface + self.ground_clearance_m
        if ahead_feature is not None:
            desired_z_m = max(
                desired_z_m, ahead_surface + self.ground_clearance_m
            )
        if desired_z_m >= previous_z_m:
            next_z_m = min(
                desired_z_m,
                previous_z_m + self.maximum_step_up_speed_mps * dt_s,
            )
        else:
            next_z_m = max(
                desired_z_m,
                previous_z_m - self.maximum_step_down_speed_mps * dt_s,
            )
        # Collision safety is stronger than the smoothing preference.  The
        # look-ahead normally raises the root before an edge; this clamp is a
        # deterministic final guard against the character entering a block.
        self._root_z_m = max(next_z_m, surface + self.ground_clearance_m)
        self._vertical_speed_mps = (self._root_z_m - previous_z_m) / dt_s
        completed = self.route_distance_m >= self.total_length_m
        return self._pose(0.0 if completed else speed)

    def hold(self) -> TargetMotionPose:
        """Return the current grounded pose without advancing route progress."""

        return self._pose(0.0)

    def _pose(self, speed_mps: float) -> TargetMotionPose:
        x_m, y_m = self._xy_at(self.route_distance_m)
        surface_z_m, feature_id, ahead_surface, ahead_feature = self._surface_context(
            self.route_distance_m
        )
        behind_distance = max(0.0, self.route_distance_m - self.step_anticipation_m)
        behind_x, behind_y = self._xy_at(behind_distance)
        behind_feature = self._feature_at(behind_x, behind_y)
        if ahead_feature is not None and (
            feature_id is None or ahead_surface > surface_z_m + 0.015
        ):
            mode = "step_up"
            active_feature = ahead_feature
        elif feature_id is not None and ahead_surface < surface_z_m - 0.015:
            mode = "step_down"
            active_feature = feature_id
        elif feature_id is None and behind_feature is not None and (
            self._root_z_m > surface_z_m + self.ground_clearance_m + 0.015
        ):
            mode = "step_down"
            active_feature = behind_feature
        elif feature_id is not None:
            mode = "traversable_surface"
            active_feature = feature_id
        else:
            mode = "walk"
            active_feature = None
        return TargetMotionPose(
            position_m=(x_m, y_m, self._root_z_m),
            yaw_deg=self.yaw_deg,
            speed_mps=float(speed_mps),
            route_distance_m=self.route_distance_m,
            completed=self.route_distance_m >= self.total_length_m,
            locomotion_mode=mode,
            traversable_feature_id=active_feature,
            walkable_surface_z_m=surface_z_m,
            surface_clearance_m=max(
                0.0, self._root_z_m - surface_z_m - self.ground_clearance_m
            ),
            vertical_speed_mps=self._vertical_speed_mps,
        )

    def _surface_at(self, distance_m: float) -> float:
        x_m, y_m = self._xy_at(distance_m)
        base_z_m = terrain_mesh_height(
            self.map_config, x_m, y_m,
            grid_spacing_m=self.terrain_grid_spacing_m,
            terrain_buffer_m=self.terrain_buffer_m,
        )
        return traversable_feature_surface_height(
            self.map_config, x_m, y_m, base_height_m=base_z_m
        )

    def _surface_context(
        self, distance_m: float
    ) -> tuple[float, str | None, float, str | None]:
        x_m, y_m = self._xy_at(distance_m)
        ahead_distance = min(self.total_length_m, distance_m + self.step_anticipation_m)
        ahead_x, ahead_y = self._xy_at(ahead_distance)
        return (
            self._surface_at(distance_m),
            self._feature_at(x_m, y_m),
            self._surface_at(ahead_distance),
            self._feature_at(ahead_x, ahead_y),
        )

    def _feature_at(self, x_m: float, y_m: float) -> str | None:
        for feature in self.map_config.get("traversable_step_features", []):
            kind = str(feature.get("kind", ""))
            if kind not in {"curb_table", "platform", "stair_platform"}:
                continue
            cx, cy = map(float, feature["center_m"])
            yaw = float(feature.get("yaw_deg", 0.0)) * pi / 180.0
            dx, dy = x_m - cx, y_m - cy
            along = dx * cos(yaw) + dy * sin(yaw)
            across = -dx * sin(yaw) + dy * cos(yaw)
            if abs(across) > float(feature["width_m"]) * 0.5:
                continue
            if kind in {"curb_table", "platform"}:
                half_length = float(feature["length_m"]) * 0.5
            else:
                half_length = (
                    2.0 * int(feature["step_count"]) * float(feature["tread_depth_m"])
                    + float(feature["platform_length_m"])
                ) * 0.5
            if abs(along) <= half_length:
                return str(feature["id"])
        return None

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
        # A person approaches raised geometry deliberately instead of keeping
        # the flat-ground playback speed until the collision edge.
        feature_probe_m = max(0.8, self.step_anticipation_m)
        probes = (
            max(0.0, distance_m - 0.25),
            distance_m,
            min(self.total_length_m, distance_m + feature_probe_m),
        )
        traversing = any(
            self._feature_at(*self._xy_at(probe)) is not None for probe in probes
        )
        if traversing:
            speed = min(speed, self.traversal_speed_mps)
        lower_bound = min(self.minimum_speed_mps, self.traversal_speed_mps) if traversing else self.minimum_speed_mps
        return max(lower_bound, min(self.maximum_speed_mps, speed))
