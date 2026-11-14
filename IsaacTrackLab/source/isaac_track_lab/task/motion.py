"""Deterministic scripted actor motion used by scene and task smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class OcclusionEvent:
    start_s: float
    duration_s: float

    def active(self, elapsed_s: float) -> bool:
        return self.start_s <= elapsed_s < self.start_s + self.duration_s


def scheduled_visibility(elapsed_s: float, events: Sequence[OcclusionEvent]) -> bool:
    return not any(event.active(elapsed_s) for event in events)


class RouteFollower:
    """Constant-speed follower over a polyline, optionally ping-ponging at ends."""

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        speed_mps: float,
        *,
        ping_pong: bool = False,
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("at least two waypoints are required")
        if speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive")
        self.waypoints: tuple[Point3, ...] = tuple(
            (float(point[0]), float(point[1]), float(point[2])) for point in waypoints
        )
        self.speed_mps = float(speed_mps)
        self.ping_pong = ping_pong
        self._lengths = tuple(
            self._distance(start, end)
            for start, end in zip(self.waypoints, self.waypoints[1:])
        )
        if any(length <= 0.0 for length in self._lengths):
            raise ValueError("adjacent waypoints must be distinct")
        self.total_length_m = sum(self._lengths)

    def position(self, elapsed_s: float) -> Point3:
        if elapsed_s < 0.0:
            raise ValueError("elapsed_s cannot be negative")
        distance = elapsed_s * self.speed_mps
        if self.ping_pong:
            cycle = 2.0 * self.total_length_m
            distance %= cycle
            if distance > self.total_length_m:
                distance = cycle - distance
        else:
            distance = min(distance, self.total_length_m)

        for start, end, segment_length in zip(
            self.waypoints, self.waypoints[1:], self._lengths
        ):
            if distance <= segment_length:
                ratio = distance / segment_length
                return tuple(
                    start[axis] + ratio * (end[axis] - start[axis])
                    for axis in range(3)
                )  # type: ignore[return-value]
            distance -= segment_length
        return self.waypoints[-1]

    @staticmethod
    def _distance(a: Point3, b: Point3) -> float:
        return sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def distance_to_polyline(point: Sequence[float], waypoints: Sequence[Sequence[float]]) -> float:
    """Return 2-D distance from a point to the closest route segment."""

    if len(waypoints) < 2:
        raise ValueError("at least two waypoints are required")
    px, py = float(point[0]), float(point[1])
    closest = float("inf")
    for start, end in zip(waypoints, waypoints[1:]):
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if length_squared == 0.0:
            candidate_x, candidate_y = ax, ay
        else:
            ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_squared))
            candidate_x, candidate_y = ax + ratio * dx, ay + ratio * dy
        closest = min(closest, sqrt((px - candidate_x) ** 2 + (py - candidate_y) ** 2))
    return closest


def inside_route_corridor(
    point: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    half_width_m: float,
    margin_m: float = 0.0,
) -> bool:
    if half_width_m < 0.0 or margin_m < 0.0:
        raise ValueError("corridor width and margin cannot be negative")
    return distance_to_polyline(point, waypoints) <= half_width_m + margin_m
