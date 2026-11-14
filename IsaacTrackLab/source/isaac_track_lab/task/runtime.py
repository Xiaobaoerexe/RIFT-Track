"""Simulator-independent episode termination and metric accounting.

The evaluator consumes only scalar evaluation signals.  It deliberately does
not expose privileged state to the tracking/control policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from typing import Any


class EpisodeState(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class FailureReason(StrEnum):
    TARGET_TOO_FAR = "target_too_far"
    TARGET_ABSENT = "target_absent"
    STRONG_COLLISION = "strong_collision"
    FALLEN = "fallen"
    OUT_OF_BOUNDS = "out_of_bounds"


@dataclass(frozen=True)
class TerminationThresholds:
    target_distance_m: float
    target_distance_grace_s: float
    target_absent_grace_s: float
    minimum_bbox_area_ratio: float
    base_contact_impulse_ns: float
    maximum_tilt_deg: float
    minimum_base_height_m: float
    fallen_grace_s: float
    desired_distance_m: float = 3.0

    @classmethod
    def from_task_config(cls, task: dict[str, Any]) -> "TerminationThresholds":
        termination = task["termination"]
        return cls(
            target_distance_m=float(termination["target_too_far"]["distance_m"]),
            target_distance_grace_s=float(termination["target_too_far"]["grace_s"]),
            target_absent_grace_s=float(termination["target_absent"]["grace_s"]),
            minimum_bbox_area_ratio=float(
                termination["target_absent"]["minimum_bbox_area_ratio"]
            ),
            base_contact_impulse_ns=float(
                termination["strong_collision"]["base_contact_impulse_ns"]
            ),
            maximum_tilt_deg=float(termination["fallen"]["maximum_tilt_deg"]),
            minimum_base_height_m=float(
                termination["fallen"]["minimum_base_height_m"]
            ),
            fallen_grace_s=float(termination["fallen"]["grace_s"]),
            desired_distance_m=float(task.get("objective", {}).get("desired_distance_m", 3.0)),
        )


@dataclass(frozen=True)
class StepObservation:
    elapsed_s: float
    target_distance_m: float
    target_bbox_area_ratio: float
    base_contact_impulse_ns: float
    base_height_m: float
    base_tilt_deg: float
    in_bounds: bool
    control: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class EpisodeResult:
    state: EpisodeState
    elapsed_s: float
    failure_reason: FailureReason | None
    tracking_coverage: float
    absence_events: int
    reacquisitions: int
    reacquisition_rate: float
    mean_reacquisition_time_s: float | None
    mean_target_distance_error_m: float
    collision_count: int
    fall_count: int
    boundary_violation_count: int
    control_smoothness: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["failure_reason"] = (
            self.failure_reason.value if self.failure_reason is not None else None
        )
        return data


class TaskEvaluator:
    """Stateful evaluator implementing all task success/failure rules."""

    def __init__(self, thresholds: TerminationThresholds, horizon_s: float):
        if horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        self.thresholds = thresholds
        self.horizon_s = float(horizon_s)
        self.state = EpisodeState.RUNNING
        self.failure_reason: FailureReason | None = None
        self._last_elapsed_s = 0.0
        self._visible_s = 0.0
        self._distance_error_integral = 0.0
        self._far_s = 0.0
        self._absent_s = 0.0
        self._fallen_s = 0.0
        self._absence_events = 0
        self._reacquisition_times: list[float] = []
        self._was_visible = True
        self._collision_count = 0
        self._fall_count = 0
        self._boundary_violation_count = 0
        self._previous_collision = False
        self._previous_control: tuple[float, float, float] | None = None
        self._control_variation = 0.0
        self._control_samples = 0

    @property
    def done(self) -> bool:
        return self.state is not EpisodeState.RUNNING

    def update(self, observation: StepObservation) -> EpisodeResult:
        if self.done:
            raise RuntimeError("episode has already terminated")
        self._validate(observation)
        dt = observation.elapsed_s - self._last_elapsed_s
        self._last_elapsed_s = observation.elapsed_s
        t = self.thresholds

        visible = observation.target_bbox_area_ratio >= t.minimum_bbox_area_ratio
        if visible:
            self._visible_s += dt
            if not self._was_visible:
                self._reacquisition_times.append(self._absent_s)
            self._absent_s = 0.0
        else:
            if self._was_visible:
                self._absence_events += 1
            self._absent_s += dt
        self._was_visible = visible

        self._far_s = (
            self._far_s + dt
            if observation.target_distance_m > t.target_distance_m
            else 0.0
        )
        fallen_now = (
            observation.base_height_m < t.minimum_base_height_m
            or observation.base_tilt_deg > t.maximum_tilt_deg
        )
        self._fallen_s = self._fallen_s + dt if fallen_now else 0.0

        collision_now = observation.base_contact_impulse_ns >= t.base_contact_impulse_ns
        if collision_now and not self._previous_collision:
            self._collision_count += 1
        self._previous_collision = collision_now
        if fallen_now and self._fallen_s >= t.fallen_grace_s and self._fall_count == 0:
            self._fall_count = 1
        if not observation.in_bounds:
            self._boundary_violation_count += 1

        self._distance_error_integral += (
            abs(observation.target_distance_m - t.desired_distance_m) * dt
        )
        if self._previous_control is not None and dt > 0.0:
            delta = sum(
                (current - previous) ** 2
                for current, previous in zip(observation.control, self._previous_control)
            ) ** 0.5
            self._control_variation += delta / dt
            self._control_samples += 1
        self._previous_control = observation.control

        # Immediate safety rules take priority over timer-based tracking rules.
        if collision_now:
            self._fail(FailureReason.STRONG_COLLISION)
        elif self._fallen_s >= t.fallen_grace_s:
            self._fail(FailureReason.FALLEN)
        elif not observation.in_bounds:
            self._fail(FailureReason.OUT_OF_BOUNDS)
        elif self._far_s >= t.target_distance_grace_s:
            self._fail(FailureReason.TARGET_TOO_FAR)
        elif self._absent_s >= t.target_absent_grace_s:
            self._fail(FailureReason.TARGET_ABSENT)
        elif observation.elapsed_s >= self.horizon_s:
            self.state = EpisodeState.SUCCESS

        return self.result()

    def result(self) -> EpisodeResult:
        elapsed = self._last_elapsed_s
        reacquisitions = len(self._reacquisition_times)
        return EpisodeResult(
            state=self.state,
            elapsed_s=elapsed,
            failure_reason=self.failure_reason,
            tracking_coverage=self._visible_s / elapsed if elapsed > 0.0 else 0.0,
            absence_events=self._absence_events,
            reacquisitions=reacquisitions,
            reacquisition_rate=(
                reacquisitions / self._absence_events if self._absence_events else 0.0
            ),
            mean_reacquisition_time_s=(
                sum(self._reacquisition_times) / reacquisitions
                if reacquisitions
                else None
            ),
            mean_target_distance_error_m=(
                self._distance_error_integral / elapsed if elapsed > 0.0 else 0.0
            ),
            collision_count=self._collision_count,
            fall_count=self._fall_count,
            boundary_violation_count=self._boundary_violation_count,
            control_smoothness=(
                self._control_variation / self._control_samples
                if self._control_samples
                else 0.0
            ),
        )

    def _fail(self, reason: FailureReason) -> None:
        self.state = EpisodeState.FAILURE
        self.failure_reason = reason

    def _validate(self, observation: StepObservation) -> None:
        if observation.elapsed_s <= self._last_elapsed_s:
            raise ValueError("elapsed_s must increase monotonically")
        scalar_values = (
            observation.elapsed_s,
            observation.target_distance_m,
            observation.target_bbox_area_ratio,
            observation.base_contact_impulse_ns,
            observation.base_height_m,
            observation.base_tilt_deg,
            *observation.control,
        )
        if not all(isfinite(value) for value in scalar_values):
            raise ValueError("observation values must be finite")
        if observation.target_distance_m < 0.0:
            raise ValueError("target_distance_m cannot be negative")
        if not 0.0 <= observation.target_bbox_area_ratio <= 1.0:
            raise ValueError("target_bbox_area_ratio must be in [0, 1]")
