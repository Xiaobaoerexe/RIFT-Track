"""Simulator-neutral reset/step boundary for modular and end-to-end training.

The actor observation and privileged evaluator signals are different types on
purpose.  An Isaac Sim backend may produce both, but only ``ActorObservation``
is returned to the policy from ``reset`` and ``step``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any, Protocol, runtime_checkable

from ..task.runtime import (
    EpisodeState,
    StepObservation,
    TaskEvaluator,
    TerminationThresholds,
)


def _finite_vector(name: str, values: tuple[float, ...], expected: int | None = None) -> None:
    if expected is not None and len(values) != expected:
        raise ValueError(f"{name} must contain {expected} values")
    if not values or not all(isfinite(value) for value in values):
        raise ValueError(f"{name} must be a non-empty finite vector")


def _validate_image(name: str, image: Any) -> None:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 3 or int(shape[-1]) != 3:
        raise ValueError(f"{name} must have HxWx3 shape")
    if int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError(f"{name} dimensions must be positive")


@dataclass(frozen=True)
class ActorObservation:
    """The complete and exclusive deployable actor input."""

    rgb: Any
    target_template_rgb: Any
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    base_angular_velocity: tuple[float, float, float]
    projected_gravity: tuple[float, float, float]
    foot_contacts: tuple[float, float, float, float]
    aligned_depth: Any | None = None

    def validate(self) -> None:
        _validate_image("rgb", self.rgb)
        _validate_image("target_template_rgb", self.target_template_rgb)
        if self.aligned_depth is not None:
            shape = getattr(self.aligned_depth, "shape", None)
            if shape is None or len(shape) not in (2, 3):
                raise ValueError("aligned_depth must have HxW, HxWx1, or HxWx2 shape")
            if len(shape) == 3 and int(shape[-1]) not in (1, 2):
                raise ValueError(
                    "aligned_depth channel count must be 1 (depth) or 2 "
                    "(depth plus explicit validity mask)"
                )
        _finite_vector("joint_position", self.joint_position)
        _finite_vector("joint_velocity", self.joint_velocity)
        if len(self.joint_position) != len(self.joint_velocity):
            raise ValueError("joint position and velocity lengths must match")
        _finite_vector("base_angular_velocity", self.base_angular_velocity, 3)
        _finite_vector("projected_gravity", self.projected_gravity, 3)
        _finite_vector("foot_contacts", self.foot_contacts, 4)

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))


@dataclass(frozen=True)
class PrivilegedSignals:
    """Simulator truth consumed by rewards, termination and metrics only."""

    target_distance_m: float
    target_bbox_area_ratio: float
    base_contact_impulse_ns: float
    base_height_m: float
    base_tilt_deg: float
    in_bounds: bool


@dataclass(frozen=True)
class PlanarVelocityAction:
    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float

    def validate(self) -> None:
        if not all(isfinite(value) for value in self.as_tuple()):
            raise ValueError("action values must be finite")

    def as_tuple(self) -> tuple[float, float, float]:
        return self.vx_mps, self.vy_mps, self.yaw_rate_rps


@dataclass(frozen=True)
class EpisodeRequest:
    scenario_id: str
    seed: int
    horizon_s: float

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if not isfinite(self.horizon_s) or self.horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive and finite")


@dataclass(frozen=True)
class BackendTransition:
    actor_observation: ActorObservation
    privileged: PrivilegedSignals
    elapsed_s: float


@runtime_checkable
class TrackingBackend(Protocol):
    """Minimal adapter implemented by Isaac Sim and deterministic test backends."""

    def reset(self, request: EpisodeRequest) -> BackendTransition: ...

    def step(self, action: PlanarVelocityAction) -> BackendTransition: ...


class StandardTrackingEnv:
    """Gym-like environment that enforces the actor/truth separation."""

    def __init__(self, backend: TrackingBackend, thresholds: TerminationThresholds):
        self.backend = backend
        self.thresholds = thresholds
        self._request: EpisodeRequest | None = None
        self._evaluator: TaskEvaluator | None = None
        self._last_elapsed_s = 0.0

    def reset(self, request: EpisodeRequest) -> tuple[ActorObservation, dict[str, Any]]:
        request.validate()
        transition = self.backend.reset(request)
        if transition.elapsed_s != 0.0:
            raise ValueError("backend reset transition must have elapsed_s == 0")
        transition.actor_observation.validate()
        self._request = request
        self._evaluator = TaskEvaluator(self.thresholds, request.horizon_s)
        self._last_elapsed_s = 0.0
        return transition.actor_observation, {
            "scenario_id": request.scenario_id,
            "seed": request.seed,
            "actor_observation_keys": ActorObservation.keys(),
        }

    def step(
        self, action: PlanarVelocityAction
    ) -> tuple[ActorObservation, float, bool, bool, dict[str, Any]]:
        if self._evaluator is None or self._request is None:
            raise RuntimeError("reset must be called before step")
        if self._evaluator.done:
            raise RuntimeError("episode has already terminated")
        action.validate()
        transition = self.backend.step(action)
        transition.actor_observation.validate()
        privileged = transition.privileged
        result = self._evaluator.update(
            StepObservation(
                elapsed_s=transition.elapsed_s,
                target_distance_m=privileged.target_distance_m,
                target_bbox_area_ratio=privileged.target_bbox_area_ratio,
                base_contact_impulse_ns=privileged.base_contact_impulse_ns,
                base_height_m=privileged.base_height_m,
                base_tilt_deg=privileged.base_tilt_deg,
                in_bounds=privileged.in_bounds,
                control=action.as_tuple(),
            )
        )
        dt = transition.elapsed_s - self._last_elapsed_s
        self._last_elapsed_s = transition.elapsed_s
        visible = privileged.target_bbox_area_ratio >= self.thresholds.minimum_bbox_area_ratio
        distance_error = abs(
            privileged.target_distance_m - self.thresholds.desired_distance_m
        )
        reward = dt * (1.0 - 0.25 * distance_error - (0.0 if visible else 1.0))
        if result.state is EpisodeState.FAILURE:
            reward -= 10.0
        terminated = result.state is EpisodeState.FAILURE
        truncated = result.state is EpisodeState.SUCCESS
        info: dict[str, Any] = {
            "scenario_id": self._request.scenario_id,
            "seed": self._request.seed,
            "elapsed_s": transition.elapsed_s,
            "actor_truth_isolated": True,
        }
        if result.state is not EpisodeState.RUNNING:
            info["episode_result"] = result.to_dict()
        return transition.actor_observation, reward, terminated, truncated, info
