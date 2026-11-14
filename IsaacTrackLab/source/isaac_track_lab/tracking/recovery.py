"""Truth-free visual loss detection and active reacquisition control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from ..control.velocity import VelocityCommand


class RecoveryState(StrEnum):
    TRACKING = "tracking"
    SEARCHING = "searching"
    VERIFYING = "verifying"


@dataclass(frozen=True)
class RecoveryConfig:
    loss_confirmation_frames: int = 2
    reacquire_confirmation_frames: int = 3
    search_yaw_rate_rps: float = 0.45
    sweep_duration_s: float = 1.5
    verifying_forward_mps: float = 0.25


@dataclass(frozen=True)
class RecoveryMetrics:
    state: RecoveryState
    loss_events: int
    reacquisitions: int
    current_lost_s: float
    total_lost_s: float


class VisualRecoveryController:
    """State machine driven only by deployable tracker outputs."""

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()
        if self.config.loss_confirmation_frames <= 0:
            raise ValueError("loss_confirmation_frames must be positive")
        if self.config.reacquire_confirmation_frames <= 0:
            raise ValueError("reacquire_confirmation_frames must be positive")
        if self.config.sweep_duration_s <= 0.0:
            raise ValueError("sweep_duration_s must be positive")
        self.state = RecoveryState.TRACKING
        self._invisible_frames = 0
        self._visible_frames = 0
        self._loss_events = 0
        self._reacquisitions = 0
        self._lost_since_s: float | None = None
        self._total_lost_s = 0.0
        self._last_horizontal_direction = 1.0

    def update(
        self,
        elapsed_s: float,
        tracker_visible: bool,
        horizontal_error: float | None,
    ) -> RecoveryState:
        if not isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValueError("elapsed_s must be finite and non-negative")
        if horizontal_error is not None:
            if not isfinite(horizontal_error):
                raise ValueError("horizontal_error must be finite")
            if abs(horizontal_error) > 1e-3:
                # Image x grows to the right, while the Go2 command convention
                # uses negative yaw to turn right.  Preserve the last *command*
                # direction, not the raw image-error sign, for loss recovery.
                self._last_horizontal_direction = -1.0 if horizontal_error > 0.0 else 1.0

        if self.state is RecoveryState.TRACKING:
            self._invisible_frames = 0 if tracker_visible else self._invisible_frames + 1
            if self._invisible_frames >= self.config.loss_confirmation_frames:
                self.state = RecoveryState.SEARCHING
                self._loss_events += 1
                self._lost_since_s = elapsed_s
                self._visible_frames = 0
        elif self.state is RecoveryState.SEARCHING:
            if tracker_visible:
                self.state = RecoveryState.VERIFYING
                self._visible_frames = 1
        else:
            if tracker_visible:
                self._visible_frames += 1
                if self._visible_frames >= self.config.reacquire_confirmation_frames:
                    self.state = RecoveryState.TRACKING
                    self._reacquisitions += 1
                    if self._lost_since_s is not None:
                        self._total_lost_s += elapsed_s - self._lost_since_s
                    self._lost_since_s = None
                    self._invisible_frames = 0
            else:
                self.state = RecoveryState.SEARCHING
                self._visible_frames = 0
        return self.state

    def command(self, visual_command: VelocityCommand, elapsed_s: float) -> VelocityCommand:
        if self.state is RecoveryState.TRACKING:
            return visual_command
        if self.state is RecoveryState.VERIFYING:
            return VelocityCommand(
                vx_mps=min(visual_command.vx_mps, self.config.verifying_forward_mps),
                vy_mps=0.0,
                yaw_rate_rps=visual_command.yaw_rate_rps,
            )
        lost_elapsed = max(0.0, elapsed_s - (self._lost_since_s or elapsed_s))
        sweep_index = int(lost_elapsed / self.config.sweep_duration_s)
        sweep_direction = self._last_horizontal_direction * (-1.0 if sweep_index % 2 else 1.0)
        return VelocityCommand(
            vx_mps=0.0,
            vy_mps=0.0,
            yaw_rate_rps=sweep_direction * self.config.search_yaw_rate_rps,
        )

    def metrics(self, elapsed_s: float) -> RecoveryMetrics:
        current = (
            max(0.0, elapsed_s - self._lost_since_s)
            if self._lost_since_s is not None else 0.0
        )
        return RecoveryMetrics(
            state=self.state,
            loss_events=self._loss_events,
            reacquisitions=self._reacquisitions,
            current_lost_s=current,
            total_lost_s=self._total_lost_s + current,
        )
