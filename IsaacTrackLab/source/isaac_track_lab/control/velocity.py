"""Replaceable high-level base-velocity interface for an As2 locomotion policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VelocityCommand:
    vx_mps: float
    vy_mps: float
    yaw_rate_rps: float


@runtime_checkable
class VelocityController(Protocol):
    """Adapter boundary for vendor or learned low-level locomotion."""

    def reset(self) -> None: ...

    def apply(self, command: VelocityCommand) -> None: ...
