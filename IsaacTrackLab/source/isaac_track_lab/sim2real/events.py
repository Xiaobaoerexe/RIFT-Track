"""Runtime state for scenario events; adapters apply these values to Isaac."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from isaac_track_lab.task.scenario import DisturbanceEvent

from .randomization import event_weight


@dataclass
class ExecutedEvent:
    kind: str
    variant: str
    planned_start_s: float
    planned_duration_s: float
    actual_start_s: float | None = None
    actual_end_s: float | None = None
    maximum_weight: float = 0.0
    apply_count: int = 0
    asset: str | None = None
    trajectory: list[dict[str, float]] | None = None
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScenarioEventRuntime:
    """Turns a deterministic event plan into continuous per-step commands and evidence."""

    def __init__(self, events: Iterable[DisturbanceEvent]) -> None:
        self.events = tuple(events)
        self.records = [
            ExecutedEvent(e.kind, e.variant, e.start_s, e.duration_s)
            for e in self.events
        ]

    def update(self, now_s: float) -> list[tuple[DisturbanceEvent, float, ExecutedEvent]]:
        active = []
        for event, record in zip(self.events, self.records, strict=True):
            weight = event_weight(now_s, event.start_s, event.duration_s)
            if weight > 0.0:
                if record.actual_start_s is None:
                    record.actual_start_s = now_s
                record.maximum_weight = max(record.maximum_weight, weight)
                record.apply_count += 1
                active.append((event, weight, record))
            elif record.actual_start_s is not None and record.actual_end_s is None and now_s >= event.start_s + event.duration_s:
                record.actual_end_s = now_s
        return active

    def mark_skipped(self, index: int, reason: str) -> None:
        self.records[index].skip_reason = reason

    def report(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]
