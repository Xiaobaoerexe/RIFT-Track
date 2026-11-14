"""Deterministic expansion of the six scenario families into episode plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

from .motion import OcclusionEvent


@dataclass(frozen=True)
class DisturbanceEvent:
    kind: str
    variant: str
    start_s: float
    duration_s: float
    recovery: str | None = None


@dataclass(frozen=True)
class ScenarioPlan:
    scenario_id: str
    seed: int
    duration_s: float
    interference: str
    distractor_count: int
    occlusion_events: tuple[OcclusionEvent, ...]
    disturbance_events: tuple[DisturbanceEvent, ...] = ()
    checkpoint_interval_s: float = 300.0

    @property
    def checkpoint_count(self) -> int:
        return int(self.duration_s / self.checkpoint_interval_s)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checkpoint_count"] = self.checkpoint_count
        return data


def build_scenario_plan(
    scenario: dict[str, Any],
    occlusion_config: dict[str, Any],
    seed: int,
    disturbance_config: dict[str, Any] | None = None,
    checkpoint_interval_s: float = 300.0,
) -> ScenarioPlan:
    """Sample counts and non-overlapping-in-practice event times from one seed."""

    rng = random.Random(seed)
    duration_s = float(scenario["duration_s"])
    distractor_range = scenario["distractors"]
    occlusion_range = scenario["forced_occlusions"]
    distractor_count = rng.randint(int(distractor_range[0]), int(distractor_range[1]))
    occlusion_count = rng.randint(int(occlusion_range[0]), int(occlusion_range[1]))
    minimum_duration, maximum_duration = map(float, occlusion_config["duration_s"])

    events: list[OcclusionEvent] = []
    disturbance_events: list[DisturbanceEvent] = []
    if occlusion_count:
        slot_s = duration_s / (occlusion_count + 1)
        for index in range(occlusion_count):
            event_duration = rng.uniform(minimum_duration, maximum_duration)
            nominal_start = (index + 1) * slot_s - event_duration / 2.0
            jitter = rng.uniform(-0.2 * slot_s, 0.2 * slot_s)
            start_s = max(1.0, min(nominal_start + jitter, duration_s - event_duration - 1.0))
            events.append(OcclusionEvent(start_s=start_s, duration_s=event_duration))
            disturbance_events.append(DisturbanceEvent(
                kind="occlusion",
                variant=rng.choice(occlusion_config.get("types", ["static_occluder"])),
                start_s=start_s,
                duration_s=event_duration,
                recovery=rng.choice(occlusion_config.get("reappearance", ["same_side"])),
            ))

    disturbance_config = disturbance_config or {}
    scenario_ranges = {
        "appearance_change": scenario.get("appearance_changes", [0, 0]),
        "lighting_change": scenario.get("lighting_changes", [0, 0]),
        "sensor_degradation": scenario.get("sensor_degradations", [0, 0]),
    }
    for kind, count_range in scenario_ranges.items():
        count = rng.randint(int(count_range[0]), int(count_range[1]))
        if not count:
            continue
        event_config = disturbance_config.get(kind, {})
        minimum, maximum = map(float, event_config.get("duration_s", [1.0, 1.0]))
        variants = event_config.get("types", [kind])
        slot_s = duration_s / (count + 1)
        for index in range(count):
            event_duration = rng.uniform(minimum, maximum)
            nominal_start = (index + 1) * slot_s - event_duration / 2.0
            jitter = rng.uniform(-0.2 * slot_s, 0.2 * slot_s)
            start_s = max(1.0, min(nominal_start + jitter, duration_s - event_duration - 1.0))
            disturbance_events.append(DisturbanceEvent(
                kind=kind,
                variant=rng.choice(variants),
                start_s=start_s,
                duration_s=event_duration,
            ))

    return ScenarioPlan(
        scenario_id=str(scenario["id"]),
        seed=int(seed),
        duration_s=duration_s,
        interference=str(scenario["interference"]),
        distractor_count=distractor_count,
        occlusion_events=tuple(events),
        disturbance_events=tuple(sorted(disturbance_events, key=lambda event: event.start_s)),
        checkpoint_interval_s=float(checkpoint_interval_s),
    )
