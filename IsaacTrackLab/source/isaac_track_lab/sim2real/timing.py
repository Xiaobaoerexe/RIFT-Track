"""Timestamped sensor, inference, command, and low-level actuation queues."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class TimestampedFrame:
    modality: str
    capture_time_s: float
    available_time_s: float
    payload: Any
    sequence: int
    repeated: bool = False
    source_capture_time_s: float | None = None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100.0
    lower, upper = int(index), min(len(ordered) - 1, int(index) + 1)
    alpha = index - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


class SensorControlTimeline:
    """Causal queues: consumers can never read a frame or command from the future."""

    def __init__(self, config: dict[str, Any], seed: int, enabled: dict[str, bool]) -> None:
        self.config = config
        self.seed = int(seed)
        self.rng = random.Random(seed ^ 0x51A7E)
        self.enabled = enabled
        self.sensor_heap: list[tuple[float, int, TimestampedFrame]] = []
        self.action_heap: list[tuple[float, int, tuple[float, float, float]]] = []
        self.last_delivered: dict[str, TimestampedFrame] = {}
        self.latest_available: dict[str, TimestampedFrame] = {}
        self.sequence = 0
        self.action_sequence = 0
        self.active_command = (0.0, 0.0, 0.0)
        self.latency_samples_ms: dict[str, list[float]] = {
            "rgb": [], "depth": [], "inference": [], "command": [], "actuation": [],
        }
        self.capture_counts = {"rgb": 0, "depth": 0}
        self.drop_counts = {"rgb": 0, "depth": 0}
        self.missed_policy_updates = 0
        self.frame_records: list[dict[str, Any]] = []
        self.command_records: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.__init__(self.config, self.seed, self.enabled)

    def _sample_ms(self, key: str, enabled: bool) -> float:
        if not enabled:
            return 0.0
        spec = self.config[key]
        if isinstance(spec, (int, float)):
            value = float(spec)
        else:
            value = self.rng.uniform(float(spec[0]), float(spec[1]))
        self.latency_samples_ms[key].append(value)
        return value

    def capture(self, modality: str, capture_time_s: float, payload: Any) -> TimestampedFrame:
        if modality not in ("rgb", "depth"):
            raise ValueError(f"unsupported modality {modality}")
        self.capture_counts[modality] += 1
        sensor_on = self.enabled["sensor_latency"]
        drop_probability = float(self.config[f"{modality}_drop_probability"]) if sensor_on else 0.0
        dropped = self.rng.random() < drop_probability
        source = self.last_delivered.get(modality) or self.latest_available.get(modality)
        repeated = dropped and source is not None
        if dropped:
            self.drop_counts[modality] += 1
        actual_payload = source.payload if repeated else payload
        source_time = source.capture_time_s if repeated else capture_time_s
        latency_ms = self._sample_ms(modality, sensor_on)
        if modality == "depth" and sensor_on:
            extra_ms = float(self.config.get("depth_extra_delay_ms", 0.0))
            latency_ms += extra_ms
            # Report the actual end-to-end Depth delay, not only the sampled
            # transport component before the configured RGB/Depth skew.
            self.latency_samples_ms[modality][-1] += extra_ms
        frame = TimestampedFrame(
            modality=modality,
            capture_time_s=capture_time_s,
            available_time_s=capture_time_s + latency_ms / 1000.0,
            payload=actual_payload,
            sequence=self.sequence,
            repeated=repeated,
            source_capture_time_s=source_time,
        )
        self.sequence += 1
        heapq.heappush(self.sensor_heap, (frame.available_time_s, frame.sequence, frame))
        self.frame_records.append({
            "modality": modality,
            "sequence": frame.sequence,
            "capture_time_s": frame.capture_time_s,
            "available_time_s": frame.available_time_s,
            "latency_ms": latency_ms,
            "repeated": frame.repeated,
            "source_capture_time_s": frame.source_capture_time_s,
        })
        return frame

    def observation(self, now_s: float) -> dict[str, TimestampedFrame]:
        while self.sensor_heap and self.sensor_heap[0][0] <= now_s + 1e-12:
            _, _, frame = heapq.heappop(self.sensor_heap)
            self.latest_available[frame.modality] = frame
        for modality, frame in self.latest_available.items():
            self.last_delivered[modality] = frame
        return dict(self.last_delivered)

    def schedule_policy_command(
        self,
        policy_time_s: float,
        command: tuple[float, float, float],
        *,
        measured_inference_ms: float | None = None,
    ) -> bool:
        action_on = self.enabled["action_latency"]
        if action_on and self.rng.random() < float(self.config["missed_policy_update_probability"]):
            self.missed_policy_updates += 1
            self.command_records.append({
                "policy_time_s": policy_time_s, "scheduled": False,
                "command": list(map(float, command)), "reason": "missed_policy_update",
            })
            return False
        inference_ms = (
            float(measured_inference_ms)
            if measured_inference_ms is not None
            else self._sample_ms("inference", self.enabled["sensor_latency"])
        )
        if measured_inference_ms is not None:
            self.latency_samples_ms["inference"].append(inference_ms)
        command_ms = self._sample_ms("command", action_on)
        actuation_ms = self._sample_ms("actuation", action_on)
        effective = policy_time_s + (inference_ms + command_ms + actuation_ms) / 1000.0
        heapq.heappush(self.action_heap, (effective, self.action_sequence, tuple(map(float, command))))
        self.command_records.append({
            "policy_time_s": policy_time_s, "scheduled": True,
            "command": list(map(float, command)),
            "inference_ms": inference_ms, "command_transport_ms": command_ms,
            "actuation_ms": actuation_ms, "effective_time_s": effective,
        })
        self.action_sequence += 1
        return True

    def command(self, now_s: float) -> tuple[float, float, float]:
        while self.action_heap and self.action_heap[0][0] <= now_s + 1e-12:
            _, _, self.active_command = heapq.heappop(self.action_heap)
        return self.active_command

    def report(self) -> dict[str, Any]:
        latency = {}
        for name, values in self.latency_samples_ms.items():
            latency[name] = {
                "count": len(values),
                "samples_ms": list(values),
                "mean_ms": fmean(values) if values else None,
                "p95_ms": _percentile(values, 95),
                "maximum_ms": max(values) if values else None,
            }
        return {
            "latency": latency,
            "captures": dict(self.capture_counts),
            "drops": dict(self.drop_counts),
            "drop_rate": {
                key: self.drop_counts[key] / self.capture_counts[key]
                if self.capture_counts[key] else 0.0
                for key in self.capture_counts
            },
            "missed_policy_updates": self.missed_policy_updates,
            "frame_timestamps": list(self.frame_records),
            "policy_command_timestamps": list(self.command_records),
        }
