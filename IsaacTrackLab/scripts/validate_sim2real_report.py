#!/usr/bin/env python3
"""Validate an actual Isaac sim-to-real smoke report against acceptance invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_D455 = {
    "image_size": [640, 480],
    "fx": 391.18,
    "fy": 391.18,
    "cx": 321.78,
    "cy": 239.94,
    "body_relative_translation_m": [0.32, 0.0, 0.32],
    "body_relative_quaternion_xyzw_ros": [
        0.54167522, -0.54167522, 0.45451948, -0.45451948,
    ],
    "optical_center_height_m": 0.66,
    "pitch_down_deg": 10.0,
    "rgb_hz": 30.0,
    "depth_hz": 30.0,
}
FORBIDDEN_ACTOR_FIELDS = {
    "target_world_pose", "robot_world_pose", "ground_truth_bbox",
    "route_progress", "semantic_segmentation", "simulator_object_id",
}


def validate_report(report: dict[str, Any], *, require_all_events: bool = False) -> list[str]:
    errors: list[str] = []
    d455 = report.get("d455_frozen", {})
    for key, expected in EXPECTED_D455.items():
        if d455.get(key) != expected:
            errors.append(f"D455 {key}: expected {expected!r}, got {d455.get(key)!r}")
    if report.get("backend") != "IsaacTrackingBackend":
        errors.append("report did not come from IsaacTrackingBackend")
    actor = set(report.get("actor_input_fields", []))
    privileged = set(report.get("privileged_input_fields", []))
    if actor & (privileged | FORBIDDEN_ACTOR_FIELDS):
        errors.append("actor input contains privileged or forbidden truth fields")
    if report.get("truth_leakage_detected") is not False:
        errors.append("truth_leakage_detected is not false")

    timing = report.get("timing", {})
    frames = timing.get("frame_timestamps", [])
    commands = timing.get("policy_command_timestamps", [])
    if not frames:
        errors.append("no RGB/Depth timestamp records")
    if not commands:
        errors.append("no policy command timestamp records")
    for row in frames:
        if float(row.get("available_time_s", -1)) < float(row.get("capture_time_s", 0)):
            errors.append("frame is available before it was captured")
            break
    for row in commands:
        if row.get("scheduled") and float(row.get("effective_time_s", -1)) < float(row.get("policy_time_s", 0)):
            errors.append("command becomes effective before policy time")
            break
    for modality in ("rgb", "depth"):
        latency = timing.get("latency", {}).get(modality, {})
        if latency.get("count", 0) != len(latency.get("samples_ms", [])):
            errors.append(f"{modality} latency count does not match samples")

    sensor = report.get("sensor_statistics", {})
    invalid = sensor.get("depth_invalid_fraction", {})
    if sensor.get("processed_frame_count", 0) <= 0:
        errors.append("no processed sensor frames")
    if not invalid.get("samples"):
        errors.append("no per-frame depth invalid-fraction samples")
    elif not all(0.0 <= float(value) <= 1.0 for value in invalid["samples"]):
        errors.append("depth invalid fraction is outside [0, 1]")

    events = report.get("events", [])
    executed = sum(
        int(row.get("apply_count", 0) > 0 and row.get("skip_reason") is None)
        for row in events
    )
    if executed != report.get("executed_event_count"):
        errors.append("executed_event_count disagrees with event evidence")
    if require_all_events:
        skipped = [row for row in events if row.get("skip_reason")]
        inactive = [row for row in events if int(row.get("apply_count", 0)) == 0]
        if skipped:
            errors.append(f"events were skipped: {skipped}")
        if inactive:
            errors.append(f"events were not activated: {inactive}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--require-all-events", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    errors = validate_report(payload, require_all_events=args.require_all_events)
    result = {"report": str(args.report), "valid": not errors, "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
