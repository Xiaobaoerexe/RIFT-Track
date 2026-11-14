#!/usr/bin/env python3
"""Validate the unchanged-scene, no-event, ten-minute pure-e2e gate."""

from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/ten_minute_acceptance.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("result", type=Path, nargs="?")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    required = spec["required"]
    result_path = args.result or ROOT / spec["result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recovery = result["truth_free_visual_recovery"]

    checks = {
        "episode success": result["state"] == "success",
        "full 600 second horizon": isclose(
            result["elapsed_s"], required["duration_s"], abs_tol=1e-6
        ),
        "configured horizon matches": isclose(
            result["configured_horizon_s"], required["duration_s"], abs_tol=1e-6
        ),
        "pure_e2e mode": result["control_mode"] == required["control_mode"],
        "truth leakage guard": result["actor_truth_leakage_guard"] is True,
        "no privileged actor inputs": result["privileged_actor_inputs"] == [],
        "no scripted events": result["scripted_scene_events"] == required["scripted_scene_events"],
        "no distractor people": result["distractor_people"] == required["distractor_people"],
        "unchanged collider count": result["physics_colliders"] == required["physics_colliders"],
        "route completed": result["route_distance_completed_m"] >= result["route_length_m"] - 0.01,
        "tracking coverage": result["tracking_coverage"] >= required["minimum_tracking_coverage"],
        "tracker visible ratio": result["tracker_visible_ratio"] >= required["minimum_tracker_visible_ratio"],
        "no collision": result["collision_count"] <= required["maximum_collision_count"],
        "no fall": result["fall_count"] <= required["maximum_fall_count"],
        "no boundary violation": result["boundary_violation_count"] <= required["maximum_boundary_violation_count"],
        "recovery finished tracking": recovery["state"] == "tracking",
        "no forbidden recovery inputs": recovery["forbidden_inputs_used"] == [],
        "road friction unchanged": result["road_friction"] == required["road_friction"],
        "D455 intrinsics unchanged": result["d455_intrinsics"] == required["d455_intrinsics"],
        "video exists": Path(result["video"]).is_file(),
        "final frame exists": Path(result["final_frame"]).is_file(),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise SystemExit("TEN_MINUTE_GATE_FAILED: " + ", ".join(failures))
    print(
        "TEN_MINUTE_GATE_OK",
        f"elapsed_s={result['elapsed_s']:.1f}",
        f"route_m={result['route_distance_completed_m']:.3f}",
        f"coverage={result['tracking_coverage']:.4f}",
        f"visible={result['tracker_visible_ratio']:.4f}",
        f"losses={recovery['loss_events']}",
        f"reacquisitions={recovery['reacquisitions']}",
        "events=0 collisions=0 falls=0 boundary=0",
    )


if __name__ == "__main__":
    main()
