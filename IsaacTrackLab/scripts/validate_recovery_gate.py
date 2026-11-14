#!/usr/bin/env python3
"""Validate the current-scene physical occlusion and truth-free recovery gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT / "outputs/baseline_305m/recovery/physical_occlusion_recovery_gate_v4.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, nargs="?", default=DEFAULT_RESULT)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    recovery = result["truth_free_visual_recovery"]
    occlusion = result["forced_physical_occlusion"]
    checks = {
        "episode success": result["state"] == "success",
        "pure_e2e mode": result["control_mode"] == "pure_e2e",
        "truth leakage guard": result["actor_truth_leakage_guard"] is True,
        "no privileged actor inputs": result["privileged_actor_inputs"] == [],
        "one physical occlusion executed": occlusion["events_executed"] == 1,
        "collision proxy present": occlusion["collision_proxy_m"] == [1.2, 3.0, 1.5],
        "visual loss detected": recovery["loss_events"] >= 1,
        "every visual loss reacquired": recovery["reacquisitions"] == recovery["loss_events"],
        "finished tracking": recovery["state"] == "tracking",
        "no forbidden recovery input": recovery["forbidden_inputs_used"] == [],
        "no truth reinitialization": result["appearance_reinitializations"] == 0,
        "no collision": result["collision_count"] == 0,
        "no fall": result["fall_count"] == 0,
        "no boundary violation": result["boundary_violation_count"] == 0,
        "distance recovered": result["final_distance_m"] <= 4.0,
        "tracking coverage": result["tracking_coverage"] >= 0.90,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise SystemExit("RECOVERY_GATE_FAILED: " + ", ".join(failures))
    print(
        "RECOVERY_GATE_OK",
        f"losses={recovery['loss_events']}",
        f"reacquisitions={recovery['reacquisitions']}",
        f"coverage={result['tracking_coverage']:.4f}",
        f"final_distance_m={result['final_distance_m']:.3f}",
        "collisions=0 falls=0 boundary=0",
    )


if __name__ == "__main__":
    main()
