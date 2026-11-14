#!/usr/bin/env python3
"""Aggregate the three representative Go2 dynamics checks into one acceptance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "outputs/terrain_dataset_v3_realistic/dynamics"
REPRESENTATIVES = ("city_01", "park_01", "forest_trail_01")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    result_dir = args.result_dir if args.result_dir.is_absolute() else ROOT / args.result_dir

    rows = []
    failures = []
    for map_id in REPRESENTATIVES:
        path = result_dir / f"{map_id}_route_01_dynamics.json"
        if not path.is_file():
            failures.append(f"missing result: {path}")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("map_id") != map_id or result.get("task_test") is not False:
            failures.append(f"invalid result identity: {path}")
        if not result.get("passed", False):
            failures.append(f"dynamic acceptance failed: {map_id}")
        rows.append({
            "map_id": map_id,
            "family": result.get("family"),
            "passed": bool(result.get("passed")),
            "settled_base_clearance_m": result.get("settled_base_clearance_m"),
            "route_progress_m": result.get("route_progress_m"),
            "mean_planar_speed_mps": result.get("mean_planar_speed_mps"),
            "cross_track_p95_m": result.get("cross_track_error_m", {}).get("p95"),
            "maximum_tilt_deg": result.get("tilt_deg", {}).get("maximum"),
            "foot_contact_time_ratio": result.get("foot_contact", {}).get("contact_time_ratio"),
            "foot_slip_p95_mps": result.get("foot_contact", {}).get("slip_speed_p95_mps"),
            "sole_clearance_minimum_m": result.get("foot_contact", {}).get("sole_clearance_minimum_m"),
            "sole_penetration_p95_m": result.get("foot_contact", {}).get("penetration_p95_m"),
            "fallen": result.get("fallen"),
            "source": str(path.relative_to(ROOT)),
        })

    report = {
        "schema_version": 1,
        "test": "representative_v3_go2_dynamics_batch",
        "task_test": False,
        "all_passed": not failures and len(rows) == len(REPRESENTATIVES),
        "representative_count": len(rows),
        "maps": rows,
        "failures": failures,
        "scope": [
            "real PhysX terrain contact",
            "real Go2 articulation and locomotion policy",
            "foot contact force, slip, clearance, progress, cross-track error, and fall checks",
        ],
        "limitations": [
            "three representative maps only; this is not 60-map dynamics acceptance",
            "route geometry is a privileged diagnostic command source",
            "no camera, tracking, target-following, or task termination is exercised",
        ],
    }
    output = result_dir / "representative_dynamics_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if report["all_passed"] else "FAIL"
    print(f"REPRESENTATIVE_GO2_DYNAMICS_BATCH {status} maps={len(rows)} {output}")
    if failures:
        for failure in failures:
            print("DYNAMICS_FAILURE", failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
