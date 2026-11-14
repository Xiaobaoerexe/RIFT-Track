#!/usr/bin/env python3
"""Generate a non-overwriting high-quality offline RGB-D set for three baselines.

Run this script with the Isaac Lab Python environment. It starts one fresh
Isaac process per map because SimulationContext is a process singleton.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAPS = ("city_01", "park_01", "forest_trail_01")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-profile", default="sim2real_v1")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--route", type=int, default=1)
    parser.add_argument("--scenario-id", default="short_none")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs/offline_high_quality_rgbd",
    )
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root / f"{args.training_profile}_seed{args.seed}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, map_id in enumerate(MAPS):
        map_root = run_root / map_id
        report = map_root / "report.json"
        capture = map_root / "frames"
        command = [
            sys.executable, str(ROOT / "scripts/run_sim2real_training.py"),
            "--training-profile", args.training_profile,
            "--seed", str(args.seed + index), "--map", map_id,
            "--route", str(args.route), "--scenario-id", args.scenario_id,
            "--duration-s", str(args.duration_s), "--offline-high-quality",
            "--capture-every-camera", "--command", "0.6", "0.0", "0.0",
            "--capture-dir", str(capture), "--output", str(report),
            "--device", args.device,
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        row = {
            "map_id": map_id, "seed": args.seed + index,
            "return_code": completed.returncode,
            "report": str(report), "capture_dir": str(capture),
        }
        if completed.returncode == 0 and report.is_file():
            payload = json.loads(report.read_text())
            row["camera_frames_saved"] = payload.get("camera_frames_saved")
            row["renderer_settings"] = payload.get("renderer_settings")
            row["reset_diagnostics"] = payload.get("reset_diagnostics")
        rows.append(row)
        if completed.returncode != 0:
            break
    manifest = {
        "schema_version": 1,
        "purpose": "offline_high_quality_rgbd_training_augmentation",
        "non_overwriting_run_root": str(run_root),
        "profile": args.training_profile,
        "maps": rows,
        "all_completed": len(rows) == len(MAPS) and all(row["return_code"] == 0 for row in rows),
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("OFFLINE_RGBD_MANIFEST", manifest_path)
    if not manifest["all_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
