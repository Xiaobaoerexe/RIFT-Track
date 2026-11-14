#!/usr/bin/env python3
"""Validate the frozen 305 m showcase artifacts and measured result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(manifest: dict) -> list[str]:
    errors: list[str] = []
    result_path = ROOT / "outputs/baseline_305m/closed_loop/baseline_305m_official.json"
    result = json.loads(result_path.read_text())
    expected = manifest["measured_result"]
    for key, value in expected.items():
        if result.get(key) != value:
            errors.append(f"result mismatch for {key}: {result.get(key)!r} != {value!r}")
    acceptance = manifest["acceptance"]
    checks = {
        "state": result["state"] == acceptance["state"],
        "minimum_route_distance_m": result["route_distance_completed_m"]
        >= acceptance["minimum_route_distance_m"],
        "minimum_tracking_coverage": result["tracking_coverage"]
        >= acceptance["minimum_tracking_coverage"],
        "minimum_reacquisition_rate": result["reacquisition_rate"]
        >= acceptance["minimum_reacquisition_rate"],
        "maximum_collision_count": result["collision_count"]
        <= acceptance["maximum_collision_count"],
        "maximum_fall_count": result["fall_count"] <= acceptance["maximum_fall_count"],
        "maximum_boundary_violation_count": result["boundary_violation_count"]
        <= acceptance["maximum_boundary_violation_count"],
    }
    errors.extend(f"acceptance check failed: {name}" for name, passed in checks.items() if not passed)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-large-media",
        action="store_true",
        help="Skip hashing the 531 MB MP4 while still validating all other artifacts.",
    )
    args = parser.parse_args()
    manifest_path = ROOT / "configs/baseline_305m_v1.json"
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    checked = 0
    for artifact in manifest["artifacts"]:
        if args.skip_large_media and artifact.get("large_media", False):
            continue
        path = ROOT / artifact["path"]
        if not path.is_file():
            errors.append(f"missing artifact: {artifact['path']}")
            continue
        actual = sha256(path)
        if actual != artifact["sha256"]:
            errors.append(
                f"sha256 mismatch: {artifact['path']} ({actual} != {artifact['sha256']})"
            )
        checked += 1
    errors.extend(validate_result(manifest))
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"PASS {manifest['baseline_id']}: {checked} artifacts and measured acceptance metrics validated"
    )


if __name__ == "__main__":
    main()
