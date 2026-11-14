#!/usr/bin/env python3
"""Validate the static all-terrain training baseline and target motion contract."""

from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path
import sys
from typing import Any

from pxr import Usd, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from isaac_track_lab.task import TerrainTargetFollower, terrain_height, terrain_mesh_height


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def close(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def validate_d455(contract: dict[str, Any], errors: list[str], warnings: list[str]) -> dict:
    calibration = json.loads(resolve(contract["d455_calibration"]).read_text())
    required = contract["d455_required"]
    intrinsics = calibration["color_aligned_intrinsics"]
    mount = calibration["mount"]
    if list(calibration["image_size"]) != list(required["image_size"]):
        errors.append("D455 image size differs from the frozen contract")
    for key in ("fx", "fy", "cx", "cy"):
        if not close(intrinsics[key], required[key]):
            errors.append(f"D455 {key} differs from the frozen contract")
    for key in ("optical_center_height_m", "pitch_down_deg"):
        if not close(mount[key], required[key]):
            errors.append(f"D455 {key} differs from the frozen contract")
    if not str(calibration.get("status", "")).startswith("CALIBRATED"):
        warnings.append(
            "D455 extrinsics remain partially measured; x translation and full rig "
            "calibration must not be reported as measured ground truth"
        )
    return {
        "status": calibration.get("status"),
        "image_size": calibration["image_size"],
        "intrinsics": {key: intrinsics[key] for key in ("fx", "fy", "cx", "cy")},
        "mount": {
            "optical_center_height_m": mount["optical_center_height_m"],
            "pitch_down_deg": mount["pitch_down_deg"],
            "body_relative_translation_m": mount["body_relative_translation_m"],
        },
    }


def validate_map_physics(
    scene_root: Path, map_config: dict[str, Any], physics: dict[str, Any], errors: list[str]
) -> dict:
    map_dir = scene_root / map_config["id"]
    manifest = json.loads((map_dir / "scene_manifest.json").read_text())
    stage = Usd.Stage.Open(str(map_dir / manifest["usd"]), load=Usd.Stage.LoadNone)
    if not stage:
        errors.append(f"{map_config['id']}: USD cannot be opened")
        return {"map_id": map_config["id"], "collision_count": 0}
    collision_count = sum(
        1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    expected = int(manifest["collision_prims_expected"])
    if physics["collision_required"] and collision_count != expected:
        errors.append(
            f"{map_config['id']}: collision count {collision_count} != {expected}"
        )
    friction_pairs = [
        (float(manifest["friction"]["static"]), float(manifest["friction"]["dynamic"]))
    ]
    friction_pairs.extend(
        (float(zone["static_friction"]), float(zone["dynamic_friction"]))
        for zone in manifest.get("challenge_zones", [])
        if "static_friction" in zone and "dynamic_friction" in zone
    )
    for static, dynamic in friction_pairs:
        if static < float(physics["minimum_static_friction"]):
            errors.append(f"{map_config['id']}: static friction {static} is below contract")
        if dynamic < float(physics["minimum_dynamic_friction"]):
            errors.append(f"{map_config['id']}: dynamic friction {dynamic} is below contract")
        if dynamic > static:
            errors.append(f"{map_config['id']}: dynamic friction exceeds static friction")
    return {
        "map_id": map_config["id"],
        "collision_count": collision_count,
        "collision_count_expected": expected,
        "friction_pairs": friction_pairs,
    }


def validate_motion_smoke(
    motion_contract: dict[str, Any], map_config: dict[str, Any], route: dict[str, Any],
    errors: list[str],
) -> dict:
    follower = TerrainTargetFollower(
        map_config,
        route["points_m"],
        nominal_speed_mps=motion_contract["nominal_speed_mps"],
        minimum_speed_mps=motion_contract["minimum_speed_mps"],
        maximum_speed_mps=motion_contract["maximum_speed_mps"],
        heading_lookahead_m=motion_contract["heading_lookahead_m"],
        maximum_yaw_rate_deg_s=motion_contract["maximum_yaw_rate_deg_s"],
        ground_clearance_m=motion_contract["ground_clearance_m"],
    )
    dt_s = 1.0 / 30.0
    moving_speeds: list[float] = []
    yaw_rates: list[float] = []
    maximum_height_error = 0.0
    previous_yaw = follower.yaw_deg
    maximum_steps = int(follower.total_length_m / motion_contract["minimum_speed_mps"] / dt_s) + 2
    for _ in range(maximum_steps):
        pose = follower.step(dt_s)
        x_m, y_m, z_m = pose.position_m
        expected_z = terrain_mesh_height(map_config, x_m, y_m) + motion_contract["ground_clearance_m"]
        maximum_height_error = max(maximum_height_error, abs(z_m - expected_z))
        yaw_delta = abs((pose.yaw_deg - previous_yaw + 180.0) % 360.0 - 180.0)
        yaw_rates.append(yaw_delta / dt_s)
        previous_yaw = pose.yaw_deg
        if not pose.completed:
            moving_speeds.append(pose.speed_mps)
        else:
            break
    else:
        errors.append(f"{map_config['id']}: target motion did not reach route end")
    minimum = min(moving_speeds, default=0.0)
    maximum = max(moving_speeds, default=0.0)
    if minimum < motion_contract["minimum_speed_mps"] - 1e-6:
        errors.append(f"{map_config['id']}: target motion is slower than its contract")
    if maximum > motion_contract["maximum_speed_mps"] + 1e-6:
        errors.append(f"{map_config['id']}: target motion is faster than its contract")
    if max(yaw_rates, default=0.0) > motion_contract["maximum_yaw_rate_deg_s"] + 1e-6:
        errors.append(f"{map_config['id']}: target yaw rate exceeds its contract")
    return {
        "map_id": map_config["id"],
        "route_index": 0,
        "route_length_m": route["length_m"],
        "moving_speed_range_mps": [minimum, maximum],
        "maximum_yaw_rate_deg_s": max(yaw_rates, default=0.0),
        "maximum_grounding_error_m": maximum_height_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path, default=ROOT / "configs/static_training_baseline_v1.json"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/terrain_dataset_v3_realistic/static_baseline_report.json",
    )
    args = parser.parse_args()
    contract = json.loads(args.contract.resolve().read_text())
    catalog = json.loads(resolve(contract["dataset_config"]).read_text())
    scene_root = resolve(contract["scene_root"])
    errors: list[str] = []
    warnings: list[str] = []

    authorization = contract["authorized_scope"]
    if authorization["scene_expansion"] or authorization["dynamic_events"]:
        errors.append("static baseline must not expand scenes or inject events")
    if int(authorization["distractor_people"]) != 0:
        errors.append("static baseline must not contain distractor people")

    d455 = validate_d455(contract, errors, warnings)
    map_reports = [
        validate_map_physics(scene_root, item, contract["physics"], errors)
        for item in catalog["maps"]
    ]
    route_count = 0
    maximum_height_error = 0.0
    maximum_turn = 0.0
    routes_by_map: dict[str, list[dict[str, Any]]] = {}
    step_route_count = 0
    for map_config in catalog["maps"]:
        routes = json.loads(
            (scene_root / map_config["id"] / "routes/generated_routes.json").read_text()
        )["routes"]
        routes_by_map[map_config["id"]] = routes
        route_count += len(routes)
        features = map_config.get("traversable_step_features", [])
        for route in routes:
            maximum_turn = max(maximum_turn, float(route["maximum_turn_deg"]))
            if float(route["maximum_turn_deg"]) > contract["target_motion"]["maximum_route_turn_deg"] + 1e-6:
                errors.append(f"{map_config['id']}: route exceeds maximum turn contract")
            route_crosses_step = False
            for x_m, y_m, z_m in route["points_m"]:
                maximum_height_error = max(
                    maximum_height_error,
                    abs(float(z_m) - terrain_height(map_config, float(x_m), float(y_m))),
                )
                route_crosses_step |= any(
                    hypot(float(x_m) - float(feature["center_m"][0]),
                          float(y_m) - float(feature["center_m"][1])) <= 2.0
                    for feature in features
                )
            step_route_count += int(route_crosses_step)
    if maximum_height_error > contract["acceptance"]["terrain_height_error_max_m"]:
        errors.append("authored route heights do not match analytic terrain")
    if len(catalog["maps"]) != contract["acceptance"]["map_count"]:
        errors.append("map count differs from baseline acceptance")
    if route_count != contract["acceptance"]["route_count"]:
        errors.append("route count differs from baseline acceptance")

    representative = []
    maps_by_id = {item["id"]: item for item in catalog["maps"]}
    for map_id in contract["representative_maps"]:
        representative.append(
            validate_motion_smoke(
                contract["target_motion"], maps_by_id[map_id], routes_by_map[map_id][0], errors
            )
        )
    if contract["target_motion"]["skeletal_foot_ik_status"].startswith("not_implemented"):
        warnings.append(
            "Target root motion is terrain-grounded, but skeletal foot IK is not yet "
            "accepted; do not label foot placement as physically validated"
        )

    report = {
        "schema_version": 1,
        "baseline_id": contract["baseline_id"],
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "map_count": len(catalog["maps"]),
        "route_count": route_count,
        "maximum_route_turn_deg": maximum_turn,
        "maximum_route_terrain_height_error_m": maximum_height_error,
        "routes_crossing_authored_step_features": step_route_count,
        "d455": d455,
        "representative_motion_smoke": representative,
        "map_physics": map_reports,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "STATIC_TRAINING_BASELINE", "PASS" if report["passed"] else "FAIL",
        f"maps={report['map_count']}", f"routes={route_count}",
        f"max_turn={maximum_turn:.2f}deg", args.output.resolve(),
    )
    for warning in warnings:
        print("BASELINE_WARNING", warning)
    if errors:
        for error in errors[:30]:
            print("BASELINE_ERROR", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
