#!/usr/bin/env python3
"""Validate v3 semantic layouts, USD physics, water, and logical routes."""

from __future__ import annotations

import argparse
import csv
import json
from math import hypot
from pathlib import Path
from typing import Any

from pxr import Usd, UsdPhysics

from validate_terrain_dataset_v2 import ROOT, point_segment_distance, validate_map


DEFAULT_CONFIG = ROOT / "configs/terrain_dataset_v3_realistic.json"
DEFAULT_ROOT = ROOT / "assets/scenes/terrain_dataset_v3_realistic"
DEFAULT_OUTPUT = ROOT / "outputs/terrain_dataset_v3_realistic/quality"


def obstacle_radius(item: dict[str, Any]) -> float:
    if item["proxy"] == "box":
        sx, sy = map(float, item["size_m"][:2])
        return 0.5 * hypot(sx, sy) * float(item.get("scale", 1.0))
    return float(item["radius_m"]) * float(item.get("scale", 1.0))


def inside_navigation_corridor(x_m: float, y_m: float, corridors: list[dict]) -> bool:
    for corridor in corridors:
        half_width = float(corridor["width_m"]) * 0.5 + 0.03
        for left, right in zip(corridor["points_m"], corridor["points_m"][1:]):
            if point_segment_distance(
                x_m, y_m, float(left[0]), float(left[1]), float(right[0]), float(right[1])
            ) <= half_width:
                return True
    return False


def semantic_errors(
    config: dict[str, Any], map_config: dict[str, Any], root: Path,
) -> list[str]:
    errors: list[str] = []
    family = map_config["family"]
    map_dir = root / map_config["id"]
    manifest_path = map_dir / "scene_manifest.json"
    route_path = map_dir / "routes/generated_routes.json"
    if not manifest_path.exists() or not route_path.exists():
        return ["semantic check skipped because build outputs are missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routes = json.loads(route_path.read_text(encoding="utf-8")).get("routes", [])
    stage = Usd.Stage.Open(str(map_dir / manifest["usd"]), load=Usd.Stage.LoadNone)
    corridors = map_config.get("navigation_corridors", [])
    if not corridors:
        errors.append("navigation corridors missing")
    for route_index, route in enumerate(routes):
        if any(not inside_navigation_corridor(float(x), float(y), corridors)
               for x, y, _ in route["points_m"]):
            errors.append(f"route {route_index:02d}: leaves semantic navigation corridors")
            break
        maximum_turn = float(route.get("maximum_turn_deg", 180.0))
        turn_limit = float(config["route_defaults"].get("maximum_turn_deg", 150.0))
        if maximum_turn > turn_limit + 1e-6:
            errors.append(
                f"route {route_index:02d}: instantaneous turn "
                f"{maximum_turn:.2f} deg exceeds {turn_limit:.2f} deg"
            )
            break

    instances = map_config.get("obstacle_instances", [])
    for index, left in enumerate(instances):
        left_radius = obstacle_radius(left)
        for right in instances[index + 1:]:
            separation = hypot(
                float(left["x_m"]) - float(right["x_m"]),
                float(left["y_m"]) - float(right["y_m"]),
            )
            if separation < left_radius + obstacle_radius(right) + 0.20:
                errors.append(f"planned props overlap: {left['id']} / {right['id']}")
                break
        if errors and errors[-1].startswith("planned props overlap"):
            break

    if stage:
        actual_collisions = sum(1 for prim in stage.Traverse()
                                if prim.HasAPI(UsdPhysics.CollisionAPI))
        if actual_collisions != int(manifest["collision_prims_expected"]):
            errors.append("collision prim count differs from exact manifest contract")

    surfaces = map_config.get("surface_corridors", [])
    steps = map_config.get("traversable_step_features", [])
    if stage:
        authored_surfaces = [
            prim for prim in stage.Traverse()
            if prim.GetAttribute("track:role").Get() == "semantic_surface_material_region"
        ]
        terrain = stage.GetPrimAtPath("/OutdoorMap/Terrain/Heightfield")
        geometry_mode = terrain.GetAttribute("surface:geometryMode").Get() if terrain else None
        corridor_count = terrain.GetAttribute("surface:corridorCount").Get() if terrain else None
        if not authored_surfaces or geometry_mode != "terrain_face_material_subsets":
            errors.append("semantic surfaces are not bound to the terrain face topology")
        if corridor_count != len(surfaces):
            errors.append("terrain material-subset corridor count is incorrect")
        for prim in authored_surfaces:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                errors.append(f"visual surface must not create a double collider: {prim.GetPath()}")
                break
            collision_source = prim.GetAttribute("surface:collisionSource").Get()
            if collision_source != "/OutdoorMap/Collision/TerrainTiles":
                errors.append(f"visual surface collision source is not declared: {prim.GetPath()}")
                break
    if family == "city":
        roads = [item for item in surfaces if item.get("kind") == "road"]
        if len(roads) < 6:
            errors.append("city requires at least six connected road corridors")
        if int(manifest.get("curb_count", 0)) < 40:
            errors.append("city curb network is incomplete")
        if int(manifest.get("structure_count", 0)) < 12:
            errors.append("city block zoning has too few structures")
    elif family == "park":
        feature_kinds = {item["kind"] for item in map_config["terrain"].get("features", [])}
        lake_zones = [item for item in map_config.get("challenge_zones", [])
                      if item["kind"] == "water" and item.get("shape") == "ellipse"]
        if "lake_basin" not in feature_kinds or not lake_zones:
            errors.append("park requires an elliptical water surface in a terrain bowl")
        if sum(item["kind"] == "gaussian" for item in map_config["terrain"].get("features", [])) < 2:
            errors.append("park requires at least two authored hills")
        if len(surfaces) < 3 or len(steps) < 3:
            errors.append("park path or embedded-stone contract is incomplete")
        if stage and lake_zones:
            water = stage.GetPrimAtPath(f"/OutdoorMap/ChallengeZones/{lake_zones[0]['id']}")
            clearance = water.GetAttribute("water:shore_clearance_m").Get() if water else None
            if clearance is None or float(clearance) < 0.04:
                errors.append("lake water clips or is too close to the shoreline terrain")
    elif family == "forest_trail":
        tree_count = sum(item["asset"] == "pine_tree_01" for item in instances)
        rock_count = sum(item["asset"] in ("boulder_01", "rock_09") for item in instances)
        deadwood_count = sum(item["asset"] in ("dead_tree_trunk", "tree_stump_01")
                             for item in instances)
        if tree_count < 90 or rock_count < 20 or deadwood_count < 14:
            errors.append("forest density contract is not met")
        if len(steps) < 5 or int(manifest.get("traversable_feature_count", 0)) < 9:
            errors.append("forest traversable step-feature contract is not met")
        if len(surfaces) < 3:
            errors.append("forest connected trail network is incomplete")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    root = args.scene_root.resolve()
    results = []
    for map_config in config["maps"]:
        result = validate_map(config, map_config, root)
        # v2 used one conservative terrain contract.  v3 defines family-specific
        # realism targets: urban roads may have modest total relief while still
        # requiring a non-flat P95 grade; park and forest retain larger relief.
        result["errors"] = [
            error for error in result["errors"]
            if error not in {
                "terrain height range below family quality threshold",
                "terrain p95 slope below family quality threshold",
            }
        ]
        minimums = {
            "city": (1.8, 2.0), "park": (4.0, 4.0), "forest_trail": (5.0, 5.5),
        }
        minimum_height, minimum_slope = minimums[map_config["family"]]
        if float(result["height_range_m"]) < minimum_height:
            result["errors"].append("terrain height range below v3 family contract")
        if float(result["slope_p95_deg"]) < minimum_slope:
            result["errors"].append("terrain p95 slope below v3 family contract")
        result["errors"].extend(semantic_errors(config, map_config, root))
        result["passed"] = not result["errors"]
        results.append(result)
    family_counts = {
        family: sum(item["family"] == family for item in results)
        for family in ("city", "park", "forest_trail")
    }
    route_count = sum(item.get("route_count", 0) for item in results)
    failed = [item for item in results if item["errors"]]
    report = {
        "schema_version": 1, "dataset_id": config["batch_id"],
        "passed": not failed and len(results) == 60 and route_count == 1200,
        "map_count": len(results), "family_counts": family_counts,
        "route_count": route_count, "failed_map_count": len(failed), "maps": results,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quality_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [key for key in results[0] if key != "errors"] + ["errors"]
    with (output_dir / "quality_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({**item, "errors": " | ".join(item["errors"])})
    print(
        "REALISTIC_TERRAIN_QUALITY", "PASS" if report["passed"] else "FAIL",
        f"maps={len(results)}", f"routes={route_count}", f"failed={len(failed)}", json_path,
    )
    if failed:
        for item in failed[:12]:
            print("QUALITY_ERROR", item["map_id"], "; ".join(item["errors"][:6]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
