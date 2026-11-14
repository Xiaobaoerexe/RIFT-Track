#!/usr/bin/env python3
"""Validate terrain dataset v2 geometry, physics metadata, and route quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import atan, degrees, hypot
from pathlib import Path
import statistics
import sys
from typing import Any

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from isaac_track_lab.task.terrain_routes import terrain_height


DEFAULT_CONFIG = ROOT / "configs/terrain_dataset_v2_catalog.json"
DEFAULT_ROOT = ROOT / "assets/scenes/terrain_dataset_v2"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return hypot(px - (ax + t * dx), py - (ay + t * dy))


def route_signature(route: dict[str, Any]) -> str:
    points = route["points_m"]
    stride = max(1, len(points) // 24)
    compact = [(round(p[0] / 2.0), round(p[1] / 2.0)) for p in points[::stride]]
    return hashlib.sha256(repr(compact).encode()).hexdigest()[:16]


def terrain_metrics(map_config: dict[str, Any]) -> dict[str, float]:
    xmin, xmax, ymin, ymax = map(float, map_config["bounds_m"])
    step = 3.0
    heights, slopes, curvature = [], [], []
    y = ymin
    while y <= ymax + 1e-6:
        x = xmin
        while x <= xmax + 1e-6:
            h = terrain_height(map_config, x, y)
            hx0 = terrain_height(map_config, x - 1.0, y)
            hx1 = terrain_height(map_config, x + 1.0, y)
            hy0 = terrain_height(map_config, x, y - 1.0)
            hy1 = terrain_height(map_config, x, y + 1.0)
            grade = hypot((hx1 - hx0) * 0.5, (hy1 - hy0) * 0.5)
            heights.append(h)
            slopes.append(degrees(atan(grade)))
            curvature.append(abs(hx0 + hx1 + hy0 + hy1 - 4.0 * h))
            x += step
        y += step
    return {
        "height_range_m": max(heights) - min(heights),
        "slope_p50_deg": percentile(slopes, 0.50),
        "slope_p95_deg": percentile(slopes, 0.95),
        "slope_max_deg": max(slopes),
        "roughness_p95_m": percentile(curvature, 0.95),
    }


def inside_zone(x: float, y: float, zone: dict[str, Any], margin: float) -> bool:
    cx, cy = map(float, zone["center_m"])
    width, height = map(float, zone["size_m"])
    return abs(x - cx) <= width * 0.5 + margin and abs(y - cy) <= height * 0.5 + margin


def validate_map(config: dict[str, Any], map_config: dict[str, Any], root: Path) -> dict[str, Any]:
    map_id = map_config["id"]
    map_dir = root / map_id
    errors: list[str] = []
    manifest_path = map_dir / "scene_manifest.json"
    route_path = map_dir / "routes/generated_routes.json"
    if not manifest_path.exists() or not route_path.exists():
        return {"map_id": map_id, "family": map_config["family"], "errors": ["missing manifest or route dataset"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = json.loads(route_path.read_text(encoding="utf-8"))
    usd_path = map_dir / manifest["usd"]
    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadNone)
    if stage is None:
        errors.append("USD cannot be opened")
        collision_count = 0
    else:
        collision_count = sum(1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI))
        if collision_count < int(manifest["collision_prims_expected"]):
            errors.append(f"collision prims {collision_count} < expected {manifest['collision_prims_expected']}")
        terrain = stage.GetPrimAtPath("/OutdoorMap/Terrain/Heightfield")
        if not terrain or terrain.HasAPI(UsdPhysics.CollisionAPI):
            errors.append("visual terrain must exist without a monolithic collision shape")
        collision_root = stage.GetPrimAtPath("/OutdoorMap/Collision/TerrainTiles")
        collision_tiles = [
            prim for prim in Usd.PrimRange(collision_root)
            if prim != collision_root and prim.IsA(UsdGeom.Mesh)
        ] if collision_root else []
        expected_tiles = int(manifest.get("terrain_collision_tile_count", 0))
        if not collision_tiles or len(collision_tiles) != expected_tiles:
            errors.append("terrain collision tile count differs from manifest")
        for tile in collision_tiles:
            if not tile.HasAPI(UsdPhysics.CollisionAPI):
                errors.append("terrain collision tile is missing CollisionAPI")
                break
            if UsdPhysics.MeshCollisionAPI(tile).GetApproximationAttr().Get() != UsdPhysics.Tokens.none:
                errors.append("terrain tile must use triangle-mesh approximation")
                break
            face_counts = UsdGeom.Mesh(tile).GetFaceVertexCountsAttr().Get() or []
            if not face_counts or any(int(count) != 3 for count in face_counts):
                errors.append("terrain collision tile must contain explicit triangles")
                break
            bindings = UsdShade.MaterialBindingAPI(tile).GetDirectBindingRel("physics").GetTargets()
            if not bindings:
                errors.append("terrain collision tile physics material binding missing")
                break
        if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
            errors.append("stage metersPerUnit is not 1")

    expected_routes = int(config.get("routes_per_map", 20))
    routes = dataset.get("routes", [])
    if len(routes) != expected_routes:
        errors.append(f"route count {len(routes)} != {expected_routes}")
    defaults = config["route_defaults"]
    min_length = float(defaults["minimum_length_m"])
    max_length = float(defaults["maximum_length_m"])
    clearance = float(defaults["robot_clearance_m"])
    target_slope = float(map_config.get("target_maximum_slope_deg", defaults["target_maximum_slope_deg"]))
    obstacles = manifest["obstacles"]
    signatures, coverage = [], set()
    route_lengths, route_slopes, route_clearances = [], [], []
    for index, route in enumerate(routes):
        prefix = f"route {index:02d}"
        length = float(route["length_m"])
        route_lengths.append(length)
        route_slopes.append(float(route["maximum_slope_deg"]))
        route_clearances.append(float(route["minimum_obstacle_clearance_m"]))
        if not min_length <= length <= max_length:
            errors.append(f"{prefix}: length out of range")
        if not bool(route.get("target_reachable")):
            errors.append(f"{prefix}: target_reachable false")
        if float(route["maximum_slope_deg"]) > target_slope + 0.05:
            errors.append(f"{prefix}: slope exceeds target limit")
        signatures.append(route_signature(route))
        points = route["points_m"]
        for point in points:
            x, y, z = map(float, point)
            coverage.add((int((x + 90.0) // 10.0), int((y + 90.0) // 10.0)))
            if abs(z - terrain_height(map_config, x, y)) > 0.02:
                errors.append(f"{prefix}: point height does not match USD analytic terrain")
                break
            for zone in map_config.get("challenge_zones", []):
                if not zone["traversable"] and inside_zone(x, y, zone, clearance):
                    errors.append(f"{prefix}: enters non-traversable zone {zone['id']}")
                    break
        for a, b in zip(points, points[1:]):
            ax, ay = map(float, a[:2]); bx, by = map(float, b[:2])
            if hypot(bx - ax, by - ay) > float(defaults["resample_spacing_m"]) * 1.08:
                errors.append(f"{prefix}: discontinuous point spacing")
                break
            for obstacle in obstacles:
                distance = point_segment_distance(
                    float(obstacle["x_m"]), float(obstacle["y_m"]), ax, ay, bx, by
                ) - float(obstacle["radius_m"])
                if distance < clearance - 0.02:
                    errors.append(f"{prefix}: obstacle clearance violation {obstacle['id']}")
                    break
            else:
                continue
            break
    if len(set(signatures)) != len(signatures):
        errors.append("duplicate route geometry")
    metrics = terrain_metrics(map_config)
    family = map_config["family"]
    minimum_ranges = {"city": 2.0, "park": 4.0, "forest_trail": 5.0}
    minimum_p95_slopes = {"city": 2.0, "park": 4.0, "forest_trail": 5.5}
    if metrics["height_range_m"] < minimum_ranges[family]:
        errors.append("terrain height range below family quality threshold")
    if metrics["slope_p95_deg"] < minimum_p95_slopes[family]:
        errors.append("terrain p95 slope below family quality threshold")
    return {
        "map_id": map_id, "family": map_config["family"], "errors": errors,
        "passed": not errors, "usd": str(usd_path.relative_to(ROOT)),
        "collision_prims": collision_count, "obstacle_count": len(obstacles),
        "structure_count": int(manifest.get("structure_count", 0)),
        "route_count": len(routes), "unique_route_count": len(set(signatures)),
        "route_length_min_m": min(route_lengths, default=0.0),
        "route_length_mean_m": statistics.fmean(route_lengths) if route_lengths else 0.0,
        "route_length_max_m": max(route_lengths, default=0.0),
        "route_slope_max_deg": max(route_slopes, default=0.0),
        "route_clearance_min_m": min(route_clearances, default=0.0),
        "route_coverage_10m_cells": len(coverage), **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    config.setdefault("routes_per_map", 20)
    root = args.scene_root.resolve()
    results = [validate_map(config, item, root) for item in config["maps"]]
    family_counts = {
        family: sum(item["family"] == family for item in results)
        for family in ("city", "park", "forest_trail")
    }
    total_routes = sum(item.get("route_count", 0) for item in results)
    failed = [item for item in results if item.get("errors")]
    report = {
        "schema_version": 1, "dataset_id": config["batch_id"],
        "passed": not failed and len(results) == 60 and total_routes == 1200,
        "map_count": len(results), "family_counts": family_counts,
        "route_count": total_routes, "failed_map_count": len(failed), "maps": results,
    }
    output_dir = ROOT / "outputs/terrain_dataset_v2/quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quality_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [key for key in results[0] if key != "errors"] + ["errors"]
    with (output_dir / "quality_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({**item, "errors": " | ".join(item["errors"])})
    print("TERRAIN_DATASET_QUALITY", "PASS" if report["passed"] else "FAIL",
          f"maps={len(results)}", f"routes={total_routes}", f"failed={len(failed)}", json_path)
    if failed:
        for item in failed[:10]:
            print("QUALITY_ERROR", item["map_id"], "; ".join(item["errors"][:5]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
