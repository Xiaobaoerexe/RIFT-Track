#!/usr/bin/env python3
"""Generate reproducible collision-aware route datasets for every outdoor map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from pxr import Gf, Sdf, Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from isaac_track_lab.task.terrain_routes import Obstacle2D, TerrainRouteGenerator

CONFIG = ROOT / "configs/outdoor_map_batch.json"
SCENE_ROOT = ROOT / "assets/scenes/outdoor_batch"


def author_preview(map_id: str, routes: list[dict], output: Path) -> None:
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/GeneratedRoutes")
    stage.SetDefaultPrim(root.GetPrim())
    palette = (
        Gf.Vec3f(0.95, 0.22, 0.12), Gf.Vec3f(0.08, 0.65, 1.0),
        Gf.Vec3f(0.95, 0.72, 0.08), Gf.Vec3f(0.28, 0.92, 0.34),
    )
    for index, route in enumerate(routes):
        points = [Gf.Vec3f(float(x), float(y), float(z) + 0.30) for x, y, z in route["points_m"]]
        curve = UsdGeom.BasisCurves.Define(stage, f"/GeneratedRoutes/Route_{index:03d}")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr(points)
        curve.CreateWidthsAttr([0.18])
        curve.SetWidthsInterpolation("constant")
        curve.CreateDisplayColorAttr([palette[index % len(palette)]])
        curve.GetPrim().CreateAttribute("route:seed", Sdf.ValueTypeNames.Int64).Set(route["seed"])
        curve.GetPrim().CreateAttribute("route:length_m", Sdf.ValueTypeNames.Double).Set(route["length_m"])
    stage.GetRootLayer().customLayerData = {
        "mapId": map_id, "routeCount": len(routes), "purpose": "trajectory dataset preview"
    }
    stage.GetRootLayer().Save()


def route_geometry_signature(route: dict) -> str:
    """Reject routes that differ only by sub-cell jitter at a gate."""
    points = route["points_m"]
    stride = max(1, len(points) // 24)
    compact = [(round(p[0] / 2.0), round(p[1] / 2.0)) for p in points[::stride]]
    return hashlib.sha256(repr(compact).encode()).hexdigest()


def generate_for_map(
    config: dict, map_config: dict, count: int, seed_offset: int, scene_root: Path = SCENE_ROOT,
) -> dict:
    map_dir = scene_root / map_config["id"]
    manifest_path = map_dir / "scene_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"build the map before generating routes: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obstacles = [
        Obstacle2D(item["id"], float(item["x_m"]), float(item["y_m"]), float(item["radius_m"]))
        for item in manifest["obstacles"]
    ]
    generator = TerrainRouteGenerator(map_config, config["route_defaults"], obstacles)
    routes = []
    signatures: set[str] = set()
    for index in range(count):
        base_seed = int(map_config["seed"]) * 100_000 + seed_offset + index
        for retry in range(240):
            seed = base_seed + retry * count
            try:
                route = generator.generate(seed).to_dict()
            except RuntimeError:
                continue
            signature = route_geometry_signature(route)
            if signature not in signatures:
                signatures.add(signature)
                routes.append(route)
                break
        else:
            raise RuntimeError(f"could not produce {count} unique routes for {map_config['id']}")
    dataset = {
        "schema_version": 1,
        "batch_id": config["batch_id"],
        "map_id": map_config["id"],
        "algorithm": config["route_defaults"]["algorithm"],
        "generator": "isaac_track_lab.task.terrain_routes.TerrainRouteGenerator",
        "seed_formula": "map_seed * 100000 + seed_offset + route_index + retry * route_count",
        "seed_offset": seed_offset,
        "route_count": len(routes),
        "route_defaults": config["route_defaults"],
        "routes": routes,
    }
    route_dir = map_dir / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)
    json_path = route_dir / "generated_routes.json"
    json_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    author_preview(map_config["id"], routes, route_dir / "generated_routes.usda")
    lengths = [route["length_m"] for route in routes]
    print(
        "OUTDOOR_ROUTES_OK", map_config["id"], len(routes),
        f"length_range={min(lengths):.2f}-{max(lengths):.2f}m", flush=True,
    )
    return {
        "map_id": map_config["id"], "route_count": len(routes),
        "minimum_length_m": min(lengths), "maximum_length_m": max(lengths),
        "json": str(json_path.relative_to(ROOT)),
        "preview_usd": str((route_dir / "generated_routes.usda").relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", action="append", dest="map_ids")
    parser.add_argument("--count", type=int, default=16, help="routes per map")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--scene-root", type=Path)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    scene_root = args.scene_root.resolve() if args.scene_root else (
        ROOT / config.get("scene_output_root", str(SCENE_ROOT.relative_to(ROOT)))
    )
    selected = [item for item in config["maps"] if not args.map_ids or item["id"] in args.map_ids]
    if args.map_ids and len(selected) != len(set(args.map_ids)):
        raise SystemExit("one or more requested map ids are unknown")
    summaries = [
        generate_for_map(config, item, args.count, args.seed_offset, scene_root)
        for item in selected
    ]
    output = scene_root / "route_dataset_summary.json"
    if args.map_ids and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        replacements = {item["map_id"]: item for item in summaries}
        summaries = [
            replacements.pop(item["map_id"], item)
            for item in existing.get("maps", [])
        ] + list(replacements.values())
    output.write_text(json.dumps({
        "schema_version": 1, "batch_id": config["batch_id"],
        "total_routes": sum(item["route_count"] for item in summaries), "maps": summaries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("OUTDOOR_ROUTE_BATCH_OK", sum(item["route_count"] for item in summaries), output, flush=True)


if __name__ == "__main__":
    main()
