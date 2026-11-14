#!/usr/bin/env python3
"""Validate outdoor USD maps, physics contracts, assets, and route datasets."""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path
import sys

from pxr import Usd, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from isaac_track_lab.task.terrain_routes import Obstacle2D, TerrainRouteGenerator

CONFIG = ROOT / "configs/outdoor_map_batch.json"
SCENE_ROOT = ROOT / "assets/scenes/outdoor_batch"
ASSET_ROOT = ROOT / "assets/external/polyhaven"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_map(config: dict, map_config: dict) -> dict:
    map_dir = SCENE_ROOT / map_config["id"]
    manifest_path = map_dir / "scene_manifest.json"
    routes_path = map_dir / "routes/generated_routes.json"
    require(manifest_path.exists(), f"missing {manifest_path}")
    require(routes_path.exists(), f"missing {routes_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routes = json.loads(routes_path.read_text(encoding="utf-8"))
    require(manifest["dynamic_events"] == [], "dynamic events are outside this stage")
    require(manifest["distractor_people"] == 0, "distractor people are outside this stage")
    require(manifest["obstacle_count"] > 0, "map has no static obstacles")
    require(routes["route_count"] > 0, "map has no generated routes")

    usd_path = map_dir / manifest["usd"]
    stage = Usd.Stage.Open(str(usd_path))
    require(stage is not None, f"could not open {usd_path}")
    require(stage.GetDefaultPrim().GetPath().pathString == "/OutdoorMap", "wrong default prim")
    collision_count = sum(1 for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI))
    require(collision_count >= manifest["collision_prims_expected"], (
        f"{map_config['id']} has {collision_count} collisions, expected at least "
        f"{manifest['collision_prims_expected']}"
    ))
    require(stage.GetPrimAtPath("/OutdoorMap/Terrain/Heightfield").HasAPI(UsdPhysics.CollisionAPI),
            "terrain collision missing")
    for zone in map_config["challenge_zones"]:
        if not zone["traversable"]:
            continue
        material_prim = stage.GetPrimAtPath(f"/OutdoorMap/Looks/Physics_{zone['id']}")
        subset_prim = stage.GetPrimAtPath(
            f"/OutdoorMap/Terrain/Heightfield/Physics_{zone['id']}"
        )
        require(material_prim.HasAPI(UsdPhysics.MaterialAPI), f"{zone['id']} physics material missing")
        require(subset_prim.IsValid(), f"{zone['id']} terrain material subset missing")
        api = UsdPhysics.MaterialAPI(material_prim)
        require(abs(api.GetStaticFrictionAttr().Get() - zone["static_friction"]) < 1e-6,
                f"{zone['id']} static friction mismatch")
        require(abs(api.GetDynamicFrictionAttr().Get() - zone["dynamic_friction"]) < 1e-6,
                f"{zone['id']} dynamic friction mismatch")

    texture_id = map_config["terrain"]["texture"]
    texture_dir = ASSET_ROOT / "textures" / texture_id
    require(any(texture_dir.glob("*diff*2k*.jpg")), f"{texture_id} diffuse missing")
    require(any(texture_dir.glob("*rough*2k*.jpg")), f"{texture_id} roughness missing")
    require(any(texture_dir.glob("*nor_gl*2k*.jpg")), f"{texture_id} normal missing")
    require(any((ASSET_ROOT / "hdris" / map_config["terrain"]["hdri"]).glob("*.exr")),
            "HDRI missing")

    defaults = config["route_defaults"]
    obstacles = [
        Obstacle2D(item["id"], item["x_m"], item["y_m"], item["radius_m"])
        for item in manifest["obstacles"]
    ]
    for route in routes["routes"]:
        require(defaults["minimum_length_m"] <= route["length_m"] <= defaults["maximum_length_m"],
                f"route length outside contract: {route['length_m']}")
        require(route["maximum_slope_deg"] <= defaults["maximum_slope_deg"], "route slope too high")
        require(route["minimum_obstacle_clearance_m"] >= defaults["robot_clearance_m"] - 1e-5,
                "route obstacle clearance too small")
        measured_clearance = min(
            hypot(point[0] - obstacle.x_m, point[1] - obstacle.y_m) - obstacle.radius_m
            for point in route["points_m"] for obstacle in obstacles
        )
        require(measured_clearance >= defaults["robot_clearance_m"] - 1e-5,
                "route points intersect an obstacle proxy")

    # Re-run the first seed and compare every point: route corpora must be exactly regenerable.
    first = routes["routes"][0]
    regenerated = TerrainRouteGenerator(map_config, defaults, obstacles).generate(first["seed"]).to_dict()
    regenerated_points = [list(point) for point in regenerated["points_m"]]
    require(regenerated_points == first["points_m"], "first route is not seed-reproducible")
    return {
        "map_id": map_config["id"], "collision_count": collision_count,
        "obstacle_count": manifest["obstacle_count"], "route_count": routes["route_count"],
        "length_min_m": min(item["length_m"] for item in routes["routes"]),
        "length_max_m": max(item["length_m"] for item in routes["routes"]),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    summaries = [validate_map(config, item) for item in config["maps"]]
    for item in summaries:
        print(
            "OUTDOOR_MAP_VALID", item["map_id"],
            f"collisions={item['collision_count']}", f"obstacles={item['obstacle_count']}",
            f"routes={item['route_count']}",
            f"length={item['length_min_m']:.1f}-{item['length_max_m']:.1f}m",
        )
    print("OUTDOOR_BATCH_VALIDATION_OK", len(summaries), sum(x["route_count"] for x in summaries))


if __name__ == "__main__":
    main()
