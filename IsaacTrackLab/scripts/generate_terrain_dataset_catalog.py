#!/usr/bin/env python3
"""Expand three family templates into 60 deterministic rugged map configurations."""

from __future__ import annotations

import json
from math import hypot
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/terrain_dataset_v2.json"

BASE_GATE_SETS = (
    ((-72, -60), (-34, 38), (31, -38), (70, 60)),
    ((-70, 56), (-30, -38), (32, 40), (70, -56)),
    ((-72, -48), (4, -8), (-34, 58), (68, 54)),
    ((-72, 2), (-18, 61), (39, -53), (72, 8)),
    ((-65, -65), (48, -28), (-42, 28), (64, 66)),
)


def jittered_gate_sets(rng: random.Random) -> list[list[list[float]]]:
    output = []
    for pattern in BASE_GATE_SETS:
        gates = []
        for x_m, y_m in pattern:
            gates.append([
                round(x_m + rng.uniform(-5.0, 5.0), 3),
                round(y_m + rng.uniform(-5.0, 5.0), 3),
                round(rng.uniform(18.0, 27.0), 3),
                round(rng.uniform(18.0, 27.0), 3),
            ])
        output.append(gates)
    return output


def terrain_features(rng: random.Random, family: str) -> list[dict]:
    ranges = {
        "city": (0.45, 0.9, 15.0, 24.0),
        "park": (0.8, 1.5, 13.0, 23.0),
        "forest_trail": (1.1, 2.1, 11.0, 21.0),
    }
    low, high, sigma_low, sigma_high = ranges[family]
    features = []
    for index, kind in enumerate(("gaussian", "gaussian", "ridge", "rut")):
        amplitude = rng.uniform(low, high)
        if index == 1:
            amplitude *= -1.0
        features.append({
            "id": f"{kind}_{index}", "kind": kind,
            "center_m": [round(rng.uniform(-55, 55), 3), round(rng.uniform(-55, 55), 3)],
            "amplitude_m": round(amplitude, 4), "sigma_m": round(rng.uniform(sigma_low, sigma_high), 3),
            "angle_deg": round(rng.uniform(-180, 180), 3),
            "wavelength_m": round(rng.uniform(13.0, 24.0), 3),
        })
    return features


def zones(rng: random.Random, family: str) -> list[dict]:
    if family == "city":
        return [
            {"id": "broken_road", "kind": "rough_ground", "center_m": [rng.uniform(-30, 15), rng.uniform(5, 42)], "size_m": [rng.uniform(24, 38), rng.uniform(18, 28)], "cost": 1.8, "traversable": True, "static_friction": 0.68, "dynamic_friction": 0.50},
            {"id": "construction_rut", "kind": "gully", "center_m": [rng.uniform(28, 50), rng.uniform(-42, -12)], "size_m": [rng.uniform(18, 26), rng.uniform(35, 55)], "cost": 2.1, "traversable": True, "static_friction": 0.62, "dynamic_friction": 0.44},
        ]
    if family == "park":
        pond_x, pond_y = rng.choice(((-28, 5), (30, -8), (5, 28)))
        return [
            {"id": "pond", "kind": "water", "center_m": [pond_x+rng.uniform(-5,5), pond_y+rng.uniform(-5,5)], "size_m": [rng.uniform(18, 30), rng.uniform(28, 48)], "cost": 99.0, "traversable": False},
            {"id": "wet_grass", "kind": "soft_ground", "center_m": [rng.uniform(-48,48), rng.uniform(-48,48)], "size_m": [rng.uniform(28,44), rng.uniform(22,38)], "cost": 1.7, "traversable": True, "static_friction": 0.58, "dynamic_friction": 0.40},
            {"id": "gravel_climb", "kind": "slope", "center_m": [rng.uniform(-45,45), rng.uniform(-45,45)], "size_m": [rng.uniform(24,38), rng.uniform(20,34)], "cost": 1.5, "traversable": True, "static_friction": 0.78, "dynamic_friction": 0.62},
        ]
    return [
        {"id": "mud_hollow", "kind": "soft_ground", "center_m": [rng.uniform(-45,35), rng.uniform(-35,45)], "size_m": [rng.uniform(28,45), rng.uniform(20,34)], "cost": 2.0, "traversable": True, "static_friction": 0.52, "dynamic_friction": 0.34},
        {"id": "root_ridge", "kind": "rough_ground", "center_m": [rng.uniform(-35,45), rng.uniform(-45,35)], "size_m": [rng.uniform(25,40), rng.uniform(22,36)], "cost": 1.9, "traversable": True, "static_friction": 0.74, "dynamic_friction": 0.56},
    ]


def obstacle_templates(rng: random.Random, family: str) -> list[dict]:
    if family == "city":
        return [
            {"asset": "concrete_road_barrier", "count": rng.randint(10, 16), "scale": [0.8, 1.2], "proxy": "box", "size_m": [2.5, 0.75, 1.1], "clearance_m": 1.2},
            {"asset": "covered_car", "count": rng.randint(7, 11), "scale": [0.85, 1.1], "proxy": "box", "size_m": [2.1, 4.7, 1.6], "clearance_m": 1.4},
            {"asset": "planter_box_01", "count": rng.randint(8, 13), "scale": [0.8, 1.25], "proxy": "box", "size_m": [1.5, 0.9, 0.8], "clearance_m": 1.2},
            {"asset": "metal_trash_can", "count": rng.randint(8, 14), "scale": [0.8, 1.15], "proxy": "cylinder", "radius_m": 0.45, "height_m": 1.0, "clearance_m": 1.1},
        ]
    if family == "park":
        return [
            {"asset": "pine_tree_01", "count": rng.randint(30, 44), "scale": [0.65, 1.05], "proxy": "cylinder", "radius_m": 0.5, "height_m": 6.5, "clearance_m": 1.4},
            {"asset": "painted_wooden_bench", "count": rng.randint(8, 13), "scale": [0.85, 1.15], "proxy": "box", "size_m": [1.9, 0.75, 1.2], "clearance_m": 1.2},
            {"asset": "boulder_01", "count": rng.randint(10, 16), "scale": [0.45, 0.9], "proxy": "sphere", "radius_m": 0.9, "clearance_m": 1.2},
            {"asset": "tree_stump_01", "count": rng.randint(7, 12), "scale": [0.65, 1.0], "proxy": "cylinder", "radius_m": 0.6, "height_m": 0.75, "clearance_m": 1.2},
        ]
    return [
        {"asset": "pine_tree_01", "count": rng.randint(48, 65), "scale": [0.72, 1.15], "proxy": "cylinder", "radius_m": 0.52, "height_m": 7.0, "clearance_m": 1.5},
        {"asset": "dead_tree_trunk", "count": rng.randint(12, 19), "scale": [0.75, 1.25], "proxy": "box", "size_m": [4.2, 0.9, 0.9], "clearance_m": 1.3},
        {"asset": "tree_stump_01", "count": rng.randint(12, 20), "scale": [0.75, 1.2], "proxy": "cylinder", "radius_m": 0.65, "height_m": 0.8, "clearance_m": 1.3},
        {"asset": "boulder_01", "count": rng.randint(14, 22), "scale": [0.55, 1.15], "proxy": "sphere", "radius_m": 1.0, "clearance_m": 1.4},
    ]


def city_structures(rng: random.Random, gate_sets: list[list[list[float]]]) -> list[dict]:
    gates = [(gate[0], gate[1]) for gate_set in gate_sets for gate in gate_set]
    candidates = [(x, y) for x in (-60,-30,0,30,60) for y in (-60,-30,0,30,60)]
    rng.shuffle(candidates)
    structures = []
    target_count = rng.randint(11, 15)
    for x_m, y_m in candidates:
        if len(structures) >= target_count:
            break
        x_m += rng.uniform(-5,5); y_m += rng.uniform(-5,5)
        if min(hypot(x_m-gx, y_m-gy) for gx,gy in gates) < 18.0:
            continue
        width, depth, height = rng.uniform(9,16), rng.uniform(8,15), rng.uniform(7,18)
        structures.append({
            "id": f"building_{len(structures):02d}", "kind": "building",
            "center_m": [round(x_m,3), round(y_m,3)],
            "size_m": [round(width,3), round(depth,3), round(height,3)],
            "material": rng.choice(("brick_wall_001", "beige_wall_001")),
            "yaw_deg": rng.choice((0.0, 90.0)),
        })
    return structures


def rounded_zones(items: list[dict]) -> list[dict]:
    for item in items:
        item["center_m"] = [round(float(v), 3) for v in item["center_m"]]
        item["size_m"] = [round(float(v), 3) for v in item["size_m"]]
    return items


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    maps = []
    family_seed = {"city": 71_000, "park": 82_000, "forest_trail": 93_000}
    for family, template in config["families"].items():
        for index in range(1, int(config["maps_per_family"]) + 1):
            seed = family_seed[family] + index * 97
            rng = random.Random(seed)
            gate_sets = jittered_gate_sets(rng)
            static_low, static_high = template["static_friction"]
            dynamic_low, dynamic_high = template["dynamic_friction"]
            amplitude_low, amplitude_high = template["amplitude_m"]
            rough_low, rough_high = template["roughness_m"]
            map_config = {
                "id": f"{family}_{index:02d}", "family": family,
                "label": f"{template['label']} {index:02d}", "seed": seed,
                "bounds_m": [-90.0, 90.0, -90.0, 90.0],
                "target_maximum_slope_deg": template["target_maximum_slope_deg"],
                "terrain": {
                    "profile": template["profile"], "texture": template["texture"],
                    "hdri": template["hdri"], "seed": seed,
                    "amplitude_m": round(rng.uniform(amplitude_low, amplitude_high), 4),
                    "frequency": round(rng.uniform(0.026, 0.038), 5),
                    "roughness_m": round(rng.uniform(rough_low, rough_high), 4),
                    "micro_frequency": round(rng.uniform(0.085, 0.125), 5),
                    "tilt_x": round(rng.uniform(-0.012, 0.012), 5),
                    "tilt_y": round(rng.uniform(-0.012, 0.012), 5),
                    "static_friction": round(rng.uniform(static_low, static_high), 4),
                    "dynamic_friction": round(rng.uniform(dynamic_low, dynamic_high), 4),
                    "features": terrain_features(rng, family),
                },
                "obstacle_templates": obstacle_templates(rng, family),
                "challenge_zones": rounded_zones(zones(rng, family)),
                "route_gates": gate_sets[0], "route_gate_sets": gate_sets,
                "procedural_structures": city_structures(rng, gate_sets) if family == "city" else [],
            }
            maps.append(map_config)
    catalog = {
        "schema_version": 2, "batch_id": config["dataset_id"],
        "scene_output_root": config["scene_output_root"],
        "maps_per_family": config["maps_per_family"],
        "routes_per_map": config["routes_per_map"],
        "quality_contract": config["quality_contract"],
        "route_defaults": config["route_defaults"], "maps": maps,
    }
    output = ROOT / config["catalog_output"]
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TERRAIN_DATASET_CATALOG_OK", len(maps), output)


if __name__ == "__main__":
    main()
