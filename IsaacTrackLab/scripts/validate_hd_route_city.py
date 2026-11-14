#!/usr/bin/env python3
"""Validate the intersecting HD city, route contract, friction, and collisions."""

from __future__ import annotations

import json
from math import dist, isclose
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets/scenes/hd_route_city/hd_route_city.usda"
ROUTE_CONFIG = ROOT / "configs/hd_route_city_route.json"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def role(prim: Usd.Prim) -> str | None:
    attribute = prim.GetAttribute("track:role")
    return attribute.Get() if attribute else None


def friction(material: UsdShade.Material) -> tuple[float, float]:
    api = UsdPhysics.MaterialAPI(material.GetPrim())
    return float(api.GetStaticFrictionAttr().Get()), float(api.GetDynamicFrictionAttr().Get())


def main() -> None:
    check(SCENE.exists(), f"scene missing: {SCENE}")
    route_config = json.loads(ROUTE_CONFIG.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(SCENE))
    check(stage is not None, "USD stage could not be opened")
    check(stage.GetDefaultPrim().GetPath().pathString == "/HDStreet", "wrong default prim")
    check(UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z, "stage must be Z-up")
    check(abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) < 1e-9, "stage must use meters")

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(SCENE))
    check(not unresolved, f"unresolved USD dependencies: {unresolved}")
    prims = list(stage.Traverse())
    colliders = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    materials = [UsdShade.Material(prim) for prim in prims if prim.IsA(UsdShade.Material)]
    roles = {}
    for prim in prims:
        value = role(prim)
        if value:
            roles.setdefault(value, []).append(prim)

    waypoints = sorted(
        stage.GetPrimAtPath("/HDStreet/Route").GetChildren(),
        key=lambda prim: prim.GetName(),
    )
    expected_points = [point[:2] for point in route_config["waypoints_m"]]
    check(len(waypoints) == len(expected_points) == 27, "route must contain 27 ordered waypoints")
    actual_points = []
    for index, (waypoint, expected) in enumerate(zip(waypoints, expected_points, strict=True)):
        translation = UsdGeom.Xformable(waypoint).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        actual = (float(translation[0]), float(translation[1]))
        actual_points.append(actual)
        check(dist(actual, expected) < 1e-6, f"waypoint {index} differs from route config")
    route_length = sum(dist(a, b) for a, b in zip(actual_points, actual_points[1:]))
    route_prim = stage.GetPrimAtPath("/HDStreet/Route")
    check(300.0 <= route_length <= 310.0, f"route length must be approximately 300 m, got {route_length:.3f}")
    check(abs(float(route_prim.GetAttribute("route:length_m").Get()) - route_length) < 1e-6,
          "route length metadata does not match geometry")
    speed = float(route_prim.GetAttribute("route:target_speed_mps").Get())
    check(1.0 <= speed <= 1.5, f"target speed outside requested range: {speed}")
    check(int(route_prim.GetAttribute("route:major_turn_count").Get()) == 4, "expected four major turns")

    props = list(stage.GetPrimAtPath("/HDStreet/Props").GetChildren())
    check(len(props) == 6, f"expected six high-quality props, found {len(props)}")
    check(all(len(list(Usd.PrimRange(prop))) > 2 for prop in props), "a referenced prop did not compose")
    check(len(roles.get("drivable_surface", [])) == 6, "expected six intersecting road slabs")
    check(len(roles.get("walkable_surface", [])) == 20, "expected twenty raised city blocks")
    check(len(roles.get("obstacle_collision", [])) == 6, "every high-quality prop needs a collision proxy")
    for required_role in (
        "drivable_surface", "walkable_surface", "building_collision", "roof",
        "street_light", "tree_trunk", "tree_crown", "obstacle_collision",
    ):
        check(required_role in roles, f"missing required collision role: {required_role}")
        check(all(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in roles[required_role]),
              f"not every {required_role} prim has CollisionAPI")

    road_physics = UsdShade.Material(stage.GetPrimAtPath("/HDStreet/Looks/RoadPhysics"))
    sidewalk_physics = UsdShade.Material(stage.GetPrimAtPath("/HDStreet/Looks/SidewalkPhysics"))
    road_friction = friction(road_physics)
    sidewalk_friction = friction(sidewalk_physics)
    check(all(isclose(a, b, abs_tol=1e-6) for a, b in zip(road_friction, (0.95, 0.82))),
          "road friction differs from task contract")
    check(all(isclose(a, b, abs_tol=1e-6) for a, b in zip(sidewalk_friction, (0.90, 0.78))),
          "sidewalk friction differs from task contract")
    for prim in roles["drivable_surface"] + roles["walkable_surface"]:
        relationship = prim.GetRelationship("material:binding:physics")
        check(relationship and relationship.GetTargets(), f"physics material not bound to {prim.GetPath()}")
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        normals = mesh.GetNormalsAttr().Get()
        offset = 0
        for face_index, count in enumerate(counts):
            check(count >= 3, f"degenerate collision face on {prim.GetPath()}")
            p0, p1, p2 = (points[indices[offset + i]] for i in range(3))
            geometric_normal = Gf.Cross(p1 - p0, p2 - p0)
            authored_normal = normals[offset]
            check(
                Gf.Dot(geometric_normal, authored_normal) > 0.0,
                f"collision winding opposes authored normal on {prim.GetPath()} face {face_index}",
            )
            offset += count

    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
    ).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    size = bounds.GetSize()
    check(110.0 <= size[0] <= 125.0, f"unexpected city width: {size[0]:.2f} m")
    check(90.0 <= size[1] <= 105.0, f"unexpected city depth: {size[1]:.2f} m")
    check(15.0 <= size[2] <= 20.0, f"unexpected city height: {size[2]:.2f} m")

    print(
        "HD_ROUTE_CITY_VALIDATION_OK",
        f"route_m={route_length:.3f}", f"speed_mps={speed:.2f}",
        f"waypoints={len(waypoints)}", "major_turns=4", "obstacle_deviations=3",
        f"colliders={len(colliders)}", f"materials={len(materials)}",
        f"dependencies={len(layers) + len(assets)}",
        f"bounds_m=({size[0]:.1f},{size[1]:.1f},{size[2]:.1f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
