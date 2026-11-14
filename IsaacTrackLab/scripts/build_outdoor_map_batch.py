#!/usr/bin/env python3
"""Build the configured high-quality outdoor all-terrain USD map batch."""

from __future__ import annotations

import argparse
import json
from math import atan2, ceil, cos, degrees, hypot, pi, sin
import os
from pathlib import Path
import random
import sys
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from isaac_track_lab.task.terrain_routes import terrain_height, terrain_mesh_height

CONFIG = ROOT / "configs/outdoor_map_batch.json"
ASSET_ROOT = ROOT / "assets/external/polyhaven"
OUTPUT_ROOT = ROOT / "assets/scenes/outdoor_batch"
TERRAIN_BUFFER_M = 30.0


def relative_asset(path: Path, layer_path: Path) -> Sdf.AssetPath:
    return Sdf.AssetPath(Path(os.path.relpath(path, layer_path.parent)).as_posix())


def set_role(prim: Usd.Prim, role: str) -> None:
    prim.CreateAttribute("track:role", Sdf.ValueTypeNames.String).Set(role)


def bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def bind_physics_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def solid_material(
    stage: Usd.Stage, path: str, color: tuple[float, float, float], roughness: float,
    opacity: float = 1.0,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def texture_file(texture_id: str, token: str) -> Path:
    matches = sorted((ASSET_ROOT / "textures" / texture_id).glob(f"*{token}*2k*.jpg"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {texture_id} {token} texture, found {matches}")
    return matches[0]


def pbr_material(
    stage: Usd.Stage, layer_path: Path, texture_id: str, name: str = "TerrainPBR2K",
) -> UsdShade.Material:
    path = f"/OutdoorMap/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    reader = UsdShade.Shader.Define(stage, f"{path}/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    transform = UsdShade.Shader.Define(stage, f"{path}/UVTransform")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0, 1.0))
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    channels = {
        "Diffuse": (texture_file(texture_id, "diff"), "sRGB", "rgb", "diffuseColor"),
        "Rough": (texture_file(texture_id, "rough"), "raw", "r", "roughness"),
        "Normal": (texture_file(texture_id, "nor_gl"), "raw", "rgb", "normal"),
    }
    for name, (file_path, color_space, output_name, surface_input) in channels.items():
        texture = UsdShade.Shader.Define(stage, f"{path}/{name}Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            relative_asset(file_path, layer_path)
        )
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            transform.ConnectableAPI(), "result"
        )
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        if name == "Normal":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
            )
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
            )
        output_type = Sdf.ValueTypeNames.Float if output_name == "r" else Sdf.ValueTypeNames.Float3
        texture.CreateOutput(output_name, output_type)
        input_type = {
            "diffuseColor": Sdf.ValueTypeNames.Color3f,
            "roughness": Sdf.ValueTypeNames.Float,
            "normal": Sdf.ValueTypeNames.Normal3f,
        }[surface_input]
        surface.CreateInput(surface_input, input_type).ConnectToSource(
            texture.ConnectableAPI(), output_name
        )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def make_physics_material(
    stage: Usd.Stage, path: str, static_friction: float, dynamic_friction: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(0.01)
    return material


def physics_material(stage: Usd.Stage, map_config: dict[str, Any]) -> UsdShade.Material:
    return make_physics_material(
        stage, "/OutdoorMap/Looks/TerrainPhysics",
        map_config["terrain"]["static_friction"], map_config["terrain"]["dynamic_friction"],
    )


def build_terrain(
    stage: Usd.Stage, map_config: dict[str, Any], visual: UsdShade.Material,
    physical: UsdShade.Material, spacing: float = 2.0, buffer_m: float = TERRAIN_BUFFER_M,
) -> tuple[dict[str, UsdShade.Material], int]:
    route_xmin, route_xmax, route_ymin, route_ymax = map(float, map_config["bounds_m"])
    xmin, xmax = route_xmin - buffer_m, route_xmax + buffer_m
    ymin, ymax = route_ymin - buffer_m, route_ymax + buffer_m
    nx, ny = round((xmax - xmin) / spacing) + 1, round((ymax - ymin) / spacing) + 1
    points = [
        Gf.Vec3f(xmin + ix * spacing, ymin + iy * spacing,
                 terrain_height(map_config, xmin + ix * spacing, ymin + iy * spacing))
        for iy in range(ny) for ix in range(nx)
    ]
    indices: list[int] = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            lower = iy * nx + ix
            # Author explicit triangles.  Rendering accepts quad faces, but the
            # current Isaac Sim PhysX importer did not create a runtime shape
            # for this large quad mesh even with approximation="none".
            indices.extend((
                lower, lower + 1, lower + nx + 1,
                lower, lower + nx + 1, lower + nx,
            ))
    mesh = UsdGeom.Mesh.Define(stage, "/OutdoorMap/Terrain/Heightfield")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (2 * (nx - 1) * (ny - 1)))
    mesh.CreateFaceVertexIndicesAttr(indices)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    # One texture tile per 8 m keeps the 2K material sharp in close robot-camera views.
    st.Set([Gf.Vec2f((point[0] - xmin) / 8.0, (point[1] - ymin) / 8.0) for point in points])
    bind_material(mesh.GetPrim(), visual)
    set_role(mesh.GetPrim(), "visual_terrain")
    mesh.GetPrim().CreateAttribute("terrain:grid_spacing_m", Sdf.ValueTypeNames.Double).Set(spacing)
    zone_materials: dict[str, UsdShade.Material] = {}
    for zone in map_config.get("challenge_zones", []):
        if not zone.get("traversable") or "static_friction" not in zone:
            continue
        material = make_physics_material(
            stage, f"/OutdoorMap/Looks/Physics_{zone['id']}",
            zone["static_friction"], zone["dynamic_friction"],
        )
        zone_materials[zone["id"]] = material

    # PhysX silently failed to create a runtime shape for the original single
    # 240x240 m / 115k-triangle mesh.  Keep that mesh for rendering, but cook
    # collision as bounded 40 m tiles.  Each tile also retains per-face soft-
    # ground material subsets.
    tile_cells = max(1, round(40.0 / spacing))
    collision_count = 0
    UsdGeom.Scope.Define(stage, "/OutdoorMap/Collision/TerrainTiles")
    for tile_y, iy0 in enumerate(range(0, ny - 1, tile_cells)):
        iy1 = min(ny - 1, iy0 + tile_cells)
        for tile_x, ix0 in enumerate(range(0, nx - 1, tile_cells)):
            ix1 = min(nx - 1, ix0 + tile_cells)
            local_nx = ix1 - ix0 + 1
            local_ny = iy1 - iy0 + 1
            tile_points = [
                Gf.Vec3f(
                    xmin + ix * spacing,
                    ymin + iy * spacing,
                    terrain_height(map_config, xmin + ix * spacing, ymin + iy * spacing),
                )
                for iy in range(iy0, iy1 + 1)
                for ix in range(ix0, ix1 + 1)
            ]
            tile_indices: list[int] = []
            for local_y in range(local_ny - 1):
                for local_x in range(local_nx - 1):
                    lower = local_y * local_nx + local_x
                    tile_indices.extend((
                        lower, lower + 1, lower + local_nx + 1,
                        lower, lower + local_nx + 1, lower + local_nx,
                    ))
            tile_path = f"/OutdoorMap/Collision/TerrainTiles/Tile_{tile_y:02d}_{tile_x:02d}"
            tile = UsdGeom.Mesh.Define(stage, tile_path)
            tile.CreateSubdivisionSchemeAttr("none")
            tile.CreatePointsAttr(tile_points)
            tile.CreateFaceVertexCountsAttr(
                [3] * (2 * (local_nx - 1) * (local_ny - 1))
            )
            tile.CreateFaceVertexIndicesAttr(tile_indices)
            bind_physics_material(tile.GetPrim(), physical)
            UsdPhysics.CollisionAPI.Apply(tile.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(tile.GetPrim()).CreateApproximationAttr().Set(
                UsdPhysics.Tokens.none
            )
            set_role(tile.GetPrim(), "collidable_terrain_tile")
            tile.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            collision_count += 1

            for zone in map_config.get("challenge_zones", []):
                if zone["id"] not in zone_materials:
                    continue
                cx, cy = map(float, zone["center_m"])
                width, height = map(float, zone["size_m"])
                faces: list[int] = []
                for local_y in range(local_ny - 1):
                    world_y = ymin + (iy0 + local_y + 0.5) * spacing
                    for local_x in range(local_nx - 1):
                        world_x = xmin + (ix0 + local_x + 0.5) * spacing
                        if (
                            abs(world_x - cx) <= width * 0.5
                            and abs(world_y - cy) <= height * 0.5
                        ):
                            face = 2 * (local_y * (local_nx - 1) + local_x)
                            faces.extend((face, face + 1))
                if faces:
                    subset = UsdGeom.Subset.Define(
                        stage, f"{tile_path}/Physics_{zone['id']}"
                    )
                    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
                    subset.CreateFamilyNameAttr("materialBind")
                    subset.CreateIndicesAttr(faces)
                    bind_physics_material(subset.GetPrim(), zone_materials[zone["id"]])
                    set_role(subset.GetPrim(), "terrain_physics_region")
    return zone_materials, collision_count


def cube(
    stage: Usd.Stage, path: str, center: tuple[float, float, float],
    size: tuple[float, float, float], material: UsdShade.Material, collision: bool = False,
    yaw_deg: float = 0.0,
) -> UsdGeom.Cube:
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(shape)
    xform.SetTranslate(Gf.Vec3d(*center))
    xform.SetRotate(Gf.Vec3f(0, 0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xform.SetScale(Gf.Vec3f(*size))
    bind_material(shape.GetPrim(), material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def conforming_strip(
    stage: Usd.Stage, path: str, map_config: dict[str, Any], points_m: list[list[float]],
    width_m: float, z_offset_m: float, material: UsdShade.Material,
    terrain_spacing_m: float, terrain_buffer_m: float, sample_spacing_m: float = 1.0,
) -> UsdGeom.Mesh:
    """Author a dense ribbon on the exact rendered/collision triangle surface.

    The ribbon remains visual-only to avoid double-contact surfaces.  Its small
    positive offset only prevents z-fighting; it must never act as a raised
    platform above the actual PhysX terrain.
    """
    sampled: list[tuple[float, float, float]] = []
    cumulative_m = 0.0
    for left, right in zip(points_m, points_m[1:]):
        ax, ay = map(float, left)
        bx, by = map(float, right)
        length = hypot(bx - ax, by - ay)
        steps = max(1, int(ceil(length / sample_spacing_m)))
        for step in range(steps + 1):
            if sampled and step == 0:
                continue
            ratio = step / steps
            x_m = ax + ratio * (bx - ax)
            y_m = ay + ratio * (by - ay)
            if sampled:
                cumulative_m += hypot(x_m - sampled[-1][0], y_m - sampled[-1][1])
            sampled.append((x_m, y_m, cumulative_m))
    if len(sampled) < 2:
        raise ValueError(f"surface corridor requires at least two distinct samples: {path}")

    points: list[Gf.Vec3f] = []
    indices: list[int] = []
    uv: list[Gf.Vec2f] = []
    for index, (x_m, y_m, distance_m) in enumerate(sampled):
        before = sampled[max(0, index - 1)]
        after = sampled[min(len(sampled) - 1, index + 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = max(hypot(dx, dy), 1e-6)
        px, py = -dy / length * width_m * 0.5, dx / length * width_m * 0.5
        for edge_x, edge_y in ((x_m + px, y_m + py), (x_m - px, y_m - py)):
            points.append(Gf.Vec3f(
                edge_x, edge_y,
                terrain_mesh_height(
                    map_config, edge_x, edge_y,
                    grid_spacing_m=terrain_spacing_m,
                    terrain_buffer_m=terrain_buffer_m,
                ) + z_offset_m,
            ))
        tile_width = max(width_m / 4.0, 1.0)
        uv.extend((Gf.Vec2f(distance_m / 4.0, 0.0),
                   Gf.Vec2f(distance_m / 4.0, tile_width)))
    for index in range(len(sampled) - 1):
        base = index * 2
        indices.extend((base, base + 1, base + 3, base + 2))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (len(sampled) - 1))
    mesh.CreateFaceVertexIndicesAttr(indices)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(uv)
    bind_material(mesh.GetPrim(), material)
    set_role(mesh.GetPrim(), "semantic_surface_visual")
    mesh.GetPrim().CreateAttribute("surface:visualOffsetM", Sdf.ValueTypeNames.Double).Set(
        z_offset_m
    )
    mesh.GetPrim().CreateAttribute("surface:sampleSpacingM", Sdf.ValueTypeNames.Double).Set(
        sample_spacing_m
    )
    mesh.GetPrim().CreateAttribute("surface:collisionSource", Sdf.ValueTypeNames.String).Set(
        "/OutdoorMap/Collision/TerrainTiles"
    )
    return mesh


def build_semantic_surfaces(
    stage: Usd.Stage, layer_path: Path, map_config: dict[str, Any],
    terrain_spacing_m: float, terrain_buffer_m: float,
) -> int:
    corridors = map_config.get("surface_corridors", [])
    if not corridors:
        return 0
    texture_ids = sorted({item["texture"] for item in corridors})
    materials = {
        texture_id: pbr_material(
            stage, layer_path, texture_id,
            "Surface_" + "".join(ch if ch.isalnum() else "_" for ch in texture_id),
        )
        for texture_id in texture_ids
    }
    lane_marking = solid_material(
        stage, "/OutdoorMap/Looks/RoadCenterLine", (0.94, 0.88, 0.55), 0.62
    )
    # Apply road/path appearance directly to the visual terrain's own faces.
    # The visible and physical terrain are authored from the same points and
    # triangle split, so this eliminates raised overlays, z-fighting, and the
    # visual/physics height mismatch which made character feet appear buried.
    terrain = UsdGeom.Mesh(stage.GetPrimAtPath("/OutdoorMap/Terrain/Heightfield"))
    terrain_points = terrain.GetPointsAttr().Get() or []
    face_counts = terrain.GetFaceVertexCountsAttr().Get() or []
    face_indices = terrain.GetFaceVertexIndicesAttr().Get() or []
    assignments: dict[str, list[int]] = {texture_id: [] for texture_id in texture_ids}
    priority = {"sidewalk": 0, "park_path": 1, "forest_trail": 1, "road": 2}

    def distance_to_corridor(x_m: float, y_m: float, corridor: dict[str, Any]) -> float:
        closest = float("inf")
        for left, right in zip(corridor["points_m"], corridor["points_m"][1:]):
            ax, ay = map(float, left)
            bx, by = map(float, right)
            dx, dy = bx - ax, by - ay
            length_squared = dx * dx + dy * dy
            ratio = 0.0 if length_squared <= 1e-12 else max(
                0.0, min(1.0, ((x_m - ax) * dx + (y_m - ay) * dy) / length_squared)
            )
            closest = min(closest, hypot(x_m - (ax + ratio * dx), y_m - (ay + ratio * dy)))
        return closest

    cursor = 0
    for face_index, count in enumerate(face_counts):
        vertices = [terrain_points[face_indices[cursor + local]] for local in range(count)]
        cursor += count
        center_x = sum(float(vertex[0]) for vertex in vertices) / count
        center_y = sum(float(vertex[1]) for vertex in vertices) / count
        matches = [
            corridor for corridor in corridors
            if distance_to_corridor(center_x, center_y, corridor)
            <= float(corridor["width_m"]) * 0.5
        ]
        if matches:
            selected = max(matches, key=lambda item: priority.get(item.get("kind", ""), 1))
            assignments[selected["texture"]].append(face_index)

    for texture_id, indices_for_material in assignments.items():
        if not indices_for_material:
            continue
        safe_id = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in texture_id)
        subset = UsdGeom.Subset.Define(
            stage, f"/OutdoorMap/Terrain/Heightfield/SurfaceMaterial_{safe_id}"
        )
        subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(indices_for_material)
        bind_material(subset.GetPrim(), materials[texture_id])
        set_role(subset.GetPrim(), "semantic_surface_material_region")
        subset.GetPrim().CreateAttribute("surface:collisionSource", Sdf.ValueTypeNames.String).Set(
            "/OutdoorMap/Collision/TerrainTiles"
        )
    UsdGeom.Subset.SetFamilyType(terrain, "materialBind", UsdGeom.Tokens.nonOverlapping)
    terrain.GetPrim().CreateAttribute("surface:geometryMode", Sdf.ValueTypeNames.String).Set(
        "terrain_face_material_subsets"
    )
    terrain.GetPrim().CreateAttribute("surface:corridorCount", Sdf.ValueTypeNames.Int).Set(
        len(corridors)
    )

    # Center markings are narrow visual curves.  They remain non-colliding and
    # use only a 4 mm separation from the common terrain surface.
    for corridor in corridors:
        if corridor.get("kind") != "road":
            continue
        safe_id = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in corridor["id"])
        marking = conforming_strip(
            stage, f"/OutdoorMap/Surfaces/{safe_id}_CenterLine", map_config,
            corridor["points_m"], 0.16, 0.004, lane_marking,
            terrain_spacing_m, terrain_buffer_m, sample_spacing_m=0.25,
        )
        set_role(marking.GetPrim(), "road_marking_visual")
    return len(corridors)


def build_curbs(
    stage: Usd.Stage, map_config: dict[str, Any], material: UsdShade.Material,
    physical: UsdShade.Material,
) -> int:
    count = 0
    for index, item in enumerate(map_config.get("curb_segments", [])):
        x_m, y_m = map(float, item["center_m"])
        height = float(item.get("height_m", 0.16))
        ground = terrain_height(map_config, x_m, y_m)
        curb = cube(
            stage, f"/OutdoorMap/Curbs/Curb_{index:03d}",
            (x_m, y_m, ground + height * 0.5),
            (float(item["length_m"]), float(item.get("width_m", 0.22)), height),
            material, collision=True, yaw_deg=float(item.get("yaw_deg", 0.0)),
        )
        bind_physics_material(curb.GetPrim(), physical)
        set_role(curb.GetPrim(), "curb_collision")
        count += 1
    return count


def circle_radius(template: dict[str, Any], scale: float) -> float:
    if template["proxy"] == "box":
        sx, sy, _ = map(float, template["size_m"])
        return 0.5 * hypot(sx, sy) * scale
    return float(template["radius_m"]) * scale


def visual_scale_for_proxy(model_path: Path, template: dict[str, Any], proxy_scale: float) -> float:
    model_stage = Usd.Stage.Open(str(model_path), load=Usd.Stage.LoadNone)
    if model_stage is None or not model_stage.GetDefaultPrim():
        raise RuntimeError(f"could not inspect model bounds: {model_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    )
    extent = cache.ComputeWorldBound(model_stage.GetDefaultPrim()).ComputeAlignedRange()
    size = extent.GetSize()
    if template["proxy"] == "sphere":
        base_radius = max(float(size[0]), float(size[1])) * 0.5
        target_radius = float(template["radius_m"])
        return proxy_scale * target_radius / max(base_radius, 1e-6)
    if template["proxy"] == "box":
        target_length = max(map(float, template["size_m"][:2]))
        base_length = max(float(size[0]), float(size[1]))
        return proxy_scale * target_length / max(base_length, 1e-6)
    if template["asset"] == "pine_tree_01":
        return proxy_scale * float(template["height_m"]) / max(float(size[2]), 1e-6)
    base_radius = max(float(size[0]), float(size[1])) * 0.5
    return proxy_scale * float(template["radius_m"]) / max(base_radius, 1e-6)


def inside_gate(x_m: float, y_m: float, gate: list[float], margin: float) -> bool:
    cx, cy, width, height = map(float, gate)
    return abs(x_m - cx) <= width * 0.5 + margin and abs(y_m - cy) <= height * 0.5 + margin


def place_obstacles(
    stage: Usd.Stage, layer_path: Path, map_config: dict[str, Any],
    debug_material: UsdShade.Material, occupied: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(int(map_config["seed"]))
    xmin, xmax, ymin, ymax = map(float, map_config["bounds_m"])
    placed: list[dict[str, Any]] = list(occupied or [])
    first_prop_index = len(placed)
    route_gates = [
        gate for gate_set in map_config.get("route_gate_sets", [map_config["route_gates"]])
        for gate in gate_set
    ]
    for template in map_config["obstacle_templates"]:
        model_matches = sorted((ASSET_ROOT / "models" / template["asset"]).glob("*.usd*"))
        if len(model_matches) != 1:
            raise RuntimeError(f"expected one model USD for {template['asset']}: {model_matches}")
        base_visual_scale = visual_scale_for_proxy(model_matches[0], template, 1.0)
        for local_index in range(int(template["count"])):
            for _ in range(5000):
                scale = rng.uniform(*map(float, template["scale"]))
                radius = circle_radius(template, scale)
                x_m = rng.uniform(xmin + radius + 3.0, xmax - radius - 3.0)
                y_m = rng.uniform(ymin + radius + 3.0, ymax - radius - 3.0)
                if any(inside_gate(x_m, y_m, gate, radius + 2.0) for gate in route_gates):
                    continue
                if any(hypot(x_m-item["x_m"], y_m-item["y_m"]) < radius+item["radius_m"]+0.75 for item in placed):
                    continue
                break
            else:
                raise RuntimeError(f"could not place {template['asset']} without overlap")
            yaw = rng.uniform(-180.0, 180.0)
            z_m = terrain_height(map_config, x_m, y_m)
            object_id = f"{template['asset']}_{local_index:03d}"
            root_path = f"/OutdoorMap/Props/{object_id}"
            root = stage.DefinePrim(root_path, "Xform")
            root.GetReferences().AddReference(relative_asset(model_matches[0], layer_path).path)
            # Repeated trees and rocks share one composed prototype. This keeps the
            # display scene high resolution without multiplying mesh memory by count.
            root.SetInstanceable(True)
            transform = UsdGeom.XformCommonAPI(root)
            transform.SetTranslate(Gf.Vec3d(x_m, y_m, z_m))
            transform.SetRotate(Gf.Vec3f(0, 0, yaw), UsdGeom.XformCommonAPI.RotationOrderXYZ)
            visual_scale = base_visual_scale * scale
            transform.SetScale(Gf.Vec3f(visual_scale, visual_scale, visual_scale))
            set_role(root, "visual_obstacle")

            proxy_path = f"/OutdoorMap/Collision/{object_id}"
            kind = template["proxy"]
            if kind == "sphere":
                proxy = UsdGeom.Sphere.Define(stage, proxy_path)
                proxy.CreateRadiusAttr(float(template["radius_m"]) * scale)
                UsdGeom.XformCommonAPI(proxy).SetTranslate(Gf.Vec3d(x_m, y_m, z_m + radius * 0.72))
            elif kind == "cylinder":
                proxy = UsdGeom.Cylinder.Define(stage, proxy_path)
                proxy.CreateAxisAttr("Z")
                proxy.CreateRadiusAttr(float(template["radius_m"]) * scale)
                height = float(template["height_m"]) * scale
                proxy.CreateHeightAttr(height)
                UsdGeom.XformCommonAPI(proxy).SetTranslate(Gf.Vec3d(x_m, y_m, z_m + height * 0.5))
            elif kind == "box":
                sx, sy, sz = (float(value) * scale for value in template["size_m"])
                proxy = cube(stage, proxy_path, (x_m, y_m, z_m + sz * 0.5), (sx, sy, sz),
                             debug_material, collision=False, yaw_deg=yaw)
            else:
                raise ValueError(f"unknown proxy type: {kind}")
            bind_material(proxy.GetPrim(), debug_material)
            UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
            UsdGeom.Imageable(proxy.GetPrim()).MakeInvisible()
            set_role(proxy.GetPrim(), "obstacle_collision_proxy")
            placed.append({
                "id": object_id, "asset": template["asset"], "x_m": round(x_m, 5),
                "y_m": round(y_m, 5), "z_m": round(z_m, 5), "yaw_deg": round(yaw, 4),
                "scale": round(scale, 5), "visual_scale": round(visual_scale, 6),
                "radius_m": round(radius, 5), "proxy": kind,
                "collision_prim": proxy_path,
            })
    return placed[first_prop_index:]


def place_explicit_obstacles(
    stage: Usd.Stage, layer_path: Path, map_config: dict[str, Any],
    debug_material: UsdShade.Material,
) -> list[dict[str, Any]]:
    """Place semantically planned props exactly where the catalog specifies."""
    placed: list[dict[str, Any]] = []
    model_cache: dict[str, tuple[Path, float]] = {}
    for item in map_config.get("obstacle_instances", []):
        asset = item["asset"]
        if asset not in model_cache:
            matches = sorted((ASSET_ROOT / "models" / asset).glob("*.usd*"))
            if len(matches) != 1:
                raise RuntimeError(f"expected one model USD for {asset}: {matches}")
            model_cache[asset] = (matches[0], visual_scale_for_proxy(matches[0], item, 1.0))
        model_path, base_visual_scale = model_cache[asset]
        x_m, y_m = float(item["x_m"]), float(item["y_m"])
        yaw = float(item.get("yaw_deg", 0.0))
        scale = float(item.get("scale", 1.0))
        z_m = terrain_height(map_config, x_m, y_m)
        object_id = str(item["id"])
        root_path = f"/OutdoorMap/Props/{object_id}"
        root = stage.DefinePrim(root_path, "Xform")
        root.GetReferences().AddReference(relative_asset(model_path, layer_path).path)
        root.SetInstanceable(True)
        transform = UsdGeom.XformCommonAPI(root)
        transform.SetTranslate(Gf.Vec3d(x_m, y_m, z_m))
        transform.SetRotate(Gf.Vec3f(0, 0, yaw), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        visual_scale = base_visual_scale * scale
        transform.SetScale(Gf.Vec3f(visual_scale, visual_scale, visual_scale))
        set_role(root, "visual_obstacle")

        proxy_path = f"/OutdoorMap/Collision/{object_id}"
        kind = item["proxy"]
        if kind == "sphere":
            radius = float(item["radius_m"]) * scale
            proxy = UsdGeom.Sphere.Define(stage, proxy_path)
            proxy.CreateRadiusAttr(radius)
            UsdGeom.XformCommonAPI(proxy).SetTranslate(Gf.Vec3d(x_m, y_m, z_m + radius * 0.72))
        elif kind == "cylinder":
            radius = float(item["radius_m"]) * scale
            height = float(item["height_m"]) * scale
            proxy = UsdGeom.Cylinder.Define(stage, proxy_path)
            proxy.CreateAxisAttr("Z")
            proxy.CreateRadiusAttr(radius)
            proxy.CreateHeightAttr(height)
            UsdGeom.XformCommonAPI(proxy).SetTranslate(Gf.Vec3d(x_m, y_m, z_m + height * 0.5))
        elif kind == "box":
            sx, sy, sz = (float(value) * scale for value in item["size_m"])
            radius = 0.5 * hypot(sx, sy)
            proxy = cube(
                stage, proxy_path, (x_m, y_m, z_m + sz * 0.5), (sx, sy, sz),
                debug_material, collision=False, yaw_deg=yaw,
            )
        else:
            raise ValueError(f"unknown proxy type: {kind}")
        bind_material(proxy.GetPrim(), debug_material)
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
        UsdGeom.Imageable(proxy.GetPrim()).MakeInvisible()
        set_role(proxy.GetPrim(), "obstacle_collision_proxy")
        record = {
            "id": object_id, "asset": asset, "x_m": round(x_m, 5),
            "y_m": round(y_m, 5), "z_m": round(z_m, 5), "yaw_deg": round(yaw, 4),
            "scale": round(scale, 5), "visual_scale": round(visual_scale, 6),
            "radius_m": round(radius, 5), "proxy": kind,
            "collision_prim": proxy_path,
        }
        if kind == "box":
            record["size_m"] = [sx, sy, sz]
        placed.append(record)
    return placed


def build_procedural_structures(
    stage: Usd.Stage, layer_path: Path, map_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Author lightweight, textured city blocks with conservative box collisions."""
    structures = []
    if not map_config.get("procedural_structures"):
        return structures
    material_ids = sorted({item["material"] for item in map_config["procedural_structures"]})
    materials = {
        material_id: pbr_material(stage, layer_path, material_id, f"Building_{material_id}")
        for material_id in material_ids
    }
    window = solid_material(stage, "/OutdoorMap/Looks/WindowGlass", (0.025, 0.07, 0.11), 0.16)
    roof = solid_material(stage, "/OutdoorMap/Looks/BuildingRoof", (0.11, 0.12, 0.13), 0.88)
    for item in map_config["procedural_structures"]:
        x_m, y_m = map(float, item["center_m"])
        width, depth, height = map(float, item["size_m"])
        yaw = float(item.get("yaw_deg", 0.0))
        z_m = terrain_height(map_config, x_m, y_m)
        path = f"/OutdoorMap/Structures/{item['id']}"
        body = cube(stage, f"{path}/Body", (x_m, y_m, z_m + height * 0.5),
                    (width, depth, height), materials[item["material"]], collision=True,
                    yaw_deg=yaw)
        set_role(body.GetPrim(), "building_collision")
        cube(stage, f"{path}/Roof", (x_m, y_m, z_m + height + 0.16),
             (width + 0.35, depth + 0.35, 0.28), roof, collision=False, yaw_deg=yaw)
        # Window strips preserve an urban silhouette without multiplying geometry.
        facade_y = y_m - depth * 0.5 - 0.012
        for floor in range(1, max(2, int(height // 3.0))):
            cube(stage, f"{path}/Windows_{floor:02d}",
                 (x_m, facade_y, z_m + min(height - 1.0, 1.5 + floor * 2.7)),
                 (width * 0.72, 0.025, 0.65), window, collision=False, yaw_deg=yaw)
        structures.append({
            "id": item["id"], "asset": "procedural_building", "x_m": round(x_m, 5),
            "y_m": round(y_m, 5), "z_m": round(z_m, 5), "yaw_deg": yaw,
            "scale": 1.0, "visual_scale": 1.0,
            "radius_m": round(0.5 * hypot(width, depth), 5), "proxy": "box",
            "collision_prim": f"{path}/Body", "size_m": [width, depth, height],
        })
    return structures


def build_traversable_features(
    stage: Usd.Stage, layer_path: Path, map_config: dict[str, Any],
    materials: dict[str, UsdShade.Material], physical: UsdShade.Material,
) -> list[dict[str, Any]]:
    """Build collidable route features with physically usable top surfaces."""
    instances: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for feature in map_config.get("traversable_step_features", []):
        cx, cy = map(float, feature["center_m"])
        yaw = float(feature.get("yaw_deg", 0.0))
        count = max(1, int(feature.get("count", 1)))
        safe_id = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in feature["id"])
        if feature["kind"] == "low_log":
            # The log axis is across the trail; multiple logs are spaced along the trail.
            trail_angle = (yaw - 90.0) * pi / 180.0
            for index in range(count):
                offset = (index - (count - 1) * 0.5) * 0.56
                instances.append({
                    "id": f"{safe_id}_log_{index}", "asset": "dead_tree_trunk",
                    "x_m": cx + cos(trail_angle) * offset,
                    "y_m": cy + sin(trail_angle) * offset,
                    "yaw_deg": yaw, "scale": 1.0, "proxy": "box",
                    "size_m": [3.1, 0.42, 0.20],
                })
        elif feature["kind"] == "embedded_stones":
            cross_angle = yaw * pi / 180.0
            for index in range(count):
                offset = (index - (count - 1) * 0.5) * 0.50
                instances.append({
                    "id": f"{safe_id}_stone_{index}", "asset": "rock_09",
                    "x_m": cx + cos(cross_angle) * offset,
                    "y_m": cy + sin(cross_angle) * offset,
                    "yaw_deg": yaw + index * 37.0, "scale": 0.85 + 0.08 * index,
                    "proxy": "sphere", "radius_m": 0.17,
                })
        elif feature["kind"] in {"curb_table", "platform", "stair_platform"}:
            yaw_radians = yaw * pi / 180.0
            width = float(feature["width_m"])
            reference_ground = terrain_height(map_config, cx, cy)
            bottom = reference_ground - 0.08
            material_id = str(feature.get("material", "rock"))
            if material_id not in materials:
                raise ValueError(f"unknown traversable feature material: {material_id}")

            def add_block(local_id: str, along_m: float, length_m: float, top_m: float) -> None:
                x_m = cx + cos(yaw_radians) * along_m
                y_m = cy + sin(yaw_radians) * along_m
                height = max(0.02, top_m - bottom)
                path = f"/OutdoorMap/TraversableFeatures/{safe_id}/{local_id}"
                block = cube(
                    stage, path, (x_m, y_m, bottom + height * 0.5),
                    (length_m, width, height), materials[material_id], collision=True,
                    yaw_deg=yaw,
                )
                bind_physics_material(block.GetPrim(), physical)
                set_role(block.GetPrim(), "traversable_step_collision")
                block.GetPrim().CreateAttribute(
                    "route:featureId", Sdf.ValueTypeNames.String
                ).Set(str(feature["id"]))
                block.GetPrim().CreateAttribute(
                    "route:required", Sdf.ValueTypeNames.Bool
                ).Set(bool(feature.get("required_on_route", False)))
                records.append({
                    "id": f"{safe_id}_{local_id}", "feature_id": str(feature["id"]),
                    "kind": str(feature["kind"]), "x_m": round(x_m, 5),
                    "y_m": round(y_m, 5), "z_m": round(bottom + height * 0.5, 5),
                    "yaw_deg": yaw, "size_m": [length_m, width, height],
                    "surface_top_m": round(top_m, 5), "proxy": "box",
                    "collision_prim": path,
                })

            if feature["kind"] in {"curb_table", "platform"}:
                add_block(
                    "Body", 0.0, float(feature["length_m"]),
                    reference_ground + float(feature["height_m"]),
                )
            else:
                step_count = int(feature["step_count"])
                tread = float(feature["tread_depth_m"])
                plateau = float(feature["platform_length_m"])
                rise = float(feature["step_rise_m"])
                total_length = 2.0 * step_count * tread + plateau
                cursor = -total_length * 0.5
                for index in range(step_count):
                    add_block(
                        f"Up_{index + 1:02d}", cursor + tread * 0.5, tread,
                        reference_ground + (index + 1) * rise,
                    )
                    cursor += tread
                add_block(
                    "Plateau", cursor + plateau * 0.5, plateau,
                    reference_ground + step_count * rise,
                )
                cursor += plateau
                for index in range(step_count):
                    add_block(
                        f"Down_{index + 1:02d}", cursor + tread * 0.5, tread,
                        reference_ground + (step_count - index) * rise,
                    )
                    cursor += tread
        else:
            raise ValueError(f"unknown traversable feature: {feature['kind']}")
    if instances:
        feature_config = dict(map_config)
        feature_config["obstacle_instances"] = instances
        asset_records = place_explicit_obstacles(
            stage, layer_path, feature_config, materials["debug"]
        )
        for record in asset_records:
            visual = stage.GetPrimAtPath(f"/OutdoorMap/Props/{record['id']}")
            proxy = stage.GetPrimAtPath(record["collision_prim"])
            set_role(visual, "traversable_step_visual")
            set_role(proxy, "traversable_step_collision")
        records.extend(asset_records)
    return records


def build_elliptical_water(
    stage: Usd.Stage, map_config: dict[str, Any], zone: dict[str, Any],
    water_material: UsdShade.Material, rock_material: UsdShade.Material,
) -> int:
    cx, cy = map(float, zone["center_m"])
    width, height = map(float, zone["size_m"])
    rx, ry = width * 0.5, height * 0.5
    segments = 64
    shore_samples = [
        terrain_height(
            map_config, cx + rx * cos(2.0 * pi * index / segments),
            cy + ry * sin(2.0 * pi * index / segments),
        )
        for index in range(segments)
    ]
    requested_level = terrain_height(map_config, cx, cy) + float(zone.get("water_raise_m", 0.5))
    water_level = min(requested_level, min(shore_samples) - 0.045)
    points = [Gf.Vec3f(cx, cy, water_level)]
    points.extend(
        Gf.Vec3f(cx + rx * cos(2.0 * pi * index / segments),
                 cy + ry * sin(2.0 * pi * index / segments), water_level)
        for index in range(segments)
    )
    mesh = UsdGeom.Mesh.Define(stage, f"/OutdoorMap/ChallengeZones/{zone['id']}")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * segments)
    indices: list[int] = []
    for index in range(segments):
        indices.extend((0, index + 1, (index + 1) % segments + 1))
    mesh.CreateFaceVertexIndicesAttr(indices)
    bind_material(mesh.GetPrim(), water_material)
    mesh.GetPrim().CreateAttribute("water:level_m", Sdf.ValueTypeNames.Double).Set(water_level)
    mesh.GetPrim().CreateAttribute("water:shore_clearance_m", Sdf.ValueTypeNames.Double).Set(
        min(shore_samples) - water_level
    )

    # An irregular, embedded stone edge visually joins water to terrain.  Keep the
    # non-uniformly scaled ellipsoid visual separate from its collision proxy:
    # PhysX warns that a scaled sphere can cook a shape which no longer matches
    # the visible ellipsoid.  A hidden, uniformly sized sphere is deterministic
    # and deliberately conservative for locomotion.
    collision_count = 0
    for index in range(28):
        angle = 2.0 * pi * index / 28.0
        jitter = 1.0 + 0.035 * sin(index * 2.17)
        x_m, y_m = cx + rx * jitter * cos(angle), cy + ry * jitter * sin(angle)
        ground = terrain_height(map_config, x_m, y_m)
        radius = 0.25 + 0.10 * (0.5 + 0.5 * sin(index * 1.73))
        stone_path = f"/OutdoorMap/Shore/{zone['id']}_Stone_{index:02d}"
        stone = UsdGeom.Sphere.Define(stage, stone_path)
        stone.CreateRadiusAttr(radius)
        xform = UsdGeom.XformCommonAPI(stone)
        xform.SetTranslate(Gf.Vec3d(x_m, y_m, ground + radius * 0.35))
        xform.SetScale(Gf.Vec3f(1.35, 0.85, 0.65))
        bind_material(stone.GetPrim(), rock_material)
        set_role(stone.GetPrim(), "lake_shore_visual")

        proxy_radius = radius * 0.65
        proxy = UsdGeom.Sphere.Define(stage, f"{stone_path}_Collision")
        proxy.CreateRadiusAttr(proxy_radius)
        proxy_xform = UsdGeom.XformCommonAPI(proxy)
        proxy_xform.SetTranslate(Gf.Vec3d(x_m, y_m, ground + radius * 0.35))
        UsdGeom.Imageable(proxy.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
        set_role(proxy.GetPrim(), "lake_shore_collision_proxy")
        proxy.GetPrim().CreateAttribute("track:visual_prim", Sdf.ValueTypeNames.String).Set(stone_path)
        collision_count += 1
    return collision_count


def build_challenge_zones(
    stage: Usd.Stage, map_config: dict[str, Any], materials: dict[str, UsdShade.Material],
    zone_physics: dict[str, UsdShade.Material],
) -> int:
    collision_count = 0
    for zone in map_config.get("challenge_zones", []):
        cx, cy = map(float, zone["center_m"])
        width, height = map(float, zone["size_m"])
        z_m = terrain_height(map_config, cx, cy)
        path = f"/OutdoorMap/ChallengeZones/{zone['id']}"
        if zone["kind"] == "water":
            if zone.get("shape") == "ellipse":
                collision_count += build_elliptical_water(
                    stage, map_config, zone, materials["water"], materials["rock"]
                )
            else:
                cube(stage, path, (cx, cy, z_m + 0.03), (width, height, 0.06),
                     materials["water"], collision=False)
        elif zone["kind"] == "boardwalk":
            plank_count = max(2, int(height / 1.4))
            for index in range(plank_count):
                y_m = cy - height * 0.5 + (index + 0.5) * height / plank_count
                ground = terrain_height(map_config, cx, y_m)
                plank = cube(stage, f"{path}/Plank_{index:03d}", (cx, y_m, ground + 0.24),
                             (width, height / plank_count * 0.92, 0.36), materials["wood"], collision=True)
                bind_physics_material(plank.GetPrim(), zone_physics[zone["id"]])
                collision_count += 1
        elif zone["kind"] == "steps":
            step_count = 9
            for index in range(step_count):
                x_m = cx - width * 0.5 + (index + 0.5) * width / step_count
                ground = terrain_height(map_config, x_m, cy)
                rise = 0.05 + 0.045 * (index % 3)
                step = cube(stage, f"{path}/Step_{index:02d}", (x_m, cy, ground + rise * 0.5),
                            (width / step_count * 0.9, height, rise), materials["rock"], collision=True)
                bind_physics_material(step.GetPrim(), zone_physics[zone["id"]])
                collision_count += 1
        else:
            # A thin non-colliding tint communicates the training zone in debug views;
            # the actual geometry and physics continue to come from the heightfield.
            debug_zone = cube(stage, path, (cx, cy, z_m + 0.012), (width, height, 0.018),
                              materials["zone"], collision=False)
            UsdGeom.Imageable(debug_zone.GetPrim()).MakeInvisible()
        prim = stage.GetPrimAtPath(path)
        if prim:
            set_role(prim, f"challenge_{zone['kind']}")
            prim.CreateAttribute("challenge:cost", Sdf.ValueTypeNames.Double).Set(float(zone["cost"]))
            prim.CreateAttribute("challenge:traversable", Sdf.ValueTypeNames.Bool).Set(bool(zone["traversable"]))
    return collision_count


def add_lighting(stage: Usd.Stage, layer_path: Path, map_config: dict[str, Any]) -> None:
    hdri_matches = sorted((ASSET_ROOT / "hdris" / map_config["terrain"]["hdri"]).glob("*.exr"))
    if len(hdri_matches) != 1:
        raise RuntimeError(f"expected one HDRI for {map_config['terrain']['hdri']}")
    dome = UsdLux.DomeLight.Define(stage, "/OutdoorMap/Lighting/Sky")
    dome.CreateTextureFileAttr(relative_asset(hdri_matches[0], layer_path))
    dome.CreateTextureFormatAttr("latlong")
    dome.CreateIntensityAttr(450.0)
    dome.CreateExposureAttr(-1.7)
    sun = UsdLux.DistantLight.Define(stage, "/OutdoorMap/Lighting/Sun")
    sun.CreateIntensityAttr(1200.0)
    sun.CreateAngleAttr(0.22)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.84))
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(42.0, -28.0, 32.0))


def build_one(
    config: dict[str, Any], map_config: dict[str, Any], output_root: Path = OUTPUT_ROOT,
) -> Path:
    quality = config.get("quality_contract", {})
    terrain_spacing = float(quality.get("terrain_grid_spacing_m", 2.0))
    terrain_buffer = float(quality.get("terrain_buffer_m", TERRAIN_BUFFER_M))
    usd_extension = str(quality.get("usd_extension", "usda")).lstrip(".")
    output_dir = output_root / map_config["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_path = output_dir / f"{map_config['id']}.{usd_extension}"
    stage = Usd.Stage.CreateNew(str(layer_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(30.0)
    root = UsdGeom.Xform.Define(stage, "/OutdoorMap")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().customLayerData = {
        "batchId": config["batch_id"], "mapId": map_config["id"],
        "mapLabel": map_config["label"], "assetLicense": "CC0",
        "dynamicEvents": "none", "distractorPeople": 0,
    }
    for scope in (
        "Looks", "Terrain", "Surfaces", "Curbs", "Structures", "Props", "Collision",
        "TraversableFeatures",
        "ChallengeZones", "Shore", "Route", "Lighting",
    ):
        UsdGeom.Scope.Define(stage, f"/OutdoorMap/{scope}")

    visual = pbr_material(stage, layer_path, map_config["terrain"]["texture"])
    physical = physics_material(stage, map_config)
    materials = {
        "debug": solid_material(stage, "/OutdoorMap/Looks/CollisionDebug", (1.0, 0.0, 0.8), 1.0),
        "zone": solid_material(stage, "/OutdoorMap/Looks/ZoneDebug", (0.22, 0.54, 0.15), 0.82, 0.23),
        "water": solid_material(stage, "/OutdoorMap/Looks/Water", (0.025, 0.17, 0.23), 0.16, 0.72),
        "wood": solid_material(stage, "/OutdoorMap/Looks/BoardwalkWood", (0.24, 0.14, 0.065), 0.78),
        "rock": solid_material(stage, "/OutdoorMap/Looks/StepRock", (0.33, 0.34, 0.31), 0.91),
        "curb": solid_material(stage, "/OutdoorMap/Looks/CurbConcrete", (0.49, 0.50, 0.48), 0.86),
    }
    zone_physics, terrain_collision_count = build_terrain(
        stage, map_config, visual, physical, spacing=terrain_spacing, buffer_m=terrain_buffer
    )
    surface_count = build_semantic_surfaces(
        stage, layer_path, map_config, terrain_spacing, terrain_buffer
    )
    curb_count = build_curbs(stage, map_config, materials["curb"], physical)
    structures = build_procedural_structures(stage, layer_path, map_config)
    if "obstacle_instances" in map_config:
        props = place_explicit_obstacles(stage, layer_path, map_config, materials["debug"])
    else:
        props = place_obstacles(stage, layer_path, map_config, materials["debug"], occupied=structures)
    obstacles = structures + props
    traversable_features = build_traversable_features(
        stage, layer_path, map_config, materials, physical
    )
    challenge_collisions = build_challenge_zones(stage, map_config, materials, zone_physics)
    add_lighting(stage, layer_path, map_config)
    route = stage.GetPrimAtPath("/OutdoorMap/Route")
    route.CreateAttribute("route:generator", Sdf.ValueTypeNames.String).Set(
        "isaac_track_lab.task.terrain_routes.TerrainRouteGenerator"
    )
    route.CreateAttribute("route:seeded", Sdf.ValueTypeNames.Bool).Set(True)
    route.CreateAttribute("route:minimum_length_m", Sdf.ValueTypeNames.Double).Set(
        float(config["route_defaults"]["minimum_length_m"])
    )
    route.CreateAttribute("route:maximum_length_m", Sdf.ValueTypeNames.Double).Set(
        float(config["route_defaults"]["maximum_length_m"])
    )
    stage.GetRootLayer().Save()

    manifest = {
        "schema_version": 1, "batch_id": config["batch_id"], "map_id": map_config["id"],
        "label": map_config["label"], "usd": layer_path.name, "seed": map_config["seed"],
        "bounds_m": map_config["bounds_m"], "terrain": map_config["terrain"],
        "terrain_footprint_bounds_m": [
            map_config["bounds_m"][0] - terrain_buffer,
            map_config["bounds_m"][1] + terrain_buffer,
            map_config["bounds_m"][2] - terrain_buffer,
            map_config["bounds_m"][3] + terrain_buffer,
        ],
        "friction": {
            "static": map_config["terrain"]["static_friction"],
            "dynamic": map_config["terrain"]["dynamic_friction"],
        },
        "family": map_config.get("family"), "terrain_grid_spacing_m": terrain_spacing,
        "obstacles": obstacles, "obstacle_count": len(obstacles),
        "structure_count": len(structures), "prop_count": len(props),
        "surface_corridor_count": surface_count, "curb_count": curb_count,
        "traversable_features": traversable_features,
        "traversable_feature_count": len(traversable_features),
        "planning_contract": config.get("planning_contract", {}),
        "navigation_corridors": map_config.get("navigation_corridors", []),
        "challenge_zones": map_config["challenge_zones"],
        "terrain_collision_tile_count": terrain_collision_count,
        "collision_prims_expected": (
            terrain_collision_count + len(obstacles) + curb_count
            + len(traversable_features) + challenge_collisions
        ),
        "route_generator": "isaac_track_lab.task.terrain_routes.TerrainRouteGenerator",
        "route_defaults": config["route_defaults"],
        "dynamic_events": [], "distractor_people": 0,
    }
    (output_dir / "scene_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("OUTDOOR_MAP_BUILD_OK", map_config["id"], layer_path, len(obstacles), flush=True)
    return layer_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", action="append", dest="map_ids", help="build only this map id")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve() if args.output_root else (
        ROOT / config.get("scene_output_root", str(OUTPUT_ROOT.relative_to(ROOT)))
    )
    selected = [item for item in config["maps"] if not args.map_ids or item["id"] in args.map_ids]
    if args.map_ids and len(selected) != len(set(args.map_ids)):
        raise SystemExit("one or more requested map ids are unknown")
    for map_config in selected:
        build_one(config, map_config, output_root)
    print("OUTDOOR_BATCH_BUILD_OK", len(selected), output_root, flush=True)


if __name__ == "__main__":
    main()
