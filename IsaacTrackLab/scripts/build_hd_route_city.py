#!/usr/bin/env python3
"""Build the intersecting-road HD city and its approximately 300 m task route."""

from __future__ import annotations

from math import dist
from pathlib import Path
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hd_street as base


OUTPUT = ROOT / "assets/scenes/hd_route_city/hd_route_city.usda"
POLYHAVEN = ROOT / "assets/external/polyhaven"
SURFACE_Z = 0.20
ROUTE_POINTS = (
    (-52.0, -36.0), (-30.0, -36.0), (-18.0, -34.5), (-6.0, -36.0),
    (20.0, -36.0), (44.0, -36.0), (46.8, -34.8), (48.0, -32.0),
    (48.0, -27.0), (44.8, -23.0), (48.0, -19.0), (48.0, -16.0),
    (46.8, -13.2), (44.0, -12.0), (18.0, -12.0), (6.0, -10.5),
    (-6.0, -12.0), (-32.0, -12.0), (-34.8, -10.8), (-36.0, -8.0),
    (-36.0, 8.0), (-34.8, 10.8), (-32.0, 12.0), (-8.0, 12.0),
    (4.0, 13.5), (16.0, 12.0), (42.0, 12.0),
)
ROUTE_LENGTH_M = sum(dist(a, b) for a, b in zip(ROUTE_POINTS, ROUTE_POINTS[1:]))


def bind_physics_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def add_building(stage, name, center, size, height, material, palette, window_side="south"):
    x, y = center
    width, depth = size
    root = stage.DefinePrim(f"/HDStreet/Architecture/{name}", "Xform")
    base.set_role(root, "architecture")
    base.box_mesh(stage, f"{root.GetPath()}/Shell", (x, y, 0.36 + height * 0.5),
                  (width, depth, height), material, "building_shell")
    base.cube(stage, f"{root.GetPath()}/Roof", (x, y, 0.48 + height),
              (width + 0.25, depth + 0.25, 0.24), palette["trim"], "roof", collision=True)
    # One detailed street-facing facade keeps close views rich without making
    # every window a physics shape. The shell gets one separate collision proxy.
    along_x = window_side in {"south", "north"}
    span = width if along_x else depth
    count = max(3, int(span // 2.4))
    floors = max(2, int((height - 1.2) // 2.5))
    for floor in range(floors):
        z = 2.0 + floor * 2.45
        for column in range(count):
            offset = -span * 0.5 + (column + 1) * span / (count + 1)
            if along_x:
                wx, wy = x + offset, y + (-depth * 0.5 - 0.035 if window_side == "south" else depth * 0.5 + 0.035)
                wsize = (1.0, 0.07, 1.25)
            else:
                wx, wy = x + (-width * 0.5 - 0.035 if window_side == "west" else width * 0.5 + 0.035), y + offset
                wsize = (0.07, 1.0, 1.25)
            base.cube(stage, f"{root.GetPath()}/Windows/F{floor}_C{column}",
                      (wx, wy, z), wsize, palette["glass"], "window")
    proxy = base.cube(stage, f"/HDStreet/Collision/{name}", (x, y, 0.36 + height * 0.5),
                      (width, depth, height), palette["collision"], "building_collision", collision=True)
    UsdGeom.Imageable(proxy).MakeInvisible()


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(30.0)
    root = UsdGeom.Xform.Define(stage, "/HDStreet")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().customLayerData = {
        "assetSource": "Intersecting procedural city + Poly Haven CC0",
        "routeLengthMeters": ROUTE_LENGTH_M,
        "targetSpeedMetersPerSecond": 1.25,
        "roadStaticFriction": 0.95,
        "roadDynamicFriction": 0.82,
    }
    for scope in ("Looks", "Ground", "Architecture", "StreetFurniture", "Props", "Collision", "Route", "Lighting"):
        UsdGeom.Scope.Define(stage, f"/HDStreet/{scope}")

    asphalt_dir = POLYHAVEN / "textures/asphalt_02"
    paving_dir = POLYHAVEN / "textures/brick_pavement_02"
    brick_dir = POLYHAVEN / "textures/brick_wall_001"
    beige_dir = POLYHAVEN / "textures/beige_wall_001"
    asphalt = base.pbr_material(stage, "Asphalt2K", asphalt_dir,
        ("asphalt_02_diff_2k.jpg", "asphalt_02_rough_2k.jpg", "asphalt_02_nor_gl_2k.jpg"), (8, 1.5), use_normal=False)
    paving = base.pbr_material(stage, "BrickPaving2K", paving_dir,
        ("brick_pavement_02_diff_2k.jpg", "brick_pavement_02_rough_2k.jpg", "brick_pavement_02_nor_gl_2k.jpg"), (4, 2), use_normal=False)
    brick = base.pbr_material(stage, "BrickWall2K", brick_dir,
        ("brick_wall_001_diffuse_2k.jpg", "brick_wall_001_rough_2k.jpg", "brick_wall_001_nor_gl_2k.jpg"), (2, 2), use_normal=False)
    beige = base.pbr_material(stage, "BeigeWall2K", beige_dir,
        ("beige_wall_001_diff_2k.jpg", "beige_wall_001_rough_2k.jpg", "beige_wall_001_nor_gl_2k.jpg"), (2, 2), use_normal=False)
    palette = {
        "line": base.solid_material(stage, "RoadMarking", (0.91, 0.87, 0.68), 0.45),
        "glass": base.solid_material(stage, "WindowGlass", (0.045, 0.15, 0.22), 0.14, 0.10),
        "trim": base.solid_material(stage, "WarmTrim", (0.25, 0.17, 0.11), 0.64),
        "metal": base.solid_material(stage, "StreetMetal", (0.06, 0.07, 0.08), 0.38, 0.70),
        "green": base.solid_material(stage, "UrbanGreen", (0.04, 0.18, 0.055), 0.83),
        "collision": base.solid_material(stage, "CollisionDebug", (1.0, 0.0, 0.8), 1.0),
    }
    base.palette = palette

    road_physics = UsdShade.Material.Define(stage, "/HDStreet/Looks/RoadPhysics")
    road_api = UsdPhysics.MaterialAPI.Apply(road_physics.GetPrim())
    road_api.CreateStaticFrictionAttr(0.95)
    road_api.CreateDynamicFrictionAttr(0.82)
    road_api.CreateRestitutionAttr(0.02)
    sidewalk_physics = UsdShade.Material.Define(stage, "/HDStreet/Looks/SidewalkPhysics")
    walk_api = UsdPhysics.MaterialAPI.Apply(sidewalk_physics.GetPrim())
    walk_api.CreateStaticFrictionAttr(0.90)
    walk_api.CreateDynamicFrictionAttr(0.78)
    walk_api.CreateRestitutionAttr(0.02)

    # Intersecting road grid: three east-west roads and three north-south roads.
    road_index = 0
    for y in (-36.0, -12.0, 12.0):
        road = base.box_mesh(stage, f"/HDStreet/Ground/Road_H_{road_index}", (2.0, y, 0.10),
                             (116.0, 8.0, 0.20), asphalt, "drivable_surface", collision=True)
        bind_physics_material(road.GetPrim(), road_physics)
        road_index += 1
    for x in (-36.0, 0.0, 48.0):
        road = base.box_mesh(stage, f"/HDStreet/Ground/Road_V_{road_index}", (x, 0.0, 0.10),
                             (8.0, 96.0, 0.20), asphalt, "drivable_surface", collision=True)
        bind_physics_material(road.GetPrim(), road_physics)
        road_index += 1

    # Raised city blocks act as continuous sidewalks and have their own friction.
    block_columns = ((-50.0, 12.0), (-18.0, 28.0), (24.0, 40.0), (56.0, 8.0))
    block_rows = ((-46.0, 12.0), (-24.0, 16.0), (0.0, 16.0), (26.0, 20.0), (44.0, 12.0))
    for ci, (x, width) in enumerate(block_columns):
        for ri, (y, depth) in enumerate(block_rows):
            block = base.box_mesh(stage, f"/HDStreet/Ground/Block_{ci}_{ri}", (x, y, 0.18),
                                  (width, depth, 0.36), paving, "walkable_surface", collision=True)
            bind_physics_material(block.GetPrim(), sidewalk_physics)

    # Lane markings stop before intersections, which makes route turns legible.
    mark_id = 0
    for y in (-36.0, -12.0, 12.0):
        for x in range(-52, 55, 8):
            if any(abs(x - crossing) < 6 for crossing in (-36, 0, 48)):
                continue
            base.cube(stage, f"/HDStreet/Ground/Mark_{mark_id}", (x, y, 0.214),
                      (3.6, 0.11, 0.018), palette["line"], "road_marking")
            mark_id += 1
    for x in (-36.0, 0.0, 48.0):
        for y in range(-44, 45, 8):
            if any(abs(y - crossing) < 6 for crossing in (-36, -12, 12)):
                continue
            base.cube(stage, f"/HDStreet/Ground/Mark_{mark_id}", (x, y, 0.214),
                      (0.11, 3.6, 0.018), palette["line"], "road_marking")
            mark_id += 1

    buildings = (
        ("WestSouth", (-50, -24), (8, 12), 11, brick, "east"),
        ("MidSouthA", (-24, -24), (14, 11), 15, beige, "south"),
        ("MidSouthB", (-10, -24), (10, 11), 10, brick, "south"),
        ("EastSouthA", (15, -24), (15, 11), 13, brick, "south"),
        ("EastSouthB", (34, -24), (14, 11), 17, beige, "south"),
        ("WestCenter", (-50, 0), (8, 12), 14, beige, "east"),
        ("MidCenter", (-18, 0), (20, 12), 12, brick, "south"),
        ("EastCenterA", (14, 0), (16, 12), 15, beige, "north"),
        ("EastCenterB", (35, 0), (13, 12), 11, brick, "north"),
        ("WestNorth", (-50, 26), (8, 15), 12, brick, "east"),
        ("MidNorth", (-18, 26), (20, 14), 16, beige, "south"),
        ("EastNorthA", (13, 26), (15, 14), 12, brick, "south"),
        ("EastNorthB", (34, 26), (14, 14), 15, beige, "south"),
    )
    for name, center, size, height, material, window_side in buildings:
        add_building(stage, name, center, size, height, material, palette, window_side)

    # Collidable lamp poles and tree trunks are placed on sidewalks, never in lanes.
    for index, (x, y) in enumerate(((-42,-31),(-28,-31),(-6,-31),(8,-31),(40,-31),
                                     (-42,-7),(-28,-7),(-6,-7),(8,-7),(40,-7),
                                     (-42,17),(-28,17),(-6,17),(8,17),(40,17))):
        base.cylinder(stage, f"/HDStreet/StreetFurniture/Lamp_{index}", (x, y, 2.36),
                      0.07, 4.0, palette["metal"], "street_light", collision=True)
    for index, (x, y) in enumerate(((-46,-20),(-15,-18),(28,-18),(-46,3),(-14,4),(30,4),(-46,29),(-12,30),(29,30))):
        base.cylinder(stage, f"/HDStreet/StreetFurniture/Tree_{index}/Trunk", (x, y, 1.18),
                      0.20, 2.0, palette["trim"], "tree_trunk", collision=True)
        crown = UsdGeom.Sphere.Define(stage, f"/HDStreet/StreetFurniture/Tree_{index}/Crown")
        crown.CreateRadiusAttr(1.15)
        UsdGeom.XformCommonAPI(crown).SetTranslate(Gf.Vec3d(x, y, 2.75))
        UsdShade.MaterialBindingAPI.Apply(crown.GetPrim()).Bind(palette["green"])
        base.set_role(crown.GetPrim(), "tree_crown")
        UsdPhysics.CollisionAPI.Apply(crown.GetPrim())

    # Three obstacles force route deviations; the remaining props enrich curbside collision cases.
    base.add_model(stage, "BarrierRoute", "concrete_road_barrier", (-18.0, -36.0, SURFACE_Z), 0, 1.15, (1.8, 0.8, 1.0))
    base.add_model(stage, "CoveredCarRoute", "covered_car", (48.0, -23.0, SURFACE_Z), 0, 1.0, (2.0, 4.5, 1.5))
    base.add_model(stage, "PlanterRoute", "planter_box_01", (6.0, -12.0, SURFACE_Z), 0, 1.6, (1.5, 0.9, 0.75))
    base.add_model(stage, "Hydrant", "fire_hydrant", (-33.5, 13.7, 0.36), 0, 1.0, (0.8, 0.8, 0.9))
    base.add_model(stage, "TrashCan", "metal_trash_can", (21.0, 16.5, 0.36), 0, 0.9, (1.8, 0.7, 0.9))
    base.add_model(stage, "Bench", "painted_wooden_bench", (30.0, -17.0, 0.36), 90, 1.55, (0.8, 1.9, 1.4))

    for index, (x, y) in enumerate(ROUTE_POINTS):
        waypoint = UsdGeom.Xform.Define(stage, f"/HDStreet/Route/Waypoint_{index:02d}")
        UsdGeom.XformCommonAPI(waypoint).SetTranslate(Gf.Vec3d(x, y, SURFACE_Z))
        base.set_role(waypoint.GetPrim(), "route_waypoint")
        waypoint.GetPrim().CreateAttribute("route:index", Sdf.ValueTypeNames.Int).Set(index)
    route_prim = stage.GetPrimAtPath("/HDStreet/Route")
    route_prim.CreateAttribute("route:length_m", Sdf.ValueTypeNames.Double).Set(ROUTE_LENGTH_M)
    route_prim.CreateAttribute("route:target_speed_mps", Sdf.ValueTypeNames.Double).Set(1.25)
    route_prim.CreateAttribute("route:major_turn_count", Sdf.ValueTypeNames.Int).Set(4)

    dome = UsdLux.DomeLight.Define(stage, "/HDStreet/Lighting/UrbanSky")
    dome.CreateTextureFileAttr(base.asset_path(POLYHAVEN / "hdris/urban_street_01/urban_street_01_2k.exr"))
    dome.CreateIntensityAttr(450.0)
    dome.CreateExposureAttr(-2.0)
    dome.CreateTextureFormatAttr("latlong")
    UsdLux.ShadowAPI.Apply(dome.GetPrim()).CreateShadowEnableAttr(False)
    sun = UsdLux.DistantLight.Define(stage, "/HDStreet/Lighting/Sun")
    sun.CreateIntensityAttr(3200.0)
    sun.CreateAngleAttr(0.12)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.80))
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(48.0, -28.0, 24.0))

    stage.GetRootLayer().Save()
    print("HD_ROUTE_CITY_BUILD_OK", OUTPUT, f"route_length_m={ROUTE_LENGTH_M:.3f}")


if __name__ == "__main__":
    build()
