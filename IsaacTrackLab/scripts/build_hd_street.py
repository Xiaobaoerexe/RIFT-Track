#!/usr/bin/env python3
"""Shared USD construction helpers for the 305 m HD route builder.

This module is retained because ``build_hd_route_city.py`` imports its material,
mesh, lighting, and asset-reference helpers. It is no longer a standalone
straight-street baseline.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/scenes/hd_street/hd_street.usda"
POLYHAVEN = ROOT / "assets/external/polyhaven"
SURFACE_Z = 0.20


def asset_path(path: Path) -> Sdf.AssetPath:
    """Return an asset path relative to assets/scenes/hd_street."""
    return Sdf.AssetPath(Path("../../../").joinpath(path.relative_to(ROOT)).as_posix())


def solid_material(stage: Usd.Stage, name: str, color, roughness=0.7, metallic=0.0):
    material = UsdShade.Material.Define(stage, f"/HDStreet/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/HDStreet/Looks/{name}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def pbr_material(
    stage: Usd.Stage, name: str, folder: Path, names: tuple[str, str, str], scale,
    use_normal: bool = True,
):
    material = UsdShade.Material.Define(stage, f"/HDStreet/Looks/{name}")
    surface = UsdShade.Shader.Define(stage, f"/HDStreet/Looks/{name}/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    reader = UsdShade.Shader.Define(stage, f"/HDStreet/Looks/{name}/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    transform = UsdShade.Shader.Define(stage, f"/HDStreet/Looks/{name}/UVTransform")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*scale))
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    for channel, filename in zip(("diffuse", "rough", "normal"), names, strict=True):
        texture = UsdShade.Shader.Define(stage, f"/HDStreet/Looks/{name}/{channel.title()}Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(asset_path(folder / filename))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(transform.ConnectableAPI(), "result")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if channel == "diffuse":
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                texture.ConnectableAPI(), "rgb"
            )
        elif channel == "rough":
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
            surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
                texture.ConnectableAPI(), "r"
            )
        else:
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            if use_normal:
                surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                    texture.ConnectableAPI(), "rgb"
                )
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def set_role(prim: Usd.Prim, role: str) -> None:
    prim.CreateAttribute("track:role", Sdf.ValueTypeNames.String).Set(role)


def box_mesh(stage: Usd.Stage, path: str, center, size, material, role: str, collision=False):
    sx, sy, sz = (value * 0.5 for value in size)
    # USD/PhysX derives collision-face direction from vertex winding, not from
    # the authored display normals below.  The original tuples were clockwise
    # when viewed from outside, so the road top was a one-sided face pointing
    # down and Go2 feet fell through it.  Reverse every face to right-handed,
    # outward counter-clockwise winding.
    inward_faces = (
        ((-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz)),
        ((-sx, sy, sz), (sx, sy, sz), (sx, -sy, sz), (-sx, -sy, sz)),
        ((-sx, -sy, sz), (sx, -sy, sz), (sx, -sy, -sz), (-sx, -sy, -sz)),
        ((sx, -sy, sz), (sx, sy, sz), (sx, sy, -sz), (sx, -sy, -sz)),
        ((sx, sy, sz), (-sx, sy, sz), (-sx, sy, -sz), (sx, sy, -sz)),
        ((-sx, sy, sz), (-sx, -sy, sz), (-sx, -sy, -sz), (-sx, sy, -sz)),
    )
    faces = tuple(tuple(reversed(face)) for face in inward_faces)
    normals = (
        (0, 0, -1), (0, 0, 1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)
    )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for face in faces for point in face])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(list(range(24)))
    mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in normals for _ in range(4)])
    mesh.SetNormalsInterpolation("faceVarying")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set([Gf.Vec2f(*uv) for _ in faces for uv in ((0, 0), (1, 0), (1, 1), (0, 1))])
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    set_role(mesh.GetPrim(), role)
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def cube(stage: Usd.Stage, path: str, center, size, material, role: str, collision=False):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(shape)
    xform.SetTranslate(Gf.Vec3d(*center))
    xform.SetScale(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
    set_role(shape.GetPrim(), role)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def cylinder(stage, path, center, radius, height, material, role, collision=False):
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateAxisAttr("Z")
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(height)
    UsdGeom.XformCommonAPI(shape).SetTranslate(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
    set_role(shape.GetPrim(), role)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def add_building(stage, name, center_x, side, width, depth, height, wall, palette):
    y = side * (7.2 + depth * 0.5)
    root = stage.DefinePrim(f"/HDStreet/Architecture/{name}", "Xform")
    set_role(root, "architecture")
    box_mesh(stage, f"{root.GetPath()}/Shell", (center_x, y, height * 0.5 + 0.36),
             (width, depth, height), wall, "building_shell", collision=False)
    cube(stage, f"{root.GetPath()}/Roof", (center_x, y, height + 0.48),
         (width + 0.28, depth + 0.28, 0.24), palette["trim"], "roof")

    facade_y = side * 7.18
    floors = max(2, int((height - 1.5) // 2.6))
    columns = max(2, int((width - 1.0) // 2.1))
    spacing_x = width / (columns + 1)
    for floor in range(floors):
        z = 2.25 + floor * 2.45
        if z + 0.75 > height + 0.25:
            continue
        for column in range(columns):
            x = center_x - width * 0.5 + spacing_x * (column + 1)
            glass = cube(stage, f"{root.GetPath()}/Windows/F{floor}_C{column}",
                         (x, facade_y, z), (1.02, 0.055, 1.32), palette["glass"], "window")
            UsdGeom.XformCommonAPI(glass).SetRotate(Gf.Vec3f(0, 0, 0))
            for suffix, dx, dz, frame_size in (
                ("L", -0.55, 0, (0.07, 0.09, 1.46)),
                ("R", 0.55, 0, (0.07, 0.09, 1.46)),
                ("T", 0, 0.70, (1.16, 0.09, 0.07)),
                ("B", 0, -0.70, (1.16, 0.09, 0.07)),
            ):
                cube(stage, f"{root.GetPath()}/WindowFrames/F{floor}_C{column}_{suffix}",
                     (x + dx, facade_y - side * 0.025, z + dz), frame_size,
                     palette["frame"], "window_frame")

    door_x = center_x - width * 0.25
    cube(stage, f"{root.GetPath()}/Entrance/Door", (door_x, facade_y - side * 0.02, 1.35),
         (1.35, 0.08, 2.25), palette["glass"], "door")
    cube(stage, f"{root.GetPath()}/Entrance/Canopy", (door_x, facade_y - side * 0.65, 2.75),
         (2.1, 1.25, 0.16), palette["trim"], "canopy")
    # Simple low-poly building collision instead of per-window triangle collision.
    cube(stage, f"/HDStreet/Collision/{name}", (center_x, y, height * 0.5 + 0.36),
         (width, depth, height), palette["collision"], "building_collision", collision=True)
    UsdGeom.Imageable(stage.GetPrimAtPath(f"/HDStreet/Collision/{name}")).MakeInvisible()


def add_model(stage, name, model_id, position, yaw=0.0, scale=1.0, proxy_size=None):
    root = stage.DefinePrim(f"/HDStreet/Props/{name}", "Xform")
    root.GetReferences().AddReference(asset_path(
        POLYHAVEN / f"models/{model_id}/{model_id}_2k.usdc"
    ).path)
    transform = UsdGeom.XformCommonAPI(root)
    transform.SetTranslate(Gf.Vec3d(*position))
    transform.SetRotate(Gf.Vec3f(0, 0, yaw), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    transform.SetScale(Gf.Vec3f(scale, scale, scale))
    set_role(root, "high_quality_obstacle")
    root.CreateAttribute("asset:source", Sdf.ValueTypeNames.String).Set("Poly Haven")
    root.CreateAttribute("asset:license", Sdf.ValueTypeNames.String).Set("CC0")
    if proxy_size:
        sx, sy, sz = proxy_size
        proxy = cube(stage, f"/HDStreet/Collision/{name}",
                     (position[0], position[1], SURFACE_Z + sz * 0.5),
                     (sx, sy, sz), palette["collision"], "obstacle_collision", collision=True)
        UsdGeom.Imageable(proxy).MakeInvisible()


def build() -> None:
    global palette
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(30.0)
    root = UsdGeom.Xform.Define(stage, "/HDStreet")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().customLayerData = {
        "assetSource": "Local procedural layout + Poly Haven CC0 assets",
        "purpose": "Compact close-range outdoor human tracking",
        "surfaceZMeters": SURFACE_Z,
        "polyHavenCredit": "Powered by Poly Haven — https://polyhaven.com",
    }
    for scope in ("Looks", "Ground", "Architecture", "StreetFurniture", "Props", "Collision", "Route", "Lighting"):
        UsdGeom.Scope.Define(stage, f"/HDStreet/{scope}")

    asphalt_dir = POLYHAVEN / "textures/asphalt_02"
    paving_dir = POLYHAVEN / "textures/brick_pavement_02"
    brick_dir = POLYHAVEN / "textures/brick_wall_001"
    beige_dir = POLYHAVEN / "textures/beige_wall_001"
    asphalt = pbr_material(stage, "Asphalt2K", asphalt_dir,
                            ("asphalt_02_diff_2k.jpg", "asphalt_02_rough_2k.jpg", "asphalt_02_nor_gl_2k.jpg"), (14, 2.5))
    paving = pbr_material(stage, "BrickPaving2K", paving_dir,
                           ("brick_pavement_02_diff_2k.jpg", "brick_pavement_02_rough_2k.jpg", "brick_pavement_02_nor_gl_2k.jpg"), (16, 2.5))
    brick = pbr_material(stage, "BrickWall2K", brick_dir,
                          ("brick_wall_001_diffuse_2k.jpg", "brick_wall_001_rough_2k.jpg", "brick_wall_001_nor_gl_2k.jpg"), (4, 4))
    beige = pbr_material(stage, "BeigeWall2K", beige_dir,
                          ("beige_wall_001_diff_2k.jpg", "beige_wall_001_rough_2k.jpg", "beige_wall_001_nor_gl_2k.jpg"), (4, 4))
    palette = {
        "curb": solid_material(stage, "CurbConcrete", (0.34, 0.35, 0.34), 0.9),
        "line": solid_material(stage, "RoadMarking", (0.91, 0.86, 0.68), 0.48),
        "glass": solid_material(stage, "WindowGlass", (0.055, 0.16, 0.22), 0.14, 0.12),
        "frame": solid_material(stage, "DarkFrame", (0.035, 0.045, 0.052), 0.42, 0.35),
        "trim": solid_material(stage, "WarmTrim", (0.28, 0.19, 0.13), 0.62),
        "metal": solid_material(stage, "StreetMetal", (0.075, 0.085, 0.09), 0.34, 0.72),
        "lamp": solid_material(stage, "LampLens", (1.0, 0.72, 0.37), 0.18),
        "green": solid_material(stage, "UrbanGreen", (0.045, 0.19, 0.065), 0.82),
        "collision": solid_material(stage, "CollisionDebug", (1.0, 0.0, 0.8), 1.0),
    }

    # 56 m task corridor; the top face remains at z=0.20 for existing robot/people logic.
    box_mesh(stage, "/HDStreet/Ground/Road", (24.0, 0.0, 0.10), (64.0, 9.0, 0.20),
             asphalt, "drivable_surface", collision=True)
    for side, y in (("South", -5.8), ("North", 5.8)):
        box_mesh(stage, f"/HDStreet/Ground/Sidewalk{side}", (24.0, y, 0.18),
                 (64.0, 2.6, 0.36), paving, "walkable_surface", collision=True)
        cube(stage, f"/HDStreet/Ground/Curb{side}", (24.0, -4.55 if side == "South" else 4.55, 0.27),
             (64.0, 0.22, 0.54), palette["curb"], "curb", collision=True)
    for x in range(-4, 57, 6):
        cube(stage, f"/HDStreet/Ground/CenterLine_{x + 4:02d}", (x + 1.7, 0.0, 0.212),
             (3.0, 0.11, 0.018), palette["line"], "road_marking")
    for x in (2.0, 2.65, 3.3, 3.95, 4.6, 5.25):
        cube(stage, f"/HDStreet/Ground/Crosswalk_{int(x * 100)}", (x, 0.0, 0.216),
             (0.36, 7.4, 0.022), palette["line"], "road_marking")

    buildings = (
        ("SouthLoft", 2.0, -1, 10.0, 5.2, 12.0, brick),
        ("SouthStudio", 14.5, -1, 11.0, 5.8, 15.0, beige),
        ("SouthCorner", 28.0, -1, 12.0, 5.5, 10.5, brick),
        ("SouthOffice", 43.0, -1, 14.0, 6.0, 16.0, beige),
        ("NorthBrick", 1.5, 1, 11.0, 5.5, 14.0, brick),
        ("NorthGallery", 15.0, 1, 10.5, 5.0, 10.5, beige),
        ("NorthApartments", 28.5, 1, 12.0, 6.0, 16.5, brick),
        ("NorthCivic", 44.0, 1, 15.0, 5.5, 12.5, beige),
    )
    for record in buildings:
        add_building(stage, *record, palette)

    # Lamps and small greenery give close-range parallax and scale cues.
    for side, y in (("S", -5.05), ("N", 5.05)):
        for index, x in enumerate((-2.0, 8.0, 20.0, 32.0, 44.0, 56.0)):
            base = f"/HDStreet/StreetFurniture/Lamp_{side}_{index}"
            cylinder(stage, f"{base}/Pole", (x, y, 2.38), 0.065, 4.25,
                     palette["metal"], "street_light", collision=False)
            cube(stage, f"{base}/Arm", (x, y - (0.36 if side == "S" else -0.36), 4.44),
                 (0.10, 0.72, 0.10), palette["metal"], "street_light")
            cube(stage, f"{base}/Lens", (x, y - (0.68 if side == "S" else -0.68), 4.38),
                 (0.26, 0.40, 0.12), palette["lamp"], "street_light")
    for index, (x, y) in enumerate(((7.5, -6.0), (25.0, 6.0), (39.0, -6.0), (51.0, 6.0))):
        cylinder(stage, f"/HDStreet/StreetFurniture/Tree_{index}/Trunk", (x, y, 1.15),
                 0.17, 2.0, palette["trim"], "vegetation")
        crown = UsdGeom.Sphere.Define(stage, f"/HDStreet/StreetFurniture/Tree_{index}/Crown")
        crown.CreateRadiusAttr(1.15)
        UsdGeom.XformCommonAPI(crown).SetTranslate(Gf.Vec3d(x, y, 2.75))
        UsdShade.MaterialBindingAPI.Apply(crown.GetPrim()).Bind(palette["green"])
        set_role(crown.GetPrim(), "vegetation")

    add_model(stage, "Barrier", "concrete_road_barrier", (10.0, 3.15, SURFACE_Z), 0, 1.15, (1.8, 0.8, 1.0))
    add_model(stage, "CoveredCar", "covered_car", (21.0, 3.0, SURFACE_Z), 90, 1.0, (4.5, 2.0, 1.5))
    add_model(stage, "Hydrant", "fire_hydrant", (31.0, 5.55, 0.36), 0, 1.0, (0.8, 0.8, 0.9))
    add_model(stage, "TrashCan", "metal_trash_can", (37.0, -5.75, 0.36), 0, 0.9, (1.8, 0.7, 0.9))
    add_model(stage, "Bench", "painted_wooden_bench", (15.0, -5.85, 0.36), 0, 1.55, (1.9, 0.8, 1.4))
    add_model(stage, "Planter", "planter_box_01", (45.0, -5.7, 0.36), 0, 1.4, (1.4, 0.9, 0.7))

    for index, (x, y) in enumerate(((-3.0, 0.0), (0.0, 0.0), (8.0, 0.0), (20.0, 0.0), (36.0, 0.0), (52.0, 0.0))):
        waypoint = UsdGeom.Xform.Define(stage, f"/HDStreet/Route/Waypoint_{index:02d}")
        UsdGeom.XformCommonAPI(waypoint).SetTranslate(Gf.Vec3d(x, y, SURFACE_Z))
        set_role(waypoint.GetPrim(), "route_waypoint")
        waypoint.GetPrim().CreateAttribute("route:index", Sdf.ValueTypeNames.Int).Set(index)

    dome = UsdLux.DomeLight.Define(stage, "/HDStreet/Lighting/UrbanSky")
    dome.CreateTextureFileAttr(asset_path(POLYHAVEN / "hdris/urban_street_01/urban_street_01_2k.exr"))
    dome.CreateIntensityAttr(450.0)
    dome.CreateExposureAttr(-2.0)
    dome.CreateTextureFormatAttr("latlong")
    UsdLux.ShadowAPI.Apply(dome.GetPrim()).CreateShadowEnableAttr(False)
    sun = UsdLux.DistantLight.Define(stage, "/HDStreet/Lighting/Sun")
    sun.CreateIntensityAttr(3200.0)
    # A narrow sun produces stable real-time shadows; large angular lights are
    # intentionally avoided because their Monte-Carlo soft shadows caused grain.
    sun.CreateAngleAttr(0.12)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.80))
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(48.0, -28.0, 24.0))

    stage.GetRootLayer().Save()
    print("HD_STREET_BUILD_OK", OUTPUT)


if __name__ == "__main__":
    raise SystemExit("Run scripts/build_hd_route_city.py; the straight-street baseline was removed.")
