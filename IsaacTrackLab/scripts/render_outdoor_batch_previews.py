#!/usr/bin/env python3
"""Render one RTX acceptance view and one route plan for every outdoor map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", default="outputs/outdoor_batch/previews")
parser.add_argument("--map", action="append", dest="map_ids")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args, headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pxr import UsdGeom, UsdLux
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

from isaac_track_lab.task.terrain_routes import terrain_height


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/outdoor_map_batch.json").read_text(encoding="utf-8"))
SCENE_ROOT = ROOT / "assets/scenes/outdoor_batch"
MAP_IDS = [item["id"] for item in CONFIG["maps"]]
OFFSETS = {map_id: index * 240.0 for index, map_id in enumerate(MAP_IDS)}

RENDERER = IsaacRtxRendererCfg(
    enable_scene_partitioning=False,
    global_settings=IsaacRtxRendererGlobalSettingsCfg(
        antialiasing_mode="TAA", enable_dl_denoiser=True, samples_per_pixel=16,
        enable_direct_lighting=True, enable_global_illumination=False,
        enable_reflections=True, enable_shadows=True, enable_ambient_occlusion=True,
        ambient_light_intensity=0.8,
        carb_settings={
            "/rtx/rendermode": "RaytracedLighting",
            "/rtx/directLighting/sampledLighting/enabled": False,
            "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
        },
    ),
)


def asset_cfg(map_id: str) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"/World/OutdoorMaps/{map_id}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_ROOT / map_id / f"{map_id}.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(OFFSETS[map_id], 0.0, 0.0)),
    )


@configclass
class OutdoorBatchPreviewCfg(InteractiveSceneCfg):
    forest_trail = asset_cfg("forest_trail")
    rocky_canyon = asset_cfg("rocky_canyon")
    alpine_slope = asset_cfg("alpine_slope")
    wetland_boardwalk = asset_cfg("wetland_boardwalk")
    desert_dunes = asset_cfg("desert_dunes")
    coastal_scrub = asset_cfg("coastal_scrub")
    camera = CameraCfg(
        prim_path="/World/PreviewCamera", update_period=0.0, height=720, width=1280,
        data_types=["rgb"], renderer_cfg=RENDERER,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=31.0, horizontal_aperture=36.0,
            focus_distance=90.0, clipping_range=(0.1, 1500.0),
        ),
        offset=CameraCfg.OffsetCfg(),
    )


def render_plan(map_config: dict, out_dir: Path) -> None:
    map_dir = SCENE_ROOT / map_config["id"]
    manifest = json.loads((map_dir / "scene_manifest.json").read_text(encoding="utf-8"))
    routes = json.loads((map_dir / "routes/generated_routes.json").read_text(encoding="utf-8"))
    image = np.full((1000, 1000, 3), (229, 226, 214), dtype=np.uint8)
    xmin, xmax, ymin, ymax = map(float, map_config["bounds_m"])

    def pixel(point):
        x_m, y_m = point[:2]
        return int(50 + (x_m-xmin)/(xmax-xmin)*900), int(950 - (y_m-ymin)/(ymax-ymin)*900)

    for zone in map_config["challenge_zones"]:
        cx, cy = zone["center_m"]
        width, height = zone["size_m"]
        p1, p2 = pixel((cx-width/2, cy-height/2)), pixel((cx+width/2, cy+height/2))
        color = (145, 86, 38) if not zone["traversable"] else (105, 156, 118)
        cv2.rectangle(image, (p1[0], p2[1]), (p2[0], p1[1]), color, -1)
        cv2.putText(image, zone["id"], (min(p1[0], p2[0])+3, min(p1[1], p2[1])+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245,245,245), 1, cv2.LINE_AA)
    for item in manifest["obstacles"]:
        radius_px = max(2, round(float(item["radius_m"]) / (xmax-xmin) * 900))
        cv2.circle(image, pixel((item["x_m"], item["y_m"])), radius_px, (48, 65, 46), -1, cv2.LINE_AA)
    colors = ((42,65,225), (235,112,29), (48,173,72), (184,55,168))
    for index, route in enumerate(routes["routes"][:4]):
        line = np.asarray([pixel(point) for point in route["points_m"]], dtype=np.int32)
        cv2.polylines(image, [line], False, colors[index], 3, cv2.LINE_AA)
    cv2.putText(image, f"{map_config['id']} | obstacles {manifest['obstacle_count']} | first 4 seeded routes",
                (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (24,30,34), 2, cv2.LINE_AA)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(out_dir / f"{map_config['id']}_plan.png")


def render_contact_sheet(selected: list[dict], out_dir: Path, suffix: str) -> None:
    cells = []
    for item in selected:
        source = Image.open(out_dir / f"{item['id']}_{suffix}.png").convert("RGB")
        source.thumbnail((640, 360))
        cell = Image.new("RGB", (640, 400), (242, 242, 238))
        cell.paste(source, ((640-source.width)//2, 40+(360-source.height)//2))
        ImageDraw.Draw(cell).text((14, 12), f"{item['id']} | {item['label']}", fill=(18, 24, 30))
        cells.append(cell)
    sheet = Image.new("RGB", (1280, 1200), (242, 242, 238))
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % 2) * 640, (index // 2) * 400))
    sheet.save(out_dir / f"{suffix}_contact_sheet.png")


def main() -> None:
    selected = [item for item in CONFIG["maps"] if not args.map_ids or item["id"] in args.map_ids]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in selected:
        render_plan(item, out_dir)
    render_contact_sheet(selected, out_dir, "plan")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0/30.0, device=args.device))
    scene = InteractiveScene(OutdoorBatchPreviewCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    camera = scene["camera"]
    stage = omni.usd.get_context().get_stage()
    for item in selected:
        for map_id in MAP_IDS:
            visible = map_id == item["id"]
            root_prim = stage.GetPrimAtPath(f"/World/OutdoorMaps/{map_id}")
            UsdGeom.Imageable(root_prim).GetVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
            )
            UsdLux.DomeLight(stage.GetPrimAtPath(
                f"/World/OutdoorMaps/{map_id}/Lighting/Sky"
            )).GetIntensityAttr().Set(450.0 if visible else 0.0)
            UsdLux.DistantLight(stage.GetPrimAtPath(
                f"/World/OutdoorMaps/{map_id}/Lighting/Sun"
            )).GetIntensityAttr().Set(1200.0 if visible else 0.0)
        offset = OFFSETS[item["id"]]
        eye_local = (-70.0, -62.0)
        look_local = (-40.0, -18.0)
        eye = (offset + eye_local[0], eye_local[1], terrain_height(item, *eye_local) + 2.4)
        look = (offset + look_local[0], look_local[1], terrain_height(item, *look_local) + 1.1)
        camera.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=scene.device),
            torch.tensor([look], dtype=torch.float32, device=scene.device),
        )
        for _ in range(72):
            scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
        rgb = np.asarray(camera.data.output["rgb"].torch[0, :, :, :3].cpu(), dtype=np.uint8)
        Image.fromarray(cv2.medianBlur(rgb, 3)).save(out_dir / f"{item['id']}_rtx.png")
        print("OUTDOOR_PREVIEW_OK", item["id"], flush=True)
    render_contact_sheet(selected, out_dir, "rtx")
    print("OUTDOOR_BATCH_PREVIEWS_OK", out_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
