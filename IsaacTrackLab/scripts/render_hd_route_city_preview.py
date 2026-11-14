#!/usr/bin/env python3
"""Render visual acceptance views of the intersecting 305 m HD task city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", default="outputs/hd_route_city_preview")
parser.add_argument("--quick", action="store_true", help="Render only route start and route plan.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args, headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg


ROOT = Path(__file__).resolve().parents[1]
CITY_USD = ROOT / "assets/scenes/hd_route_city/hd_route_city.usda"
ROUTE_CONFIG = ROOT / "configs/hd_route_city_route.json"
TARGET_USD = ROOT / "assets/external/nvidia/people/F_Business_02.usd"

RENDERER = IsaacRtxRendererCfg(
    enable_scene_partitioning=False,
    global_settings=IsaacRtxRendererGlobalSettingsCfg(
        antialiasing_mode="TAA", enable_dl_denoiser=True, samples_per_pixel=4,
        enable_direct_lighting=True, enable_global_illumination=False,
        enable_reflections=False, enable_shadows=True, enable_ambient_occlusion=False,
        ambient_light_intensity=1.0,
        carb_settings={
            "/rtx/rendermode": "RaytracedLighting",
            "/rtx/directLighting/sampledLighting/enabled": False,
            "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
        },
    ),
)


@configclass
class HdRouteCityPreviewCfg(InteractiveSceneCfg):
    city = AssetBaseCfg(
        prim_path="/World/HDStreet",
        spawn=sim_utils.UsdFileCfg(usd_path=str(CITY_USD)),
    )
    target = AssetBaseCfg(
        prim_path="/World/TargetHuman",
        spawn=sim_utils.UsdFileCfg(usd_path=str(TARGET_USD)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-52.0, -36.0, 0.20)),
    )
    camera = CameraCfg(
        prim_path="/World/PreviewCamera", update_period=0.0, height=720, width=1280,
        data_types=["rgb"], renderer_cfg=RENDERER,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=36.0,
            focus_distance=30.0, clipping_range=(0.1, 350.0),
        ),
        offset=CameraCfg.OffsetCfg(),
    )


def render_view(scene, sim, name: str, eye, look, out_dir: Path) -> None:
    camera = scene["camera"]
    camera.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=scene.device),
        torch.tensor([look], dtype=torch.float32, device=scene.device),
    )
    for _ in range(48):
        scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
    rgb = np.asarray(camera.data.output["rgb"].torch[0, :, :, :3].cpu(), dtype=np.uint8)
    # The local driver cannot initialize DLSS/NGX, so use a 5x5 spatial
    # firefly pass for human-facing acceptance images. Geometry remains intact.
    Image.fromarray(cv2.medianBlur(rgb, 5)).save(out_dir / f"{name}.png")
    print("PREVIEW_VIEW_OK", name, flush=True)


def render_route_plan(out_dir: Path) -> None:
    config = json.loads(ROUTE_CONFIG.read_text(encoding="utf-8"))
    points = np.asarray([point[:2] for point in config["waypoints_m"]], dtype=np.float64)
    image = np.full((900, 1200, 3), (238, 239, 234), dtype=np.uint8)
    x_min, x_max, y_min, y_max = -60.0, 60.0, -52.0, 48.0

    def pixel(point):
        x, y = point
        return int((x - x_min) / (x_max - x_min) * 1100 + 50), int(850 - (y - y_min) / (y_max - y_min) * 800)

    for y in (-36.0, -12.0, 12.0):
        cv2.line(image, pixel((x_min, y)), pixel((x_max, y)), (82, 86, 91), 58, cv2.LINE_AA)
    for x in (-36.0, 0.0, 48.0):
        cv2.line(image, pixel((x, y_min)), pixel((x, y_max)), (82, 86, 91), 58, cv2.LINE_AA)
    route_pixels = np.asarray([pixel(point) for point in points], dtype=np.int32)
    cv2.polylines(image, [route_pixels], False, (44, 78, 232), 9, cv2.LINE_AA)
    for point in ((-18.0, -36.0), (48.0, -23.0), (6.0, -12.0)):
        cv2.circle(image, pixel(point), 14, (35, 35, 220), -1, cv2.LINE_AA)
    cv2.circle(image, tuple(route_pixels[0]), 15, (45, 170, 45), -1, cv2.LINE_AA)
    cv2.circle(image, tuple(route_pixels[-1]), 15, (20, 145, 235), -1, cv2.LINE_AA)
    cv2.putText(image, "START", (route_pixels[0, 0] + 18, route_pixels[0, 1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 90, 20), 2, cv2.LINE_AA)
    cv2.putText(image, "END", (route_pixels[-1, 0] - 75, route_pixels[-1, 1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 90, 180), 2, cv2.LINE_AA)
    cv2.putText(image, f"Route {config['route_length_m']:.1f} m | target 1.25 m/s | 4 turns | 3 obstacle deviations",
                (50, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 35, 42), 2, cv2.LINE_AA)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(out_dir / "route_plan.png")
    print("ROUTE_PLAN_OK", flush=True)


def main() -> None:
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    render_route_plan(out_dir)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 30.0, device=args.device))
    scene = InteractiveScene(HdRouteCityPreviewCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    views = {
        "route_start": ((-58.0, -41.0, 3.0), (-43.0, -35.5, 1.3)),
        "first_turn": ((40.0, -43.0, 5.5), (48.0, -29.0, 1.1)),
        "covered_car_deviation": ((44.0, -31.0, 3.2), (48.0, -22.0, 1.1)),
        "middle_cross_street": ((14.0, -18.5, 5.0), (0.0, -11.5, 1.0)),
        "aerial_city": ((0.0, -76.0, 82.0), (0.0, -5.0, 0.0)),
    }
    for name, (eye, look) in views.items():
        if args.quick and name != "route_start":
            continue
        render_view(scene, sim, name, eye, look, out_dir)
    print("HD_ROUTE_CITY_PREVIEW_OK", out_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
