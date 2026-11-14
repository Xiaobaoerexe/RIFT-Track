#!/usr/bin/env python3
"""Render a live Isaac Sim preview of the terrain-grounded target route motion."""

from __future__ import annotations

import argparse
import json
from math import atan2, cos, degrees, hypot, pi, sin
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
CATALOG = json.loads((ROOT / "configs/terrain_dataset_v3_realistic.json").read_text())
MAPS = {item["id"]: item for item in CATALOG["maps"]}

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--map", dest="map_id", choices=sorted(MAPS), default="park_01")
parser.add_argument("--route", type=int, default=1)
parser.add_argument("--duration-s", type=float, default=8.0)
parser.add_argument("--video-fps", type=float, default=15.0)
parser.add_argument("--render-samples", type=int, default=1)
parser.add_argument("--target-speed-mps", type=float, default=1.25)
parser.add_argument("--start-distance-m", type=float, help="default: begin 5 m before a route turn")
parser.add_argument("--output-dir", default="outputs/terrain_dataset_v3_realistic/live_target")
parser.add_argument("--gui", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.route <= 20:
    parser.error("--route must be in 1..20")
if args.duration_s <= 0.0 or args.video_fps <= 0.0:
    parser.error("duration and video fps must be positive")
if args.render_samples < 1:
    parser.error("render samples must be at least one")
if not 1.0 <= args.target_speed_mps <= 1.5:
    parser.error("target speed must remain in 1.0..1.5 m/s")

app_launcher = AppLauncher(args, headless=not args.gui, enable_cameras=True)
simulation_app = app_launcher.app

import cv2
import numpy as np
import omni.kit.commands
import omni.usd
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSkel

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
from isaaclab_physx.sim.views import FabricFrameView

from isaac_track_lab.task import TerrainTargetFollower


DT = 1.0 / 30.0
WIDTH, HEIGHT = 1280, 720
RENDER_WIDTH, RENDER_HEIGHT = 1920, 1080
MAP_CONFIG = MAPS[args.map_id]
MAP_DIR = ROOT / "assets/scenes/terrain_dataset_v3_realistic" / args.map_id
MANIFEST = json.loads((MAP_DIR / "scene_manifest.json").read_text())
ROUTES = json.loads((MAP_DIR / "routes/generated_routes.json").read_text())["routes"]
ROUTE = ROUTES[args.route - 1]
CHARACTER_USD = ROOT / "assets/external/nvidia/people/F_Business_02.usd"
WALK_SEQUENCE = ROOT / "assets/external/nvidia/people/behavior_walk_sequence.json"

RENDERER = IsaacRtxRendererCfg(
    enable_scene_partitioning=False,
    global_settings=IsaacRtxRendererGlobalSettingsCfg(
        antialiasing_mode="TAA",
        enable_dl_denoiser=True,
        samples_per_pixel=16,
        enable_direct_lighting=True,
        enable_global_illumination=False,
        enable_reflections=True,
        enable_shadows=True,
        enable_ambient_occlusion=True,
        ambient_light_intensity=0.8,
        carb_settings={
            "/rtx/rendermode": "RaytracedLighting",
            # The current driver reports DLSS-RR denoising as unavailable.
            # Sampled lighting without that denoiser creates severe salt-and-
            # pepper noise in moving video, so use the stable direct-lighting
            # path for this acceptance recording.
            "/rtx/directLighting/sampledLighting/enabled": False,
            "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
        },
    ),
)


@configclass
class LiveTargetPreviewCfg(InteractiveSceneCfg):
    target_collider: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/TargetHumanCollider", spawn=None,
    )
    camera = CameraCfg(
        prim_path="/World/LiveTargetCamera",
        update_period=1.0 / args.video_fps,
        height=RENDER_HEIGHT,
        width=RENDER_WIDTH,
        data_types=["rgb"],
        renderer_cfg=RENDERER,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.0,
            horizontal_aperture=36.0,
            focus_distance=12.0,
            clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(),
    )


def pump_loading(timeout_s: float = 180.0) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        simulation_app.update()
        status = omni.usd.get_context().get_stage_loading_status()
        if len(status) >= 3 and status[2] == 0:
            return
    raise TimeoutError("USD loading did not finish")


def find_skelroot(stage: Usd.Stage, prefix: str) -> Usd.Prim:
    return next(
        prim for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(Sdf.Path(prefix)) and prim.IsA(UsdSkel.Root)
    )


def create_walk_cycle(stage: Usd.Stage, skelroot: Usd.Prim) -> dict:
    skeleton_prim = next(
        prim for prim in Usd.PrimRange(skelroot)
        if prim.GetPath().HasPrefix(skelroot.GetPath()) and prim.IsA(UsdSkel.Skeleton)
    )
    skeleton = UsdSkel.Skeleton(skeleton_prim)
    joints = list(skeleton.GetJointsAttr().Get())
    sequence = json.loads(WALK_SEQUENCE.read_text())
    if sequence["joints"] != [str(joint) for joint in joints]:
        raise RuntimeError("captured walk sequence does not match the target skeleton")
    animation_path = skeleton_prim.GetPath().AppendChild("TerrainRouteWalk")
    animation = UsdSkel.Animation.Define(stage, animation_path)
    animation.CreateJointsAttr().Set(joints)
    UsdSkel.BindingAPI.Apply(skeleton_prim).CreateAnimationSourceRel().SetTargets([animation_path])
    controller = {
        "frames": sequence["frames"][:29],
        "fps": float(sequence.get("fps", 30.0)),
        "playback_rate": args.target_speed_mps / 0.82,
        "translations": animation.CreateTranslationsAttr(),
        "rotations": animation.CreateRotationsAttr(),
        "scales": animation.CreateScalesAttr(),
        "last_index": None,
    }
    update_walk_cycle(controller, 0.0)
    return controller


def update_walk_cycle(controller: dict, elapsed_s: float) -> None:
    index = int(elapsed_s * controller["fps"] * controller["playback_rate"]) % len(controller["frames"])
    if index == controller["last_index"]:
        return
    pose = controller["frames"][index]
    controller["translations"].Set([Gf.Vec3f(*value) for value in pose["translations"]])
    controller["rotations"].Set(
        [Gf.Quatf(value[0], Gf.Vec3f(*value[1:])) for value in pose["rotations"]]
    )
    controller["scales"].Set([Gf.Vec3h(*value) for value in pose["scales"]])
    controller["last_index"] = index


def automatic_start_distance(points: list[list[float]]) -> float:
    cumulative = [0.0]
    lengths = []
    for left, right in zip(points, points[1:]):
        length = hypot(right[0] - left[0], right[1] - left[1])
        lengths.append(length)
        cumulative.append(cumulative[-1] + length)
    for index in range(1, len(points) - 1):
        incoming = atan2(points[index][1] - points[index - 1][1], points[index][0] - points[index - 1][0])
        outgoing = atan2(points[index + 1][1] - points[index][1], points[index + 1][0] - points[index][0])
        turn = abs((outgoing - incoming + pi) % (2.0 * pi) - pi)
        if degrees(turn) >= 20.0 and cumulative[index] >= 5.0:
            return cumulative[index] - 5.0
    return 0.0


def create_target_collider(stage: Usd.Stage) -> None:
    capsule = UsdGeom.Capsule.Define(stage, "/World/TargetHumanCollider")
    capsule.CreateAxisAttr("Z")
    capsule.CreateRadiusAttr(0.28)
    capsule.CreateHeightAttr(1.25)
    capsule.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    capsule.GetPrim().CreateAttribute("track:role", Sdf.ValueTypeNames.String).Set(
        "target_human_collision"
    )
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
    body.CreateKinematicEnabledAttr(True)


def main() -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.map_id}_route_{args.route:02d}_live_target"
    raw_video = output_dir / f"{stem}.mjpeg.avi"
    final_video = output_dir / f"{stem}.mp4"
    summary_path = output_dir / f"{stem}.json"

    stage = omni.usd.get_context().get_stage()
    map_prim = stage.DefinePrim("/World/Map", "Xform")
    map_prim.GetReferences().AddReference(str(MAP_DIR / MANIFEST["usd"]))
    omni.kit.commands.execute(
        "CreatePayloadCommand", usd_context=omni.usd.get_context(),
        path_to="/World/TargetHuman", asset_path=str(CHARACTER_USD),
        prim_path=Sdf.Path.emptyPath, instanceable=False, select_prim=False,
    )
    pump_loading()
    stage.SetTimeCodesPerSecond(30.0)
    terrain_prim = stage.GetPrimAtPath("/World/Map/Terrain/Heightfield")
    surface_geometry_mode = terrain_prim.GetAttribute("surface:geometryMode").Get()
    if surface_geometry_mode != "terrain_face_material_subsets":
        raise RuntimeError(f"unexpected semantic surface geometry mode: {surface_geometry_mode}")
    controller = create_walk_cycle(stage, find_skelroot(stage, "/World/TargetHuman"))
    create_target_collider(stage)

    follower = TerrainTargetFollower(
        MAP_CONFIG, ROUTE["points_m"], nominal_speed_mps=args.target_speed_mps,
        minimum_speed_mps=1.0, maximum_speed_mps=1.5,
        terrain_grid_spacing_m=float(MANIFEST["terrain_grid_spacing_m"]),
    )
    start_distance = (
        float(args.start_distance_m) if args.start_distance_m is not None
        else automatic_start_distance(ROUTE["points_m"])
    )
    start_distance = max(0.0, min(start_distance, follower.total_length_m))
    pose = follower.reset(start_distance)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    scene = InteractiveScene(LiveTargetPreviewCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    target_visual = FabricFrameView("/World/TargetHuman", device=args.device)
    target_visual.get_world_poses()
    target_collider = scene["target_collider"]
    camera = scene["camera"]

    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"MJPG"), args.video_fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {raw_video}")
    output_steps = max(1, round((1.0 / args.video_fps) / DT))
    speeds, yaw_rates, grounding_errors = [], [], []
    previous_yaw = pose.yaw_deg
    frames = 0

    for step in range(round(args.duration_s / DT)):
        elapsed_s = step * DT
        pose = follower.step(DT)
        yaw_radians = np.deg2rad(pose.yaw_deg + 90.0)
        visual_position = torch.tensor([pose.position_m], dtype=torch.float32, device=scene.device)
        visual_orientation = torch.tensor(
            [[0.0, 0.0, sin(0.5 * yaw_radians), cos(0.5 * yaw_radians)]],
            dtype=torch.float32, device=scene.device,
        )
        with target_visual.xform_world_space_writer() as visual_writer:
            visual_writer.set_poses(positions=visual_position, orientations=visual_orientation)
        collider_pose = torch.tensor(
            [[pose.position_m[0], pose.position_m[1], pose.position_m[2] + 0.90,
              0.0, 0.0, sin(0.5 * np.deg2rad(pose.yaw_deg)), cos(0.5 * np.deg2rad(pose.yaw_deg))]],
            dtype=torch.float32, device=scene.device,
        )
        target_collider.write_root_pose_to_sim_index(root_pose=collider_pose)
        update_walk_cycle(controller, elapsed_s)
        scene.write_data_to_sim(); sim.step(); scene.update(DT)

        delta_yaw = (pose.yaw_deg - previous_yaw + 180.0) % 360.0 - 180.0
        yaw_rates.append(abs(delta_yaw) / DT)
        speeds.append(pose.speed_mps)
        grounding_errors.append(0.0)
        previous_yaw = pose.yaw_deg

        if (step + 1) % output_steps == 0:
            heading = np.deg2rad(pose.yaw_deg)
            forward = np.array((cos(heading), sin(heading)))
            right = np.array((forward[1], -forward[0]))
            eye_xy = np.asarray(pose.position_m[:2]) - 7.0 * forward + 3.0 * right
            eye = torch.tensor(
                [[eye_xy[0], eye_xy[1], pose.position_m[2] + 3.2]],
                dtype=torch.float32, device=scene.device,
            )
            look = torch.tensor(
                [[pose.position_m[0], pose.position_m[1], pose.position_m[2] + 1.0]],
                dtype=torch.float32, device=scene.device,
            )
            camera.set_world_poses_from_view(eye, look)
            # Give TAA several stationary-camera updates before capture.  This
            # is spatial/temporal render accumulation only; physics and target
            # motion do not advance during these extra samples.
            for _ in range(args.render_samples):
                scene.write_data_to_sim(); sim.render(); scene.update(0.0)
            rgb = np.asarray(camera.data.output["rgb"].torch[0, :, :, :3].cpu(), dtype=np.uint8)
            # Supersample the RTX sensor and area-downsample for stable texture
            # detail.  This reduces high-frequency terrain aliasing without the
            # blur introduced by a median filter.
            rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frames += 1
    writer.release()

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(final_video),
    ], check=True)
    raw_video.unlink()
    summary = {
        "schema_version": 1,
        "test": "live_terrain_target_visual_preview",
        "map_id": args.map_id,
        "route_index": args.route,
        "duration_s": args.duration_s,
        "start_distance_m": start_distance,
        "end_distance_m": pose.route_distance_m,
        "video_frames": frames,
        "render_samples_per_video_frame": args.render_samples,
        "render_resolution": [RENDER_WIDTH, RENDER_HEIGHT],
        "encoded_resolution": [WIDTH, HEIGHT],
        "renderer_noise_control": "stable_direct_lighting_plus_area_supersampling",
        "target_speed_mps": {"minimum": min(speeds), "maximum": max(speeds)},
        "maximum_yaw_rate_deg_s": max(yaw_rates),
        "terrain_grounding_method": "authored_triangle_mesh_interpolation",
        "semantic_surface_geometry_mode": surface_geometry_mode,
        "visible_physical_surface_height_offset_m": 0.0,
        "target_collision": "kinematic_capsule",
        "skeletal_motion": "captured_walk_cycle",
        "skeletal_foot_ik_validated": False,
        "distractor_people": 0,
        "dynamic_events": 0,
        "video": str(final_video.relative_to(ROOT)),
        "passed": (
            frames > 0 and min(speeds) >= 1.0 and max(speeds) <= 1.5
            and max(yaw_rates) <= 120.0 + 1e-6
            and surface_geometry_mode == "terrain_face_material_subsets"
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    status = "PASS" if summary["passed"] else "FAIL"
    print(f"LIVE_TARGET_ROUTE_PREVIEW {status} {final_video}")
    print(f"LIVE_TARGET_ROUTE_PREVIEW_REPORT {summary_path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
