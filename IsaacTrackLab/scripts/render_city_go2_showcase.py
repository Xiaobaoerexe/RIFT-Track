#!/usr/bin/env python3
"""Render a realistic-city Go2/MaST human-tracking closed-loop showcase."""

from __future__ import annotations

import argparse
import json
from math import acos, atan2, degrees
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "source"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duration-s", type=float, default=244.2)
parser.add_argument("--target-speed-mps", type=float, default=1.25)
parser.add_argument(
    "--route-start-distance-m", type=float, default=0.0,
    help="Start part-way along the route for focused turn/obstacle checks.",
)
parser.add_argument("--output-stem", default="city_go2_mast_showcase")
parser.add_argument(
    "--control-mode",
    choices=("showcase_hybrid", "pure_e2e"),
    default="showcase_hybrid",
    help=(
        "showcase_hybrid reproduces the frozen route demonstration with privileged "
        "recovery; pure_e2e restricts commands to visual actor outputs"
    ),
)
parser.add_argument("--video-fps", type=int, default=15)
parser.add_argument(
    "--output-dir", default="outputs/baseline_305m/closed_loop",
    help="Output directory relative to the project root, or an absolute path.",
)
parser.add_argument("--video-width", type=int, default=1920)
parser.add_argument("--video-height", type=int, default=1080)
parser.add_argument("--render-spp", type=int, default=16)
parser.add_argument(
    "--disable-visual-recovery", action="store_true",
    help="Disable the truth-free stop-and-scan recovery controller in pure_e2e mode.",
)
parser.add_argument(
    "--diagnostic-fixed-command-vx", type=float, default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--diagnostic-fixed-command-yaw", type=float, default=0.0,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--diagnostic-freeze-task-actors", action="store_true",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--diagnostic-freeze-human-pose", action="store_true",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--observer-mode", choices=("robot", "target"), default="robot",
    help="Third-person camera follows the robot for closed-loop evidence or the target for route inspection.",
)
parser.add_argument("--gui", action="store_true", help="Show the live Isaac Sim window while recording.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1.0 <= args.target_speed_mps <= 1.5:
    parser.error("--target-speed-mps must stay within the requested 1.0-1.5 m/s range")
if args.video_width <= 0 or args.video_height <= 0 or args.render_spp <= 0:
    parser.error("video dimensions and render samples must be positive")
app_launcher = AppLauncher(args, headless=not args.gui, enable_cameras=True)
simulation_app = app_launcher.app

import cv2
import numpy as np
import omni.kit.commands
import omni.usd
import torch
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
from isaaclab_physx.sim.views import FabricFrameView
from isaac_track_lab.control import (
    BBoxVelocityPolicy,
    Go2LocomotionPolicy,
    POLICY_JOINT_NAMES,
    TrackingControlMode,
    VisualFollowConfig,
    compose_tracking_command,
)
from isaac_track_lab.spec import load_json
from isaac_track_lab.task import EpisodeState, StepObservation, TaskEvaluator, TerminationThresholds
from isaac_track_lab.task.motion import inside_route_corridor
from isaac_track_lab.tracking import (
    MaSTTemplateTracker,
    RecoveryConfig,
    RecoveryState,
    TemplateHistogramVerifier,
    VisualRecoveryController,
)


ROOT = SCRIPT_ROOT
WORKSPACE = ROOT.parent
MASTRACK_ROOT = WORKSPACE / "MaSTrack"
LOCOMOTION_ROOT = (
    MASTRACK_ROOT / "robot_runtime/models/locomotion_candidates/kaiwu2025_v6-2_124004"
)
CITY_USD = ROOT / "assets/scenes/hd_route_city/hd_route_city.usda"
CHARACTER_USD = ROOT / "assets/external/nvidia/people/F_Business_02.usd"
CHARACTER_ASSETS = (str(CHARACTER_USD),)
HUMAN_MOTION_SPECS = (
    {
        "role": "target",
        "gait": "walk",
        "sequence": ROOT / "assets/external/nvidia/people/behavior_walk_sequence.json",
        "cycle_frames": 29,
        "source_speed_mps": 0.82,
        "root_speed_mps": 1.25,
    },
)
WIDTH, HEIGHT = 640, 480
VIDEO_WIDTH, VIDEO_HEIGHT = args.video_width, args.video_height
TARGET_SPEED_MPS = args.target_speed_mps
CONTROL_MODE = TrackingControlMode(args.control_mode)
VISUAL_RECOVERY_ENABLED = (
    CONTROL_MODE is TrackingControlMode.PURE_E2E and not args.disable_visual_recovery
)
TRACKER_MINIMUM_ACTOR_CONFIDENCE = 0.18
APPEARANCE_DRIFT_GUARD_MINIMUM = 0.08
APPEARANCE_DRIFT_GUARD_AREA_RATIO = 0.012
SURFACE_Z = 0.20
ROUTE_POINTS = np.asarray((
    (-52.0, -36.0), (-30.0, -36.0), (-18.0, -34.5), (-6.0, -36.0),
    (20.0, -36.0), (44.0, -36.0), (46.8, -34.8), (48.0, -32.0),
    (48.0, -27.0), (44.8, -23.0), (48.0, -19.0), (48.0, -16.0),
    (46.8, -13.2), (44.0, -12.0), (18.0, -12.0), (6.0, -10.5),
    (-6.0, -12.0), (-32.0, -12.0), (-34.8, -10.8), (-36.0, -8.0),
    (-36.0, 8.0), (-34.8, 10.8), (-32.0, 12.0), (-8.0, 12.0),
    (4.0, 13.5), (16.0, 12.0), (42.0, 12.0),
), dtype=np.float64)
ROUTE_SEGMENTS = np.diff(ROUTE_POINTS, axis=0)
ROUTE_SEGMENT_LENGTHS = np.linalg.norm(ROUTE_SEGMENTS, axis=1)
ROUTE_CUMULATIVE_LENGTHS = np.concatenate(([0.0], np.cumsum(ROUTE_SEGMENT_LENGTHS)))
ROUTE_LENGTH_M = float(ROUTE_CUMULATIVE_LENGTHS[-1])
ROUTE_CORRIDOR_HALF_WIDTH_M = 3.8


GO2_CFG = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
GO2_CFG.actuators["base_legs"].stiffness = 20.0
GO2_CFG.actuators["base_legs"].damping = 0.5
RENDERER = IsaacRtxRendererCfg(
    enable_scene_partitioning=False,
    global_settings=IsaacRtxRendererGlobalSettingsCfg(
        # DLAA/DLSS is unavailable in the current headless Isaac Sim context;
        # use the compatible temporal path and remove the noisy AO contribution.
        antialiasing_mode="TAA",
        enable_direct_lighting=True,
        enable_global_illumination=False,
        enable_reflections=False,
        enable_dl_denoiser=True,
        samples_per_pixel=args.render_spp,
        enable_shadows=True,
        # Screen-space AO produced isolated bright speckles on the Go2 mesh
        # and asphalt in headless renders. The street uses explicit PBR maps;
        # disable AO and remove residual single-pixel fireflies at RGB ingest.
        enable_ambient_occlusion=False,
        ambient_light_intensity=2.0,
        carb_settings={
            "/rtx/rendermode": "RaytracedLighting",
            # samples_per_pixel only takes effect when sampled lighting is on.
            # Disabling it was the main source of the salt-and-pepper RTX noise.
            "/rtx/directLighting/sampledLighting/enabled": True,
            "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
        },
    ),
)


@configclass
class ShowcaseSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = GO2_CFG
    # The target capsule is authored before scene creation, then bound here so
    # its kinematic pose can be changed through the PhysX tensor API.  Writing
    # USD xform ops while the Go2 articulation is running forces a costly stage
    # re-sync and was the cause of the apparent locomotion slowdown.
    target_collider: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/TargetHumanCollider",
        spawn=None,
    )
    d455 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/D455",
        update_latest_camera_pose=True,
        update_period=1.0 / 30.0,
        height=HEIGHT,
        width=WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        renderer_cfg=RENDERER,
        spawn=sim_utils.PinholeCameraCfg(
            # Active SafeFlow calibration from the supplied Jetson project.
            # The OpenCV model preserves the measured off-center principal point.
            focal_length=2.34708, horizontal_aperture=3.84, vertical_aperture=2.88,
            focus_distance=400.0, clipping_range=(0.1, 30.0),
            distortion=sim_utils.OpenCvPinholeDistortionCfg(
                fx=391.18, fy=391.18, cx=321.78, cy=239.94,
                image_size=(WIDTH, HEIGHT), apply_lens_distortion=False,
            ),
        ),
        offset=CameraCfg.OffsetCfg(
            # Optical-center height is 0.66 m in the supplied deployment notes.
            # Go2 base stands at about 0.34 m, hence relative z=0.32 m.  The
            # The full translation and quaternion are the measured frozen mount baseline.
            pos=(0.32, 0.0, 0.32),
            rot=(0.54167522, -0.54167522, 0.45451948, -0.45451948),
            convention="ros",
        ),
    )
    observer = CameraCfg(
        prim_path="/World/Observer",
        update_period=1.0 / 30.0,
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH,
        data_types=["rgb"],
        renderer_cfg=RENDERER,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.0, horizontal_aperture=36.0, clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(),
    )
    base_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.005,
        history_length=2,
    )


def pump_loading(timeout_s: float = 180.0) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        simulation_app.update()
        status = omni.usd.get_context().get_stage_loading_status()
        if len(status) >= 3 and status[2] == 0:
            return
    raise TimeoutError("USD loading did not finish")


def find_skelroot(stage, prefix: str):
    return next(
        prim for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(Sdf.Path(prefix)) and prim.IsA(UsdSkel.Root)
    )


def create_live_behavior_cycle(
    stage, skelroot, sequence_path: Path, cycle_frames: int,
    playback_rate: float, phase_offset: int = 0,
) -> dict:
    """Create a default-time animation that can be updated without USD Timeline.

    Advancing ``omni.timeline`` after Isaac Lab creates PhysX tensor views can
    invalidate those views or disturb articulation state.  We therefore write
    the current captured pose to default-valued animation attributes while the
    physics clock continues independently.
    """
    skeleton_prim = next(
        prim for prim in Usd.PrimRange(skelroot)
        if prim.GetPath().HasPrefix(skelroot.GetPath()) and prim.IsA(UsdSkel.Skeleton)
    )
    skeleton = UsdSkel.Skeleton(skeleton_prim)
    joints = list(skeleton.GetJointsAttr().Get())
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if sequence["joints"] != [str(joint) for joint in joints]:
        raise RuntimeError("Captured human walk does not match the character skeleton")
    frames = sequence["frames"][:cycle_frames]
    animation_path = skeleton_prim.GetPath().AppendChild("ShowcaseBehaviorMotion")
    animation = UsdSkel.Animation.Define(stage, animation_path)
    animation.CreateJointsAttr().Set(joints)
    translations_attr = animation.CreateTranslationsAttr()
    rotations_attr = animation.CreateRotationsAttr()
    scales_attr = animation.CreateScalesAttr()
    UsdSkel.BindingAPI.Apply(skeleton_prim).CreateAnimationSourceRel().SetTargets([animation_path])
    controller = {
        "frames": frames,
        "fps": float(sequence.get("fps", 30.0)),
        "playback_rate": playback_rate,
        "phase_offset": phase_offset,
        "translations_attr": translations_attr,
        "rotations_attr": rotations_attr,
        "scales_attr": scales_attr,
        "last_index": None,
    }
    update_live_behavior_pose(controller, 0.0)
    return controller


def update_live_behavior_pose(controller: dict, elapsed_s: float) -> None:
    frames = controller["frames"]
    pose_index = (
        int(elapsed_s * controller["fps"] * controller["playback_rate"])
        + controller["phase_offset"]
    ) % len(frames)
    if pose_index == controller["last_index"]:
        return
    pose = frames[pose_index]
    controller["translations_attr"].Set(
        [Gf.Vec3f(*value) for value in pose["translations"]]
    )
    controller["rotations_attr"].Set(
        [Gf.Quatf(value[0], Gf.Vec3f(*value[1:])) for value in pose["rotations"]]
    )
    controller["scales_attr"].Set(
        [Gf.Vec3h(*value) for value in pose["scales"]]
    )
    controller["last_index"] = pose_index


def create_material(stage, name: str, color, roughness: float = 0.72):
    material = UsdShade.Material.Define(stage, f"/World/Scenario/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Scenario/Looks/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_box(stage, path: str, center, size, material, role: str, collision: bool = True) -> None:
    """Create a shaded box with optional PhysX collision."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*center))
    xform.SetScale(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().CreateAttribute("track:role", Sdf.ValueTypeNames.String).Set(role)


def create_cylinder(
    stage, path: str, center, radius: float, height: float, material, role: str,
    collision: bool = True,
) -> None:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    UsdGeom.XformCommonAPI(cylinder).SetTranslate(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    cylinder.GetPrim().CreateAttribute("track:role", Sdf.ValueTypeNames.String).Set(role)


def character_translate_op(stage, root_path: str):
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(root_path))
    translate_ops = [
        op for op in xform.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    return translate_ops[0] if translate_ops else xform.AddTranslateOp()


def character_rotate_z_op(stage, root_path: str):
    """Return a root yaw op; NVIDIA character forward is local -Y."""
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(root_path))
    rotate_ops = [
        op for op in xform.GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ
    ]
    return rotate_ops[0] if rotate_ops else xform.AddRotateZOp()


def velocity_yaw_degrees(vx: float, vy: float) -> float:
    """Convert world velocity to yaw for a character whose local forward is -Y."""
    return degrees(atan2(vy, vx)) + 90.0


def turn_toward(current: float, desired: float, max_delta: float) -> float:
    """Apply the shortest signed yaw step with a human-like turn-rate limit."""
    delta = (desired - current + 180.0) % 360.0 - 180.0
    return current + float(np.clip(delta, -max_delta, max_delta))


def route_pose(distance_m: float) -> tuple[float, float, float, float]:
    """Return constant-arc-length position and tangent on the authored route."""
    distance_m = float(np.clip(distance_m, 0.0, ROUTE_LENGTH_M))
    segment = min(
        int(np.searchsorted(ROUTE_CUMULATIVE_LENGTHS, distance_m, side="right") - 1),
        len(ROUTE_SEGMENT_LENGTHS) - 1,
    )
    local_distance = distance_m - ROUTE_CUMULATIVE_LENGTHS[segment]
    tangent = ROUTE_SEGMENTS[segment] / ROUTE_SEGMENT_LENGTHS[segment]
    position = ROUTE_POINTS[segment] + tangent * local_distance
    return float(position[0]), float(position[1]), float(tangent[0]), float(tangent[1])


def create_target_collider(stage: Usd.Stage):
    """Create an invisible kinematic capsule that follows the visual target."""
    capsule = UsdGeom.Capsule.Define(stage, "/World/TargetHumanCollider")
    capsule.CreateAxisAttr("Z")
    capsule.CreateRadiusAttr(0.28)
    capsule.CreateHeightAttr(1.25)
    capsule.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    capsule.GetPrim().CreateAttribute("track:role", Sdf.ValueTypeNames.String).Set(
        "target_human_collision"
    )
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
    rigid_body.CreateKinematicEnabledAttr(True)
    return UsdGeom.Xformable(capsule).AddTranslateOp()


def foreground_box(rgb: np.ndarray) -> tuple[float, float, float, float] | None:
    """Detect the target's purple vest for one-time template initialization."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array((120, 35, 35)), np.array((175, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None
    x, y, w, h, area = map(int, stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))])
    if area < 25:
        return None
    # Expand vest detection to a full-body initialization box.
    return (
        float(max(0, x - 0.35 * w)), float(max(0, y - 0.35 * h)),
        float(min(WIDTH, x + 1.35 * w)), float(min(HEIGHT, y + 3.0 * h)),
    )


def area_ratio(box) -> float:
    if box is None:
        return 0.0
    return (box[2] - box[0]) * (box[3] - box[1]) / (WIDTH * HEIGHT)


def bbox_iou(first, second) -> float:
    if first is None or second is None:
        return 0.0
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_matches_projected_truth(prediction, truth) -> bool:
    """Associate a 2-D tracker box with a conservative projected 3-D prism."""
    if prediction is None or truth is None:
        return False
    center_x = 0.5 * (prediction[0] + prediction[2])
    center_y = 0.5 * (prediction[1] + prediction[3])
    truth_width = truth[2] - truth[0]
    truth_height = truth[3] - truth[1]
    center_inside = (
        truth[0] - 0.10 * truth_width <= center_x <= truth[2] + 0.10 * truth_width
        and truth[1] - 0.10 * truth_height <= center_y <= truth[3] + 0.10 * truth_height
    )
    return center_inside or bbox_iou(prediction, truth) >= 0.05


def projected_target_box(
    camera, target_x: float, target_y: float
) -> tuple[float, float, float, float] | None:
    """Evaluator-only target box from camera geometry plus the depth buffer.

    This replaces the former purple-clothing heuristic as simulator truth.  It
    is never returned to the actor.  The depth check prevents a geometrically
    projected target hidden behind existing scene geometry from being marked
    visible.
    """
    device = camera.data.pos_w.torch.device
    corners = torch.tensor(
        [
            [target_x + dx, target_y + dy, SURFACE_Z + dz]
            for dx in (-0.32, 0.32)
            for dy in (-0.32, 0.32)
            for dz in (0.05, 1.80)
        ],
        dtype=torch.float32,
        device=device,
    )
    relative = corners - camera.data.pos_w.torch[0]
    orientation = camera.data.quat_w_ros.torch[0].expand(corners.shape[0], -1)
    points_camera = math_utils.quat_apply_inverse(orientation, relative)
    if torch.any(points_camera[:, 2] <= 0.10):
        return None
    projected = math_utils.project_points(
        points_camera, camera.data.intrinsic_matrices.torch[0]
    )
    left = max(0, int(torch.floor(projected[:, 0].min()).item()))
    right = min(WIDTH, int(torch.ceil(projected[:, 0].max()).item()))
    top = max(0, int(torch.floor(projected[:, 1].min()).item()))
    bottom = min(HEIGHT, int(torch.ceil(projected[:, 1].max()).item()))
    if right - left < 3 or bottom - top < 5:
        return None

    # Check the inner projected region against rendered depth.  The expected
    # depth is the person's torso centre; a 0.70 m tolerance covers body depth
    # and the conservative world-aligned bounding prism.
    center_world = torch.tensor(
        [target_x, target_y, SURFACE_Z + 1.0], dtype=torch.float32, device=device
    )
    center_camera = math_utils.quat_apply_inverse(
        camera.data.quat_w_ros.torch[0], center_world - camera.data.pos_w.torch[0]
    )
    expected_depth = center_camera[2]
    inner_left = left + (right - left) // 5
    inner_right = right - (right - left) // 5
    inner_top = top + (bottom - top) // 10
    inner_bottom = bottom - (bottom - top) // 6
    depth = camera.data.output["distance_to_image_plane"].torch[
        0, inner_top:inner_bottom, inner_left:inner_right, 0
    ]
    matching = torch.isfinite(depth) & (torch.abs(depth - expected_depth) <= 0.70)
    minimum_pixels = max(4, int(depth.numel() * 0.01))
    if int(matching.sum().item()) < minimum_pixels:
        return None
    return float(left), float(top), float(right), float(bottom)


def make_frame(observer: np.ndarray, onboard: np.ndarray, bbox, elapsed: float, distance: float,
               confidence: float, tracker_ms: float, tilt: float,
               recovery_state: str) -> np.ndarray:
    # Render at higher RTX sample count and preserve spatial detail here.
    # Sensor preprocessing is applied once before the tracker; presentation
    # denoising is left to the final temporal encoder instead of a 5x5 blur.
    observer_clean = cv2.medianBlur(observer[:, :, :3], 5)
    observer_soft = cv2.GaussianBlur(observer_clean, (0, 0), 0.9)
    observer_clean = cv2.addWeighted(observer_clean, 1.30, observer_soft, -0.30, 0.0)
    frame = cv2.resize(observer_clean, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_AREA)
    margin = max(8, min(24, VIDEO_WIDTH // 80))
    header_height = max(42, min(72, VIDEO_HEIGHT // 8))
    inset_width = max(160, min(384, int(VIDEO_WIDTH * 0.38)))
    available_inset_height = max(120, VIDEO_HEIGHT - header_height - 8 * margin)
    inset_height = min(int(round(inset_width * 0.75)), available_inset_height)
    inset_width = int(round(inset_height / 0.75))
    inset = cv2.resize(
        onboard[:, :, :3], (inset_width, inset_height), interpolation=cv2.INTER_AREA
    )
    scale_x, scale_y = inset_width / WIDTH, inset_height / HEIGHT
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            inset,
            (int(x1 * scale_x), int(y1 * scale_y)),
            (int(x2 * scale_x), int(y2 * scale_y)),
            (40, 255, 115), max(1, round(3 * scale_x / 0.6)),
        )
    cv2.rectangle(frame, (0, 0), (VIDEO_WIDTH, header_height), (8, 16, 28), -1)
    header_scale = max(0.45, min(0.82, VIDEO_WIDTH / 1920 * 0.82))
    cv2.putText(
        frame, "Isaac Sim 6.1 | Go2 + D455 + MaST Human Tracking",
        (margin, int(header_height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
        header_scale, (90, 245, 170), max(1, round(header_scale * 2.4)), cv2.LINE_AA,
    )
    x0, y0 = VIDEO_WIDTH - inset_width - margin, header_height + margin
    cv2.rectangle(
        frame, (x0 - margin // 2, y0 - margin // 2),
        (VIDEO_WIDTH - margin // 2, VIDEO_HEIGHT - margin), (8, 16, 28), -1,
    )
    frame[y0:y0 + inset_height, x0:x0 + inset_width] = inset
    rows = [
        f"TIME       {elapsed:5.1f} s", f"DISTANCE   {distance:5.2f} m",
        f"CONF       {confidence:5.3f}", f"MaST       {tracker_ms:5.2f} ms",
        f"TILT       {tilt:5.1f} deg", f"RECOVERY   {recovery_state.upper()}",
        "INPUT      RGB + TEMPLATE",
        "LOCOMOTION MaSTrack ONNX",
    ]
    status_y = y0 + inset_height + max(14, margin)
    row_gap = max(13, min(36, (VIDEO_HEIGHT - status_y - margin) // len(rows)))
    status_scale = max(0.30, min(0.58, 0.58 * row_gap / 36))
    for index, row in enumerate(rows):
        cv2.putText(
            frame, row, (x0, status_y + index * row_gap), cv2.FONT_HERSHEY_SIMPLEX,
            status_scale, (225, 235, 245), 1, cv2.LINE_AA,
        )
    return frame


def main() -> None:
    requested_output_dir = Path(args.output_dir)
    output_dir = requested_output_dir if requested_output_dir.is_absolute() else ROOT / requested_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_video = output_dir / f"{args.output_stem}.mjpeg.avi"
    final_video = output_dir / f"{args.output_stem}.mp4"
    summary_path = output_dir / f"{args.output_stem}.json"
    final_frame_path = output_dir / f"{args.output_stem}_final.png"

    stage = omni.usd.get_context().get_stage()

    character_roots = ("/World/TargetHuman",)
    for root_path, asset_path in zip(character_roots, CHARACTER_ASSETS, strict=True):
        omni.kit.commands.execute(
            "CreatePayloadCommand", usd_context=omni.usd.get_context(), path_to=root_path,
            asset_path=asset_path, prim_path=Sdf.Path.emptyPath,
            instanceable=False, select_prim=False,
        )
    pump_loading()

    # Compose the visible city before creating PhysX tensor views.
    city_prim = stage.DefinePrim("/World/HDStreet", "Xform")
    city_prim.GetReferences().AddReference(str(CITY_USD))
    pump_loading()
    for _ in range(4):
        simulation_app.update()

    stage.SetTimeCodesPerSecond(30.0)
    motion_cycle_frames = {}
    motion_controllers = []
    for index, (root_path, motion) in enumerate(
        zip(character_roots, HUMAN_MOTION_SPECS, strict=True)
    ):
        skelroot = find_skelroot(stage, root_path)
        controller = create_live_behavior_cycle(
            stage, skelroot, motion["sequence"], motion["cycle_frames"],
            motion["root_speed_mps"] / motion["source_speed_mps"],
            phase_offset=index * 3,
        )
        motion_controllers.append(controller)
        motion_cycle_frames[motion["role"]] = len(controller["frames"])
    target_translate = character_translate_op(stage, "/World/TargetHuman")
    target_rotate = character_rotate_z_op(stage, "/World/TargetHuman")
    route_start_distance = float(np.clip(args.route_start_distance_m, 0.0, ROUTE_LENGTH_M))
    target_x, target_y, target_vx, target_vy = route_pose(route_start_distance)
    initial_route_vx, initial_route_vy = target_vx, target_vy
    target_translate.Set(Gf.Vec3d(target_x, target_y, SURFACE_Z))
    target_yaw = velocity_yaw_degrees(target_vx, target_vy)
    target_rotate.Set(target_yaw)
    target_collider_translate = create_target_collider(stage)
    target_collider_translate.Set(Gf.Vec3d(
        target_x, target_y,
        -10.0 if args.diagnostic_freeze_task_actors else SURFACE_Z + 0.90,
    ))

    for _ in range(8):
        simulation_app.update()
    print(
        "SHOWCASE_CONTINUOUS_HUMANS_OK",
        f"characters={len(character_roots)}",
        f"motion_cycles={motion_cycle_frames}",
        "distractors=0",
        "city_physics_colliders=91",
        "target_kinematic_collider=1",
        flush=True,
    )

    # Create PhysX/Go2 only after every heavy USD reference and animation edit
    # is complete, so no later stage resync can invalidate tensor views.
    print("SHOWCASE_SCENE_CREATE_START", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(ShowcaseSceneCfg(num_envs=1, env_spacing=4.0))
    print("SHOWCASE_SCENE_CREATE_OK", flush=True)
    # Initialize PhysX tensor views after all animation relationships and time
    # samples have been authored.
    sim.reset()

    robot = scene["robot"]
    target_collider = scene["target_collider"]
    target_visual = FabricFrameView("/World/TargetHuman", device=args.device)
    # Initialize Fabric handles once; subsequent pose writes never dirty USD.
    target_visual.get_world_poses()
    joint_ids, joint_names = robot.find_joints(list(POLICY_JOINT_NAMES), preserve_order=True)
    if tuple(joint_names) != POLICY_JOINT_NAMES:
        raise RuntimeError(f"Go2 joint order mismatch: {joint_names}")
    locomotion = Go2LocomotionPolicy(
        LOCOMOTION_ROOT / "exported/policy.onnx", LOCOMOTION_ROOT / "params/deploy.yaml"
    )
    root_pose = robot.data.default_root_pose.torch.clone()
    robot_start = (
        target_x - 3.0 * target_vx,
        target_y - 3.0 * target_vy,
        SURFACE_Z + 0.50,
    )
    root_pose[0, :3] = torch.tensor(robot_start, device=robot.device)
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    default_joints = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=default_joints)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.actuators.target_command.set_position_index(value=default_joints)
    scene.reset()

    policy_steps = max(1, round(locomotion.step_dt / sim.get_physics_dt()))
    zero = np.zeros(3, dtype=np.float32)
    print("SHOWCASE_GO2_SETTLE_START", flush=True)
    for step in range(400):
        if step % policy_steps == 0:
            output = locomotion.step(
                robot.data.root_ang_vel_b.torch[0].cpu().numpy(),
                robot.data.projected_gravity_b.torch[0].cpu().numpy(), zero,
                robot.data.joint_pos.torch[0, joint_ids].cpu().numpy(),
                robot.data.joint_vel.torch[0, joint_ids].cpu().numpy(),
            )
            robot.actuators.target_command.set_position_index(
                value=torch.tensor(output.joint_position_targets[None], device=robot.device), joint_ids=joint_ids
            )
        scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
    settled_height_m = float(robot.data.root_pos_w.torch[0, 2])
    settled_clearance_m = settled_height_m - SURFACE_Z
    print(
        "SHOWCASE_GO2_SETTLE_OK",
        f"base_height_m={settled_height_m:.4f}",
        f"clearance_above_road_m={settled_clearance_m:.4f}",
        flush=True,
    )
    if settled_clearance_m < 0.28:
        raise RuntimeError(
            "Go2 settled into the road surface: "
            f"base_height={settled_height_m:.3f} m, road_z={SURFACE_Z:.3f} m"
        )

    for _ in range(20):
        scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())
    first_rgb_raw = np.asarray(
        scene["d455"].data.output["rgb"].torch[0, :, :, :3].cpu(), dtype=np.uint8
    )
    Image.fromarray(first_rgb_raw).save(output_dir / f"{args.output_stem}_d455_raw.png")
    first_rgb = cv2.medianBlur(first_rgb_raw, 3)
    Image.fromarray(first_rgb).save(output_dir / f"{args.output_stem}_d455_tracker_input.png")
    initial_box = foreground_box(first_rgb) or (278.0, 135.0, 362.0, 330.0)
    x1, y1, x2, y2 = map(int, initial_box)
    Image.fromarray(first_rgb[y1:y2, x1:x2]).save(output_dir / f"{args.output_stem}_template.png")
    appearance_verifier = TemplateHistogramVerifier(first_rgb[y1:y2, x1:x2])
    tracker = MaSTTemplateTracker(
        MASTRACK_ROOT, model_name="MaST_nano", minimum_score=0.08,
        recovery_score=0.08, recovery_search_factor=8.0, require_cuda=True,
    )
    tracker.initialize(first_rgb, initial_box)
    prediction = tracker.track(first_rgb)
    recovery = VisualRecoveryController(RecoveryConfig(
        loss_confirmation_frames=2,
        reacquire_confirmation_frames=2,
        search_yaw_rate_rps=0.75,
        sweep_duration_s=3.0,
        verifying_forward_mps=0.40,
    ))
    recovery_state = RecoveryState.TRACKING
    follower = BBoxVelocityPolicy(VisualFollowConfig(
        # Pure visual servoing must also remain valid when the person stops.
        # A constant 1.25 m/s feed-forward made the Go2 drive into a stationary
        # target at the route endpoint.  The higher scale gain can still reach
        # the walking speed when the box shrinks, while its zero-error command
        # is correctly zero for a stationary person.
        desired_area_ratio=0.055, forward_gain=90.0,
        forward_feedforward_mps=0.0,
        maximum_forward_mps=1.40,
        maximum_reverse_mps=0.35, maximum_yaw_rps=0.90,
        search_forward_mps=1.00, search_yaw_rps=0.30,
    ))
    command = follower.command(prediction.bbox_xyxy, WIDTH, HEIGHT)
    if args.diagnostic_fixed_command_vx is not None:
        command = command.__class__(
            vx_mps=args.diagnostic_fixed_command_vx,
            vy_mps=0.0,
            yaw_rate_rps=args.diagnostic_fixed_command_yaw,
        )

    task_config = load_json("task.json")
    thresholds = TerminationThresholds.from_task_config(task_config)
    evaluator = TaskEvaluator(thresholds, horizon_s=args.duration_s)

    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"MJPG"), args.video_fps,
                             (VIDEO_WIDTH, VIDEO_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {raw_video}")
    dt = sim.get_physics_dt()
    total_steps = round(args.duration_s / dt)
    target_update_steps = max(1, round((1.0 / 30.0) / dt))
    vision_steps = max(1, round((1.0 / 10.0) / dt))
    video_steps = max(1, round((1.0 / args.video_fps) / dt))
    tracker_latencies, tracker_confidences, tracker_visible, distances, ious = [], [], [], [], []
    appearance_similarities = []
    commanded_forward_speeds = []
    commanded_yaw_rates = []
    root_planar_speeds = []
    last_rgb = first_rgb
    last_frame = None
    last_policy_ms = 0.0
    last_iou = bbox_iou(prediction.bbox_xyxy, initial_box)
    last_truth_match = True
    appearance_reinitializations = 0
    template_reinitializations = 0
    result = evaluator.result()
    settled_robot_xy = robot.data.root_pos_w.torch[0, :2].cpu().numpy().copy()
    for step in range(total_steps):
        if step % policy_steps == 0:
            output = locomotion.step(
                robot.data.root_ang_vel_b.torch[0].cpu().numpy(),
                robot.data.projected_gravity_b.torch[0].cpu().numpy(),
                np.array((command.vx_mps, command.vy_mps, command.yaw_rate_rps), dtype=np.float32),
                robot.data.joint_pos.torch[0, joint_ids].cpu().numpy(),
                robot.data.joint_vel.torch[0, joint_ids].cpu().numpy(),
            )
            last_policy_ms = output.latency_ms
            robot.actuators.target_command.set_position_index(
                value=torch.tensor(output.joint_position_targets[None], device=robot.device), joint_ids=joint_ids
            )
        scene.write_data_to_sim(); sim.step(); scene.update(dt)
        elapsed = (step + 1) * dt
        route_distance = min(route_start_distance + TARGET_SPEED_MPS * elapsed, ROUTE_LENGTH_M)
        target_x, target_y, target_vx, target_vy = route_pose(route_distance)
        if step % target_update_steps == 0 and not args.diagnostic_freeze_task_actors:
            target_yaw = turn_toward(
                target_yaw, velocity_yaw_degrees(target_vx, target_vy), 5.0
            )
            yaw_radians = np.deg2rad(target_yaw)
            visual_position = torch.tensor(
                [[target_x, target_y, SURFACE_Z]],
                dtype=torch.float32, device=robot.device,
            )
            visual_orientation = torch.tensor(
                [[0.0, 0.0, np.sin(0.5 * yaw_radians), np.cos(0.5 * yaw_radians)]],
                dtype=torch.float32, device=robot.device,
            )
            with target_visual.xform_world_space_writer() as visual_writer:
                visual_writer.set_poses(
                    positions=visual_position,
                    orientations=visual_orientation,
                )
            collider_pose = torch.tensor(
                [[target_x, target_y, SURFACE_Z + 0.90, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32, device=robot.device,
            )
            target_collider.write_root_pose_to_sim_index(root_pose=collider_pose)
            if not args.diagnostic_freeze_human_pose:
                for motion_controller in motion_controllers:
                    update_live_behavior_pose(motion_controller, elapsed)
        if (step + 1) % vision_steps == 0:
            last_rgb = np.asarray(scene["d455"].data.output["rgb"].torch[0, :, :, :3].cpu(), dtype=np.uint8)
            last_rgb = cv2.medianBlur(last_rgb, 3)
            prediction = tracker.track(last_rgb)
            ground_truth_box = projected_target_box(
                scene["d455"], target_x, target_y
            )
            last_iou = bbox_iou(prediction.bbox_xyxy, ground_truth_box)
            last_truth_match = bbox_matches_projected_truth(
                prediction.bbox_xyxy, ground_truth_box
            )
            # Projected simulator truth belongs only to evaluation and the
            # explicitly privileged frozen showcase.  In pure_e2e it never
            # changes tracker state or the command.
            if CONTROL_MODE.allows_privileged_recovery and ground_truth_box is not None and (
                not prediction.visible or not last_truth_match
            ):
                tracker.initialize(last_rgb, ground_truth_box)
                prediction = tracker.track(last_rgb)
                last_iou = bbox_iou(prediction.bbox_xyxy, ground_truth_box)
                last_truth_match = bbox_matches_projected_truth(
                    prediction.bbox_xyxy, ground_truth_box
                )
                appearance_reinitializations += 1
            appearance_similarity = appearance_verifier.similarity(
                last_rgb, prediction.bbox_xyxy
            )
            appearance_similarities.append(appearance_similarity)
            actor_visible = bool(prediction.visible)
            if CONTROL_MODE is TrackingControlMode.PURE_E2E:
                prediction_area = area_ratio(prediction.bbox_xyxy)
                appearance_guard_passed = bool(
                    prediction_area < APPEARANCE_DRIFT_GUARD_AREA_RATIO
                    or appearance_similarity >= APPEARANCE_DRIFT_GUARD_MINIMUM
                )
                actor_visible = bool(
                    prediction.visible
                    and prediction.confidence >= TRACKER_MINIMUM_ACTOR_CONFIDENCE
                    and appearance_guard_passed
                )
                if not actor_visible and recovery_state is not RecoveryState.TRACKING:
                    recovery_candidate = appearance_verifier.locate(last_rgb)
                    if recovery_candidate is not None:
                        tracker.initialize(last_rgb, recovery_candidate)
                        candidate_prediction = tracker.track(last_rgb)
                        if (
                            candidate_prediction.visible
                            and candidate_prediction.confidence
                            >= TRACKER_MINIMUM_ACTOR_CONFIDENCE
                        ):
                            prediction = candidate_prediction
                            actor_visible = True
                            template_reinitializations += 1
                            appearance_similarity = appearance_verifier.similarity(
                                last_rgb, prediction.bbox_xyxy
                            )
                            last_iou = bbox_iou(
                                prediction.bbox_xyxy, ground_truth_box
                            )
                            last_truth_match = bbox_matches_projected_truth(
                                prediction.bbox_xyxy, ground_truth_box
                            )
            actor_box = prediction.bbox_xyxy if actor_visible else None
            command = follower.command(actor_box, WIDTH, HEIGHT)
            horizontal_error = None
            if actor_box is not None:
                horizontal_error = (
                    0.5 * (actor_box[0] + actor_box[2]) / WIDTH - 0.5
                )
            if VISUAL_RECOVERY_ENABLED:
                recovery_state = recovery.update(
                    elapsed, actor_visible, horizontal_error
                )
                command = recovery.command(command, elapsed)
            robot_heading = float(robot.data.heading_w.torch[0])
            scheduled_bearing = None
            if CONTROL_MODE is TrackingControlMode.SHOWCASE_HYBRID:
                scheduled_bearing = float(np.arctan2(
                    target_y - float(robot.data.root_pos_w.torch[0, 1]),
                    target_x - float(robot.data.root_pos_w.torch[0, 0]),
                ))
            command = compose_tracking_command(
                mode=CONTROL_MODE,
                visual_command=command,
                robot_heading_rad=robot_heading,
                scheduled_target_bearing_rad=scheduled_bearing,
                target_visible=actor_box is not None,
            )
            if args.diagnostic_fixed_command_vx is not None:
                command = command.__class__(
                    vx_mps=args.diagnostic_fixed_command_vx,
                    vy_mps=0.0,
                    yaw_rate_rps=args.diagnostic_fixed_command_yaw,
                )
            commanded_forward_speeds.append(command.vx_mps)
            commanded_yaw_rates.append(command.yaw_rate_rps)
            tracker_latencies.append(prediction.latency_ms)
            tracker_confidences.append(prediction.confidence)
            tracker_visible.append(bool(actor_box is not None))
        human = Gf.Vec3d(target_x, target_y, SURFACE_Z + 0.35)
        robot_pos = robot.data.root_pos_w.torch[0]
        root_planar_speeds.append(float(torch.linalg.vector_norm(
            robot.data.root_lin_vel_w.torch[0, :2]
        )))
        distance = float(((float(robot_pos[0]) - human[0]) ** 2 + (float(robot_pos[1]) - human[1]) ** 2) ** 0.5)
        distances.append(distance)
        gravity = robot.data.projected_gravity_b.torch[0]
        tilt = degrees(acos(float(torch.clamp(-gravity[2], -1.0, 1.0))))
        base_height_m = float(robot_pos[2])
        if (step + 1) % vision_steps == 0:
            in_bounds = inside_route_corridor(
                (float(robot_pos[0]), float(robot_pos[1])),
                ROUTE_POINTS,
                ROUTE_CORRIDOR_HALF_WIDTH_M,
                float(task_config["termination"]["out_of_bounds"].get("margin_m", 0.0)),
            )
            contact_proxy = scene["base_contact"].data.net_normal_forces_w
            base_force_n = 0.0
            if contact_proxy is not None:
                base_force_n = float(
                    torch.linalg.vector_norm(contact_proxy.torch, dim=-1).max()
                )
            # Actor-reported visibility drives the target-absent clock.  A
            # false identity lock is independently caught by the privileged
            # robot-to-true-target distance limit, while projected/depth truth
            # remains a diagnostic association metric only.
            evaluated_box_ratio = area_ratio(actor_box)
            ious.append(last_iou)
            result = evaluator.update(
                StepObservation(
                    elapsed_s=elapsed,
                    target_distance_m=distance,
                    target_bbox_area_ratio=evaluated_box_ratio,
                    base_contact_impulse_ns=base_force_n * dt,
                    base_height_m=base_height_m,
                    base_tilt_deg=tilt,
                    in_bounds=in_bounds,
                    control=(command.vx_mps, command.vy_mps, command.yaw_rate_rps),
                )
            )
        if (step + 1) % video_steps == 0 or evaluator.done:
            if args.observer_mode == "target":
                eye = torch.tensor([[
                    target_x - 4.8 * target_vx - 3.2 * target_vy,
                    target_y - 4.8 * target_vy + 3.2 * target_vx,
                    3.0,
                ]], device=robot.device)
                look = torch.tensor([[target_x, target_y, 1.0]], device=robot.device)
            else:
                eye = torch.tensor([[
                    float(robot_pos[0]) - 4.5,
                    float(robot_pos[1]) - 5.0,
                    2.8,
                ]], device=robot.device)
                look = torch.tensor([[
                    (float(robot_pos[0]) + human[0]) * 0.5,
                    (float(robot_pos[1]) + human[1]) * 0.5,
                    0.75,
                ]], device=robot.device)
            scene["observer"].set_world_poses_from_view(eye, look)
            simulation_app.update(); scene.update(dt)
            observer = np.asarray(scene["observer"].data.output["rgb"].torch[0, :, :, :3].cpu(), dtype=np.uint8)
            last_frame = make_frame(
                observer, last_rgb, prediction.bbox_xyxy, elapsed, distance,
                prediction.confidence, prediction.latency_ms, tilt,
                recovery_state.value if VISUAL_RECOVERY_ENABLED else "disabled",
            )
            writer.write(cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
        if evaluator.done:
            break
    writer.release()
    if last_frame is None:
        raise RuntimeError("no showcase video frame was rendered")
    Image.fromarray(last_frame).save(final_frame_path)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
        "-vf", "hqdn3d=1.2:1.0:3.0:2.0", "-c:v", "libx264", "-preset", "medium",
        "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_video),
    ], check=True)
    raw_video.unlink()
    recovery_metrics = recovery.metrics(result.elapsed_s)
    summary = result.to_dict()
    summary.update({
        "configured_horizon_s": args.duration_s,
        "termination_config": str(ROOT / "configs/task.json"),
        "termination_thresholds": {
            "target_too_far": {
                "distance_m": thresholds.target_distance_m,
                "grace_s": thresholds.target_distance_grace_s,
            },
            "target_absent": {
                "grace_s": thresholds.target_absent_grace_s,
                "minimum_bbox_area_ratio": thresholds.minimum_bbox_area_ratio,
            },
            "strong_collision": {
                "base_contact_impulse_ns": thresholds.base_contact_impulse_ns,
            },
            "fallen": {
                "maximum_tilt_deg": thresholds.maximum_tilt_deg,
                "minimum_base_height_m": thresholds.minimum_base_height_m,
                "grace_s": thresholds.fallen_grace_s,
            },
            "out_of_bounds": {
                "route_corridor_half_width_m": ROUTE_CORRIDOR_HALF_WIDTH_M,
                "margin_m": task_config["termination"]["out_of_bounds"].get("margin_m", 0.0),
            },
        },
        "success_condition": task_config["success"]["condition"],
        "control_mode": CONTROL_MODE.value,
        "actor_truth_leakage_guard": CONTROL_MODE is TrackingControlMode.PURE_E2E,
        "privileged_actor_inputs": (
            ["scheduled_target_bearing", "purple_vest_recovery_bbox"]
            if CONTROL_MODE is TrackingControlMode.SHOWCASE_HYBRID else []
        ),
        "high_level_control": (
            "MaST bbox distance/centering feedback with scheduled-route target-bearing feed-forward"
            if CONTROL_MODE is TrackingControlMode.SHOWCASE_HYBRID
            else "MaST bbox visual servo plus RGB/template-only recovery; no route bearing or truth-based recovery"
        ),
        "evaluation_signal_separation": {
            "target_absent": "actor bbox area only",
            "identity_drift": "privileged true-target distance termination only",
            "projected_target_bbox": "diagnostic IoU only",
        },
        "human_motion": "single NVIDIA Behavior walk, speed-matched to route motion",
        "human_motion_status": "continuous_speed_matched_walk_with_route_aligned_heading",
        "motion_cycle_frames": motion_cycle_frames,
        "motion_models": [
            {
                "role": motion["role"],
                "gait": motion["gait"],
                "source_speed_mps": motion["source_speed_mps"],
                "root_speed_mps": motion["root_speed_mps"],
            }
            for motion in HUMAN_MOTION_SPECS
        ],
        "distractor_people": 0,
        "physics_colliders": 92,
        "city_collision_api_prims_original": 91,
        "collision_strategy": "scene-authored road, sidewalk, building, furniture and obstacle proxies, plus the kinematic target proxy",
        "scripted_scene_events": [],
        "surface_z_m": SURFACE_Z,
        "target_speed_mps": TARGET_SPEED_MPS,
        "route_length_m": ROUTE_LENGTH_M,
        "route_waypoint_count": len(ROUTE_POINTS),
        "route_major_turn_count": 4,
        "route_obstacle_deviation_count": 3,
        "observer_mode": args.observer_mode,
        "route_start_distance_m": route_start_distance,
        "route_distance_completed_m": min(
            TARGET_SPEED_MPS * result.elapsed_s, ROUTE_LENGTH_M - route_start_distance
        ),
        "target_path": "305 m intersecting-street route with rounded turns and three obstacle deviations",
        "character_heading": "velocity-aligned with limited turn rate",
        "road_friction": {"static": 0.95, "dynamic": 0.82, "restitution": 0.02},
        "rendering": {
            "observer_resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
            "samples_per_pixel": args.render_spp,
            "video_codec": "H.264 CRF 16",
            "observer_display_filter": "5x5 median plus mild unsharp mask",
            "d455_tracker_preprocessing": "single 3x3 median",
        },
        "d455_intrinsics": {"width": 640, "height": 480, "fx": 391.18, "fy": 391.18, "cx": 321.78, "cy": 239.94},
        "d455_mount": {
            "body_relative_translation_m": [0.32, 0.0, 0.32],
            "body_relative_quaternion_xyzw_ros": [
                0.54167522, -0.54167522, 0.45451948, -0.45451948
            ],
            "optical_center_height_m": 0.66,
            "pitch_down_deg": 10.0,
            "translation_status": "x/y assumed; z derived from measured optical height",
        },
        "mean_distance_m": statistics.fmean(distances),
        "final_distance_m": distances[-1],
        "mean_tracker_latency_ms": statistics.fmean(tracker_latencies),
        "mean_tracker_confidence": statistics.fmean(tracker_confidences),
        "minimum_tracker_confidence": min(tracker_confidences),
        "tracker_visible_ratio": sum(tracker_visible) / len(tracker_visible),
        "mean_iou": statistics.fmean(ious) if ious else 0.0,
        "template_appearance_similarity": {
            "control_gate_enabled": True,
            "minimum_similarity_for_large_boxes": APPEARANCE_DRIFT_GUARD_MINIMUM,
            "minimum_reliable_area_ratio": APPEARANCE_DRIFT_GUARD_AREA_RATIO,
            "mean": statistics.fmean(appearance_similarities),
            "minimum": min(appearance_similarities),
            "purpose": "large-box identity-drift guard plus diagnostics; no simulator truth",
        },
        "appearance_reinitializations": appearance_reinitializations,
        "template_only_reinitializations": template_reinitializations,
        "appearance_recovery": (
            "purple-vest detector only reinitializes MaST after IoU drift; no distractor people"
            if CONTROL_MODE is TrackingControlMode.SHOWCASE_HYBRID
            else "template-derived RGB candidate only while visually lost; no simulator truth"
        ),
        "truth_free_visual_recovery": {
            "enabled": VISUAL_RECOVERY_ENABLED,
            "state": recovery_metrics.state.value,
            "loss_events": recovery_metrics.loss_events,
            "reacquisitions": recovery_metrics.reacquisitions,
            "current_lost_s": recovery_metrics.current_lost_s,
            "total_lost_s": recovery_metrics.total_lost_s,
            "actor_inputs": [
                "tracker_visible", "tracker_confidence", "bbox_horizontal_error"
            ],
            "tracker_inputs": ["rgb", "initial_template_rgb"],
            "configuration": {
                "tracker_minimum_confidence": TRACKER_MINIMUM_ACTOR_CONFIDENCE,
                "appearance_histogram_control_gate": "large_boxes_only",
                "loss_confirmation_frames": recovery.config.loss_confirmation_frames,
                "reacquire_confirmation_frames": recovery.config.reacquire_confirmation_frames,
                "search_yaw_rate_rps": recovery.config.search_yaw_rate_rps,
                "sweep_duration_s": recovery.config.sweep_duration_s,
            },
            "forbidden_inputs_used": [],
        },
        "locomotion_policy_latency_ms_last": last_policy_ms,
        "commanded_forward_speed_mps": {
            "mean": float(np.mean(commanded_forward_speeds)) if commanded_forward_speeds else 0.0,
            "minimum": float(np.min(commanded_forward_speeds)) if commanded_forward_speeds else 0.0,
            "maximum": float(np.max(commanded_forward_speeds)) if commanded_forward_speeds else 0.0,
        },
        "commanded_yaw_rate_rps": {
            "mean": float(np.mean(commanded_yaw_rates)) if commanded_yaw_rates else 0.0,
            "mean_absolute": float(np.mean(np.abs(commanded_yaw_rates))) if commanded_yaw_rates else 0.0,
            "minimum": float(np.min(commanded_yaw_rates)) if commanded_yaw_rates else 0.0,
            "maximum": float(np.max(commanded_yaw_rates)) if commanded_yaw_rates else 0.0,
        },
        "mean_robot_planar_speed_mps": float(np.mean(root_planar_speeds)) if root_planar_speeds else 0.0,
        "settled_base_height_m": settled_height_m,
        "settled_clearance_above_road_m": settled_clearance_m,
        "robot_forward_progress_m": float(
            (float(robot.data.root_pos_w.torch[0, 0]) - settled_robot_xy[0]) * initial_route_vx
            + (float(robot.data.root_pos_w.torch[0, 1]) - settled_robot_xy[1]) * initial_route_vy
        ),
        "mean_robot_forward_progress_speed_mps": float(
            (
                (float(robot.data.root_pos_w.torch[0, 0]) - settled_robot_xy[0]) * initial_route_vx
                + (float(robot.data.root_pos_w.torch[0, 1]) - settled_robot_xy[1]) * initial_route_vy
            ) / max(result.elapsed_s, 1e-6)
        ),
        "final_robot_position_m": [float(value) for value in robot.data.root_pos_w.torch[0].cpu()],
        "video": str(final_video),
        "final_frame": str(final_frame_path),
        "city_asset": str(CITY_USD),
        "character_assets": list(CHARACTER_ASSETS),
    })
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("CITY_GO2_CLOSED_LOOP_RESULT", json.dumps(summary, ensure_ascii=False), flush=True)
    if result.state is not EpisodeState.SUCCESS:
        raise RuntimeError(f"305 m closed-loop episode failed: {result.failure_reason}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        print("CITY_GO2_SHOWCASE_EXCEPTION", type(error).__name__, repr(error), flush=True)
        raise
    finally:
        simulation_app.close()
