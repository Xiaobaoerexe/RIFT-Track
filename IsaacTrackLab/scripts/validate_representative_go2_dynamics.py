#!/usr/bin/env python3
"""Run route-following Go2 dynamics diagnostics on one realistic v3 map.

This is an environment/locomotion acceptance test, not an end-to-end tracking
task.  Route geometry is used only to generate diagnostic velocity commands.
"""

from __future__ import annotations

import argparse
import json
from math import acos, atan2, cos, degrees, hypot, sin
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

CATALOG = json.loads((ROOT / "configs/terrain_dataset_v3_realistic.json").read_text())
MAPS = {item["id"]: item for item in CATALOG["maps"]}

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--map", dest="map_id", choices=sorted(MAPS), default="city_01")
parser.add_argument("--route", type=int, default=1, help="one-based route index")
parser.add_argument("--duration-s", type=float, default=12.0)
parser.add_argument("--settle-s", type=float, default=2.0)
parser.add_argument("--command-speed-mps", type=float, default=1.0)
parser.add_argument("--lookahead-m", type=float, default=3.0)
parser.add_argument("--output-dir", default="outputs/terrain_dataset_v3_realistic/dynamics")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--diagnostic-local-collision-patch", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.route <= 20:
    parser.error("--route must be in 1..20")
if args.duration_s <= 0.0 or args.settle_s < 0.0:
    parser.error("duration must be positive and settle time non-negative")
if not 0.2 <= args.command_speed_mps <= 1.5:
    parser.error("diagnostic command speed must be in 0.2..1.5 m/s")

app_launcher = AppLauncher(args, headless=not args.gui)
simulation_app = app_launcher.app

import numpy as np
import omni.usd
import torch
from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaac_track_lab.control import Go2LocomotionPolicy, POLICY_JOINT_NAMES
from isaac_track_lab.task import terrain_height, terrain_mesh_height


DT = 0.005
GO2_FOOT_COLLISION_RADIUS_M = 0.022
MAP_CONFIG = MAPS[args.map_id]
MAP_DIR = ROOT / "assets/scenes/terrain_dataset_v3_realistic" / args.map_id
MANIFEST = json.loads((MAP_DIR / "scene_manifest.json").read_text())
ROUTES = json.loads((MAP_DIR / "routes/generated_routes.json").read_text())["routes"]
ROUTE = ROUTES[args.route - 1]
ROUTE_POINTS = np.asarray(ROUTE["points_m"], dtype=np.float64)
ROUTE_XY = ROUTE_POINTS[:, :2]
ROUTE_SEGMENT_LENGTHS = np.linalg.norm(np.diff(ROUTE_XY, axis=0), axis=1)
ROUTE_CUMULATIVE = np.concatenate(([0.0], np.cumsum(ROUTE_SEGMENT_LENGTHS)))
START = ROUTE_POINTS[0].copy()
START[2] = terrain_mesh_height(MAP_CONFIG, float(START[0]), float(START[1]))
START_HEADING = atan2(
    ROUTE_POINTS[min(3, len(ROUTE_POINTS) - 1), 1] - START[1],
    ROUTE_POINTS[min(3, len(ROUTE_POINTS) - 1), 0] - START[0],
)
LOCOMOTION_ROOT = (
    ROOT.parent / "MaSTrack/robot_runtime/models/locomotion_candidates/kaiwu2025_v6-2_124004"
)


GO2_CFG = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
GO2_CFG.init_state.pos = (float(START[0]), float(START[1]), float(START[2]) + 0.58)
GO2_CFG.init_state.rot = (
    float(cos(0.5 * START_HEADING)), 0.0, 0.0, float(sin(0.5 * START_HEADING))
)
GO2_CFG.actuators["base_legs"].stiffness = 20.0
GO2_CFG.actuators["base_legs"].damping = 0.5


@configclass
class RepresentativeDynamicsCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = GO2_CFG
    feet_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=DT,
        history_length=2,
        track_air_time=True,
    )


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def wrap_angle(value: float) -> float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def closest_route_index(x_m: float, y_m: float, previous: int) -> int:
    lower = max(0, previous - 3)
    upper = min(len(ROUTE_XY), previous + 25)
    distances = np.linalg.norm(ROUTE_XY[lower:upper] - np.array((x_m, y_m)), axis=1)
    return lower + int(np.argmin(distances))


def lookahead_index(progress_index: int, distance_m: float) -> int:
    target = ROUTE_CUMULATIVE[progress_index] + distance_m
    return min(int(np.searchsorted(ROUTE_CUMULATIVE, target)), len(ROUTE_XY) - 1)


def main() -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compose the static map before creating PhysX tensor views.  Spawning this
    # large mesh through an AssetBaseCfg and immediately resetting allowed the
    # robot to start while PhysX mesh cooking was still pending.
    stage = omni.usd.get_context().get_stage()
    map_prim = stage.DefinePrim("/World/Map", "Xform")
    map_prim.GetReferences().AddReference(str(MAP_DIR / MANIFEST["usd"]))
    if args.diagnostic_local_collision_patch:
        # Isolate large-mesh cooking failures from robot/policy failures.
        half, spacing = 10, 1.0
        nx = 2 * half + 1
        points = [
            Gf.Vec3f(
                float(START[0] + (ix - half) * spacing),
                float(START[1] + (iy - half) * spacing),
                terrain_height(
                    MAP_CONFIG,
                    float(START[0] + (ix - half) * spacing),
                    float(START[1] + (iy - half) * spacing),
                ),
            )
            for iy in range(nx) for ix in range(nx)
        ]
        indices = []
        for iy in range(nx - 1):
            for ix in range(nx - 1):
                lower = iy * nx + ix
                indices.extend((lower, lower + 1, lower + nx + 1,
                                lower, lower + nx + 1, lower + nx))
        patch = UsdGeom.Mesh.Define(stage, "/World/DiagnosticTerrainPatch")
        patch.CreateSubdivisionSchemeAttr("none")
        patch.CreatePointsAttr(points)
        patch.CreateFaceVertexCountsAttr([3] * (2 * (nx - 1) * (nx - 1)))
        patch.CreateFaceVertexIndicesAttr(indices)
        UsdPhysics.CollisionAPI.Apply(patch.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(patch.GetPrim()).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.none
        )
    for _ in range(30):
        simulation_app.update()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    scene = InteractiveScene(RepresentativeDynamicsCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot = scene["robot"]
    contact_sensor = scene["feet_contact"]
    joint_ids, joint_names = robot.find_joints(list(POLICY_JOINT_NAMES), preserve_order=True)
    if tuple(joint_names) != POLICY_JOINT_NAMES:
        raise RuntimeError(f"Go2 joint order mismatch: {joint_names}")
    foot_ids, foot_names = robot.find_bodies(".*_foot", preserve_order=True)
    if len(foot_ids) != 4:
        raise RuntimeError(f"expected four Go2 feet, got {foot_names}")

    policy = Go2LocomotionPolicy(
        LOCOMOTION_ROOT / "exported/policy.onnx",
        LOCOMOTION_ROOT / "params/deploy.yaml",
    )
    policy_steps = max(1, round(policy.step_dt / DT))
    default_joints = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=default_joints)
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.actuators.target_command.set_position_index(value=default_joints)
    scene.reset()

    target = default_joints[0, joint_ids].cpu().numpy().copy()
    zero_command = np.zeros(3, dtype=np.float32)
    for step in range(round(args.settle_s / DT)):
        if step % policy_steps == 0:
            out = policy.step(
                robot.data.root_ang_vel_b.torch[0].cpu().numpy(),
                robot.data.projected_gravity_b.torch[0].cpu().numpy(),
                zero_command,
                robot.data.joint_pos.torch[0, joint_ids].cpu().numpy(),
                robot.data.joint_vel.torch[0, joint_ids].cpu().numpy(),
            )
            target = out.joint_position_targets
            robot.actuators.target_command.set_position_index(
                value=torch.tensor(target[None], device=robot.device), joint_ids=joint_ids
            )
        scene.write_data_to_sim(); sim.step(); scene.update(DT)

    settled = robot.data.root_pos_w.torch[0].cpu().numpy().copy()
    settled_ground = terrain_mesh_height(MAP_CONFIG, float(settled[0]), float(settled[1]))
    settled_clearance = float(settled[2] - settled_ground)
    policy.reset()

    progress_index = closest_route_index(float(settled[0]), float(settled[1]), 0)
    initial_progress_m = float(ROUTE_CUMULATIVE[progress_index])
    samples: list[dict[str, float]] = []
    contact_slip_speeds: list[float] = []
    contact_force_samples: list[float] = []
    sole_clearances: list[float] = []
    commands: list[np.ndarray] = []

    for step in range(round(args.duration_s / DT)):
        root_pos = robot.data.root_pos_w.torch[0].cpu().numpy()
        progress_index = closest_route_index(float(root_pos[0]), float(root_pos[1]), progress_index)
        target_index = lookahead_index(progress_index, args.lookahead_m)
        target_xy = ROUTE_XY[target_index]
        desired_heading = atan2(target_xy[1] - root_pos[1], target_xy[0] - root_pos[0])
        heading = float(robot.data.heading_w.torch[0])
        heading_error = wrap_angle(desired_heading - heading)
        forward = args.command_speed_mps * max(0.35, cos(heading_error))
        yaw_rate = float(np.clip(1.8 * heading_error, -1.0, 1.0))
        command = np.array((forward, 0.0, yaw_rate), dtype=np.float32)
        commands.append(command.copy())
        if step % policy_steps == 0:
            out = policy.step(
                robot.data.root_ang_vel_b.torch[0].cpu().numpy(),
                robot.data.projected_gravity_b.torch[0].cpu().numpy(),
                command,
                robot.data.joint_pos.torch[0, joint_ids].cpu().numpy(),
                robot.data.joint_vel.torch[0, joint_ids].cpu().numpy(),
            )
            target = out.joint_position_targets
            robot.actuators.target_command.set_position_index(
                value=torch.tensor(target[None], device=robot.device), joint_ids=joint_ids
            )
        scene.write_data_to_sim(); sim.step(); scene.update(DT)

        feet_pos = robot.data.body_pos_w.torch[0, foot_ids].cpu().numpy()
        feet_vel = robot.data.body_lin_vel_w.torch[0, foot_ids].cpu().numpy()
        net_forces = contact_sensor.data.net_forces_w
        force_array = (
            net_forces.torch[0].cpu().numpy() if net_forces is not None else np.zeros((4, 3))
        )
        force_norms = np.linalg.norm(force_array, axis=1)
        for index, force_n in enumerate(force_norms):
            contact_force_samples.append(float(force_n))
            if force_n >= 5.0:
                contact_slip_speeds.append(float(np.linalg.norm(feet_vel[index, :2])))
        for foot in feet_pos:
            ground = terrain_mesh_height(MAP_CONFIG, float(foot[0]), float(foot[1]))
            # NVIDIA's Go2 foot collision is a 0.022 m sphere.  Subtract its
            # authored radius rather than the older 0.055 m contact-proxy
            # threshold, which overstated penetration by about 3.3 cm.
            sole_clearances.append(
                float(foot[2] - GO2_FOOT_COLLISION_RADIUS_M - ground)
            )

        if step % policy_steps == 0:
            root_pos = robot.data.root_pos_w.torch[0].cpu().numpy()
            root_vel = robot.data.root_lin_vel_w.torch[0].cpu().numpy()
            gravity = robot.data.projected_gravity_b.torch[0]
            tilt = degrees(acos(float(torch.clamp(-gravity[2], -1.0, 1.0))))
            cross_track = float(np.linalg.norm(root_pos[:2] - ROUTE_XY[progress_index]))
            ground = terrain_mesh_height(MAP_CONFIG, float(root_pos[0]), float(root_pos[1]))
            samples.append({
                "time_s": (step + 1) * DT,
                "route_progress_m": float(ROUTE_CUMULATIVE[progress_index]),
                "cross_track_error_m": cross_track,
                "base_clearance_m": float(root_pos[2] - ground),
                "planar_speed_mps": float(np.linalg.norm(root_vel[:2])),
                "tilt_deg": tilt,
                "contact_feet": int(np.sum(force_norms >= 5.0)),
            })

    measured_progress = float(ROUTE_CUMULATIVE[progress_index] - initial_progress_m)
    minimum_base_clearance = min(row["base_clearance_m"] for row in samples)
    maximum_tilt = max(row["tilt_deg"] for row in samples)
    fallen = minimum_base_clearance < 0.20 or maximum_tilt > 60.0
    p95_cross_track = percentile([row["cross_track_error_m"] for row in samples], 95)
    p95_slip = percentile(contact_slip_speeds, 95)
    contact_ratio = sum(row["contact_feet"] > 0 for row in samples) / len(samples)
    acceptance = {
        "settled_clearance_at_least_0_28_m": settled_clearance >= 0.28,
        "not_fallen": not fallen,
        "route_progress_at_least_40_percent_commanded": measured_progress >= 0.4 * args.command_speed_mps * args.duration_s,
        "cross_track_p95_below_2_5_m": p95_cross_track is not None and p95_cross_track <= 2.5,
        "true_foot_contact_observed": contact_ratio >= 0.90 and bool(contact_force_samples),
        "sole_penetration_p95_below_0_03_m": float(np.percentile(np.maximum(0.0, -np.asarray(sole_clearances)), 95)) <= 0.03,
    }
    result = {
        "test": "representative_v3_go2_dynamics",
        "task_test": False,
        "map_id": args.map_id,
        "family": MAP_CONFIG["family"],
        "route_index": args.route,
        "route_length_m": ROUTE["length_m"],
        "duration_s": args.duration_s,
        "settle_s": args.settle_s,
        "command_speed_mps": args.command_speed_mps,
        "physics_step_s": DT,
        "policy_step_s": policy.step_dt,
        "settled_base_clearance_m": settled_clearance,
        "route_progress_m": measured_progress,
        "mean_planar_speed_mps": statistics.fmean(row["planar_speed_mps"] for row in samples),
        "cross_track_error_m": {
            "mean": statistics.fmean(row["cross_track_error_m"] for row in samples),
            "p95": p95_cross_track,
            "maximum": max(row["cross_track_error_m"] for row in samples),
        },
        "base_clearance_m": {
            "minimum": minimum_base_clearance,
            "mean": statistics.fmean(row["base_clearance_m"] for row in samples),
        },
        "tilt_deg": {
            "mean": statistics.fmean(row["tilt_deg"] for row in samples),
            "maximum": maximum_tilt,
        },
        "foot_contact": {
            "force_threshold_n": 5.0,
            "sample_count": len(contact_force_samples),
            "contact_time_ratio": contact_ratio,
            "slip_speed_mean_mps": statistics.fmean(contact_slip_speeds) if contact_slip_speeds else None,
            "slip_speed_p95_mps": p95_slip,
            "sole_clearance_minimum_m": min(sole_clearances),
            "penetration_p95_m": float(np.percentile(np.maximum(0.0, -np.asarray(sole_clearances)), 95)),
        },
        "fallen": fallen,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
        "limitations": [
            "route geometry is a privileged diagnostic command source",
            "sole penetration subtracts the authored 0.022 m Go2 foot-sphere radius",
            "this run does not include cameras, tracking, target following, or task termination",
        ],
    }
    output_path = output_dir / f"{args.map_id}_route_{args.route:02d}_dynamics.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("REPRESENTATIVE_GO2_DYNAMICS", "PASS" if result["passed"] else "FAIL", json.dumps(result, ensure_ascii=False), flush=True)
    print("REPRESENTATIVE_GO2_DYNAMICS_FILE", output_path, flush=True)
    simulation_app.close()
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
