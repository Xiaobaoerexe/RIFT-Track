#!/usr/bin/env python3
"""Benchmark the deployed Go2 locomotion policy on an empty flat PhysX plane.

This intentionally contains no camera, tracker, person, route, obstacle, or task
termination logic.  It measures the policy/robot/ground contract in isolation.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import acos, degrees
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--command-vx-mps", type=float, default=1.25)
parser.add_argument("--command-yaw-rps", type=float, default=0.0)
parser.add_argument("--duration-s", type=float, default=12.0)
parser.add_argument("--settle-s", type=float, default=2.0)
parser.add_argument("--measure-after-s", type=float, default=2.0)
parser.add_argument("--static-friction", type=float, default=0.95)
parser.add_argument("--dynamic-friction", type=float, default=0.82)
parser.add_argument("--restitution", type=float, default=0.02)
parser.add_argument(
    "--duplicate-ground", action="store_true",
    help="Add a second coplanar collision plane to diagnose duplicate contacts.",
)
parser.add_argument(
    "--environment", choices=("flat", "city"), default="flat",
    help="Use an empty plane or the authored 305 m city without any task actors.",
)
parser.add_argument(
    "--city-support-plane", action="store_true",
    help="With --environment city, add the same continuous z=0.20 support plane as the showcase.",
)
parser.add_argument(
    "--app-pump-hz", type=float, default=0.0,
    help="Call simulation_app.update() at this rate without changing USD Timeline.",
)
parser.add_argument("--output-dir", default="outputs/locomotion_benchmark")
parser.add_argument("--output-stem", default="go2_flat_ground")
parser.add_argument("--gui", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.duration_s <= args.measure_after_s or args.settle_s < 0.0:
    parser.error("duration must exceed measure-after and settle must be non-negative")
if min(args.static_friction, args.dynamic_friction, args.restitution) < 0.0:
    parser.error("material coefficients must be non-negative")

app_launcher = AppLauncher(args, headless=not args.gui)
simulation_app = app_launcher.app

import numpy as np
import omni.usd
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaac_track_lab.control import Go2LocomotionPolicy, POLICY_JOINT_NAMES


LOCOMOTION_ROOT = (
    ROOT.parent / "MaSTrack/robot_runtime/models/locomotion_candidates/kaiwu2025_v6-2_124004"
)
DT = 0.005
SURFACE_Z = 0.0

GO2_CFG = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
GO2_CFG.init_state.pos = (
    (-55.0, -36.0, 0.70) if args.environment == "city" else (0.0, 0.0, 0.50)
)
GO2_CFG.actuators["base_legs"].stiffness = 20.0
GO2_CFG.actuators["base_legs"].damping = 0.5


@configclass
class FlatGroundSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/FlatGround",
        spawn=sim_utils.GroundPlaneCfg(
            visible=True,
            size=(80.0, 20.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args.static_friction,
                dynamic_friction=args.dynamic_friction,
                restitution=args.restitution,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                0.0,
                0.0,
                0.20 if args.environment == "city" and args.city_support_plane
                else 0.0 if args.environment == "flat" else -10.0,
            )
        ),
    )
    duplicate_ground = AssetBaseCfg(
        prim_path="/World/DuplicateFlatGround",
        spawn=sim_utils.GroundPlaneCfg(
            visible=False,
            size=(80.0, 20.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args.static_friction,
                dynamic_friction=args.dynamic_friction,
                restitution=args.restitution,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, SURFACE_Z if args.duplicate_ground else -10.0)
        ),
    )
    robot: ArticulationCfg = GO2_CFG


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def main() -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.environment == "city":
        stage = omni.usd.get_context().get_stage()
        city_prim = stage.DefinePrim("/World/HDStreet", "Xform")
        city_prim.GetReferences().AddReference(
            str(ROOT / "assets/scenes/hd_route_city/hd_route_city.usda")
        )
        for _ in range(8):
            simulation_app.update()

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    scene = InteractiveScene(FlatGroundSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot = scene["robot"]
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

    zero_command = np.zeros(3, dtype=np.float32)
    target = default_joints[0, joint_ids].cpu().numpy().copy()
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

    settled_pos = robot.data.root_pos_w.torch[0].cpu().numpy().copy()
    settled_height = float(settled_pos[2])
    command = np.array((args.command_vx_mps, 0.0, args.command_yaw_rps), dtype=np.float32)
    samples: list[dict[str, float]] = []
    joint_actual: list[np.ndarray] = []
    joint_target: list[np.ndarray] = []
    torque_saturation: list[float] = []
    contact_slip_speeds: list[float] = []
    touchdown_x: list[float] = []
    previous_fl_contact = False
    # The Go2 foot body origin sits slightly above the sole. This threshold is
    # used only as a diagnostic contact proxy, not for simulation control.
    contact_height_m = (0.20 if args.environment == "city" else 0.0) + 0.055
    effort_limits = np.array(
        [23.7, 23.7, 45.43] * 4, dtype=np.float32
    )

    for step in range(round(args.duration_s / DT)):
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

        elapsed = (step + 1) * DT
        if args.app_pump_hz > 0.0:
            pump_steps = max(1, round((1.0 / args.app_pump_hz) / DT))
            if step % pump_steps == 0:
                simulation_app.update(); scene.update(DT)
        root_pos = robot.data.root_pos_w.torch[0].cpu().numpy()
        root_vel = robot.data.root_lin_vel_w.torch[0].cpu().numpy()
        gravity = robot.data.projected_gravity_b.torch[0]
        tilt = degrees(acos(float(torch.clamp(-gravity[2], -1.0, 1.0))))
        feet_pos = robot.data.body_pos_w.torch[0, foot_ids].cpu().numpy()
        feet_vel = robot.data.body_lin_vel_w.torch[0, foot_ids].cpu().numpy()
        contacts = feet_pos[:, 2] <= contact_height_m
        for index in np.flatnonzero(contacts):
            contact_slip_speeds.append(float(np.linalg.norm(feet_vel[index, :2])))
        fl_contact = bool(contacts[0])
        if fl_contact and not previous_fl_contact:
            touchdown_x.append(float(root_pos[0]))
        previous_fl_contact = fl_contact

        actual = robot.data.joint_pos.torch[0, joint_ids].cpu().numpy().copy()
        applied = np.abs(robot.data.applied_torque.torch[0, joint_ids].cpu().numpy())
        torque_saturation.append(float(np.mean(applied >= 0.98 * effort_limits)))
        joint_actual.append(actual)
        joint_target.append(target.copy())
        if step % policy_steps == 0:
            samples.append({
                "time_s": elapsed,
                "x_m": float(root_pos[0]),
                "y_m": float(root_pos[1]),
                "base_height_m": float(root_pos[2]),
                "vx_mps": float(root_vel[0]),
                "vy_mps": float(root_vel[1]),
                "tilt_deg": tilt,
                "contact_feet": int(np.sum(contacts)),
            })

    actual_array = np.asarray(joint_actual)
    target_array = np.asarray(joint_target)
    start_index = min(len(samples) - 1, round(args.measure_after_s / policy.step_dt))
    measured = samples[start_index:]
    measure_start_x = measured[0]["x_m"]
    measure_end_x = measured[-1]["x_m"]
    measure_dt = measured[-1]["time_s"] - measured[0]["time_s"]
    strides = np.diff(touchdown_x).tolist() if len(touchdown_x) >= 2 else []
    cycle_periods = []
    # Samples are 50 Hz, but touchdown events were collected at physics rate.
    # Estimate cycle period from stride and measured forward speed when both are usable.
    mean_speed = (measure_end_x - measure_start_x) / max(measure_dt, 1e-6)
    if abs(mean_speed) > 1e-3:
        cycle_periods = [abs(value / mean_speed) for value in strides]

    result = {
        "test": "standalone_go2_locomotion",
        "contains_task_logic": False,
        "duplicate_coplanar_ground": args.duplicate_ground,
        "environment": args.environment,
        "city_support_plane": args.city_support_plane,
        "app_pump_hz": args.app_pump_hz,
        "policy": str(LOCOMOTION_ROOT / "exported/policy.onnx"),
        "policy_step_s": policy.step_dt,
        "physics_step_s": DT,
        "command_vx_mps": args.command_vx_mps,
        "command_yaw_rps": args.command_yaw_rps,
        "duration_s": args.duration_s,
        "settle_s": args.settle_s,
        "measurement_window_s": [args.measure_after_s, args.duration_s],
        "ground_material": {
            "static_friction": args.static_friction,
            "dynamic_friction": args.dynamic_friction,
            "restitution": args.restitution,
            "combine_mode": "PhysX default (average unless the foot material overrides it)",
        },
        "robot_model": "Isaac Lab UNITREE_GO2_CFG / NVIDIA Go2 USD",
        "actuator": {
            "type": "Isaac Lab DCMotorCfg with effort and velocity saturation",
            "stiffness": 20.0,
            "damping": 0.5,
            "effort_limits_nm_policy_order": effort_limits.tolist(),
        },
        "settled_base_height_m": settled_height,
        "forward_distance_m": measure_end_x - measure_start_x,
        "mean_forward_speed_mps": mean_speed,
        "speed_command_ratio": mean_speed / args.command_vx_mps if args.command_vx_mps else None,
        "mean_abs_lateral_speed_mps": statistics.fmean(abs(row["vy_mps"]) for row in measured),
        "lateral_drift_m": measured[-1]["y_m"] - measured[0]["y_m"],
        "base_height_m": {
            "mean": statistics.fmean(row["base_height_m"] for row in measured),
            "minimum": min(row["base_height_m"] for row in measured),
        },
        "tilt_deg": {
            "mean": statistics.fmean(row["tilt_deg"] for row in measured),
            "maximum": max(row["tilt_deg"] for row in measured),
        },
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean((actual_array - target_array) ** 2))),
        "joint_peak_to_peak_rad": {
            name: {
                "target": float(np.ptp(target_array[:, index])),
                "actual": float(np.ptp(actual_array[:, index])),
            }
            for index, name in enumerate(POLICY_JOINT_NAMES)
        },
        "torque_saturation_fraction": statistics.fmean(torque_saturation),
        "foot_contact_proxy": {
            "height_threshold_m": contact_height_m,
            "contact_samples": len(contact_slip_speeds),
            "slip_speed_mean_mps": statistics.fmean(contact_slip_speeds) if contact_slip_speeds else None,
            "slip_speed_p95_mps": percentile(contact_slip_speeds, 95),
            "fl_touchdown_count": len(touchdown_x),
            "stride_length_mean_m": statistics.fmean(strides) if strides else None,
            "gait_cycle_period_mean_s": statistics.fmean(cycle_periods) if cycle_periods else None,
        },
        "fallen": min(row["base_height_m"] for row in measured) < 0.28
        or max(row["tilt_deg"] for row in measured) > 60.0,
    }

    json_path = output_dir / f"{args.output_stem}.json"
    csv_path = output_dir / f"{args.output_stem}.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    print("GO2_LOCOMOTION_BENCHMARK_RESULT", json.dumps(result, ensure_ascii=False), flush=True)
    print(f"GO2_LOCOMOTION_BENCHMARK_FILES json={json_path} csv={csv_path}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
