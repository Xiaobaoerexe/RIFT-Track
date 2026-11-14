#!/usr/bin/env python3
"""Run the actual Isaac RGB-D training backend or a short smoke episode."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from isaac_track_lab.sim2real.config import REQUIRED_SWITCHES, load_profile
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--training-profile", default="sim2real_v1")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--map", dest="map_id", choices=("city_01", "park_01", "forest_trail_01"), default="city_01")
parser.add_argument("--route", type=int, default=1)
parser.add_argument("--scenario-id", default="short_none")
parser.add_argument("--duration-s", type=float, default=2.0)
parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=(0.0, 0.0, 0.0))
parser.add_argument("--measured-inference-ms", type=float)
parser.add_argument("--offline-high-quality", action="store_true")
parser.add_argument(
    "--capture-every-camera", action="store_true",
    help="save every 30 Hz degraded camera frame instead of only 10 Hz actor observations",
)
parser.add_argument("--capture-dir", type=Path)
parser.add_argument("--output", type=Path)
for switch in sorted(REQUIRED_SWITCHES):
    parser.add_argument(f"--disable-{switch.replace('_', '-')}", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.duration_s <= 0.0:
    parser.error("--duration-s must be positive")
if not 1 <= args.route <= 20:
    parser.error("--route must be in 1..20")

app_launcher = AppLauncher(args, headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np
import torch
from PIL import Image

from isaac_track_lab.envs import (
    EpisodeRequest, IsaacBackendOptions, IsaacTrackingBackend,
    PlanarVelocityAction, StandardTrackingEnv,
)
from isaac_track_lab.task import TerminationThresholds


def save_observation(root: Path, index: int, observation) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rgb = torch.as_tensor(observation.rgb).detach().cpu().numpy().astype(np.uint8)
    depth_mask = torch.as_tensor(observation.aligned_depth).detach().cpu().numpy()
    Image.fromarray(rgb).save(root / f"rgb_{index:06d}.png")
    np.save(root / f"depth_m_{index:06d}.npy", depth_mask[..., 0].astype(np.float32))
    Image.fromarray((depth_mask[..., 1] > 0.5).astype(np.uint8) * 255).save(
        root / f"depth_valid_{index:06d}.png"
    )


def save_processed(
    root: Path, index: int, processed, timestamp_s: float, raw_rgb, raw_depth
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rgb = processed.rgb.detach().cpu().numpy().astype(np.uint8)
    depth = processed.depth_m.detach().cpu().numpy().astype(np.float32)
    mask = processed.depth_valid_mask.detach().cpu().numpy()
    Image.fromarray(rgb).save(root / f"rgb_{index:06d}.png")
    raw_rgb_np = torch.as_tensor(raw_rgb).detach().cpu().numpy()[..., :3].astype(np.uint8)
    raw_depth_np = torch.as_tensor(raw_depth).detach().cpu().numpy().astype(np.float32)
    Image.fromarray(raw_rgb_np).save(root / f"rgb_raw_{index:06d}.png")
    np.save(root / f"depth_raw_m_{index:06d}.npy", raw_depth_np)
    np.save(root / f"depth_m_{index:06d}.npy", depth)
    Image.fromarray(mask.astype(np.uint8) * 255).save(root / f"depth_valid_{index:06d}.png")
    (root / f"timestamp_{index:06d}.json").write_text(
        json.dumps({"capture_time_s": timestamp_s, "sensor_metadata": processed.metadata}, indent=2)
        + "\n"
    )


def main() -> None:
    profile = load_profile(args.training_profile, seed=args.seed)
    disabled = {
        name for name in REQUIRED_SWITCHES
        if getattr(args, f"disable_{name}")
    }
    profile = profile.with_disabled(disabled)
    if args.capture_dir and args.capture_dir.exists() and any(args.capture_dir.iterdir()):
        raise FileExistsError(
            f"capture directory is not empty; refusing to overwrite: {args.capture_dir}"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        ROOT / "outputs/sim2real_training" /
        f"{args.map_id}_{args.scenario_id}_{profile.name}_seed{args.seed}_{stamp}.json"
    )
    if output.exists():
        raise FileExistsError(f"report already exists; refusing to overwrite: {output}")
    options = IsaacBackendOptions(
        map_id=args.map_id, route_index=args.route, device=args.device,
        offline_high_quality=args.offline_high_quality,
    )
    backend = IsaacTrackingBackend.create(profile, options, simulation_app=simulation_app)
    camera_capture_count = 0
    if args.capture_dir and args.capture_every_camera:
        def capture_callback(processed, timestamp_s, raw_rgb, raw_depth):
            nonlocal camera_capture_count
            save_processed(
                args.capture_dir, camera_capture_count, processed, timestamp_s,
                raw_rgb, raw_depth,
            )
            camera_capture_count += 1
        backend.set_capture_callback(capture_callback)
    thresholds = TerminationThresholds.from_task_config(
        json.loads((ROOT / "configs/task.json").read_text())
    )
    env = StandardTrackingEnv(backend, thresholds)
    observation, _ = env.reset(EpisodeRequest(args.scenario_id, args.seed, args.duration_s))
    if args.capture_dir and not args.capture_every_camera:
        save_observation(args.capture_dir, 0, observation)
    command = PlanarVelocityAction(*map(float, args.command))
    steps = int(np.ceil(args.duration_s * profile.section("timing")["policy_hz"]))
    completed_steps = 0
    terminal = None
    for index in range(steps):
        if args.measured_inference_ms is not None:
            backend.set_measured_inference_latency_ms(args.measured_inference_ms)
        observation, reward, terminated, truncated, info = env.step(command)
        completed_steps += 1
        if args.capture_dir and not args.capture_every_camera:
            save_observation(args.capture_dir, index + 1, observation)
        if terminated or truncated:
            terminal = info.get("episode_result")
            break
    report = backend.report()
    report.update({
        "requested_duration_s": args.duration_s,
        "completed_policy_steps": completed_steps,
        "terminal_result": terminal,
        "camera_frames_saved": camera_capture_count if args.capture_every_camera else None,
        "capture_dir": str(args.capture_dir) if args.capture_dir else None,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ISAAC_SIM2REAL_REPORT", output, flush=True)
    print(json.dumps({
        "profile": report["profile"], "seed": report["seed"],
        "executed_event_count": report["executed_event_count"],
        "truth_leakage_detected": report["truth_leakage_detected"],
        "drop_rate": report["timing"]["drop_rate"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close may terminate Kit before Python prints a pending
        # traceback, so emit it explicitly first.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
