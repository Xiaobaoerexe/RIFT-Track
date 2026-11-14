#!/usr/bin/env python3
"""Load the converted As2 USD and hold a neutral stance for a bounded run."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=1000)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args = parser.parse_args()
args.physics = "isaacsim_physx"

import torch

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsCfg
from isaac_track_lab.assets import AS2_CFG


def main() -> None:
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics_cfg:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args.device, physics=physics_cfg)
        )
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        cfg = AS2_CFG.replace(prim_path="/World/As2")
        robot = cfg.class_type(cfg)
        sim.reset()

        joint_names = list(robot.data.joint_names)
        if len(joint_names) != 12:
            raise RuntimeError(f"Expected 12 actuated joints, got {len(joint_names)}: {joint_names}")

        target = robot.data.default_joint_pos.torch.clone()
        min_height = math.inf
        max_height = -math.inf
        for _ in range(args.steps):
            robot.actuators.target_command.set_position_index(value=target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            height = float(robot.data.root_pos_w.torch[0, 2])
            min_height = min(min_height, height)
            max_height = max(max_height, height)

        state_tensors = (robot.data.root_pos_w.torch, robot.data.joint_pos.torch, robot.data.joint_vel.torch)
        if not all(torch.isfinite(value).all() for value in state_tensors):
            raise RuntimeError("Non-finite As2 state encountered")
        if min_height < 0.25:
            raise RuntimeError(
                f"As2 fell below the smoke-test height threshold: min={min_height:.4f} m; "
                f"final joints={robot.data.joint_pos.torch[0].tolist()}"
            )

        print("AS2_SMOKE_OK")
        print("actuated_joint_count", len(joint_names))
        print("actuated_joint_names", ",".join(joint_names))
        print("base_height_range_m", f"{min_height:.4f}", f"{max_height:.4f}")
        print("final_base_position_m", robot.data.root_pos_w.torch[0].tolist())


if __name__ == "__main__":
    main()
