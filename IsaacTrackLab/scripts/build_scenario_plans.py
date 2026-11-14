#!/usr/bin/env python3
"""Expand all six configured scenario families into deterministic plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaac_track_lab.spec import load_json
from isaac_track_lab.task import build_scenario_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/scenario_plans.json",
    )
    args = parser.parse_args()

    config = load_json("scenarios.json")
    plans = [
        build_scenario_plan(
            item,
            config["occlusion_event"],
            args.seed + index,
            config.get("disturbance_event"),
            config.get("long_horizon", {}).get("checkpoint_interval_s", 300),
        ).to_dict()
        for index, item in enumerate(config["scenarios"])
    ]
    payload = {
        "schema_version": 1,
        "base_seed": args.seed,
        "plans": plans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("SCENARIO_PLANS_OK")
    for plan in plans:
        print(
            plan["scenario_id"],
            f"duration={plan['duration_s']:.0f}s",
            f"distractors={plan['distractor_count']}",
            f"occlusions={len(plan['occlusion_events'])}",
            f"disturbances={len(plan['disturbance_events'])}",
        )
    print("output", args.output)


if __name__ == "__main__":
    main()
