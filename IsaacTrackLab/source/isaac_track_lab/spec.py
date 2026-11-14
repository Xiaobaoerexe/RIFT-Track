from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_json(name: str) -> dict:
    with (ROOT / "configs" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate() -> list[str]:
    errors: list[str] = []
    task = load_json("task.json")
    scenarios = load_json("scenarios.json")
    assets = load_json("assets.json")

    if task["robot"]["model"] != "unitree_as2":
        errors.append("robot.model must be unitree_as2")
    urdf = ROOT / task["robot"]["urdf"]
    if not urdf.is_file():
        errors.append(f"As2 URDF is missing: {urdf}")

    expected = {(d, i) for d in (300, 3600) for i in ("none", "distractors", "distractors_occlusion")}
    actual = {(item["duration_s"], item["interference"]) for item in scenarios["scenarios"]}
    if actual != expected or len(scenarios["scenarios"]) != 6:
        errors.append("scenario matrix must contain exactly the 2x3 required combinations")
    ids = [item["id"] for item in scenarios["scenarios"]]
    if len(ids) != len(set(ids)):
        errors.append("scenario ids must be unique")
    long_horizon = scenarios.get("long_horizon", {})
    if long_horizon.get("required_duration_s") != 3600:
        errors.append("long_horizon.required_duration_s must be 3600")
    if long_horizon.get("checkpoint_interval_s") != 300:
        errors.append("long_horizon.checkpoint_interval_s must be 300")
    if not long_horizon.get("continuous_episode"):
        errors.append("long_horizon must be one continuous episode")
    if long_horizon.get("reset_between_checkpoints") is not False:
        errors.append("long_horizon checkpoints must not reset environment state")

    termination = task["termination"]
    for section, key in (("target_too_far", "distance_m"), ("target_absent", "grace_s"), ("fallen", "maximum_tilt_deg")):
        if termination[section][key] <= 0:
            errors.append(f"termination.{section}.{key} must be positive")

    if assets["robot"]["local_urdf"] != task["robot"]["urdf"]:
        errors.append("robot URDF path differs between task.json and assets.json")
    return errors


def main() -> None:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("OK: task, assets, As2 URDF, and all six 5-minute/1-hour scenarios are consistent")


if __name__ == "__main__":
    main()
