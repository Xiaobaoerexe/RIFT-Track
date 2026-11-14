#!/usr/bin/env python3
"""Build compact Behavior libraries backed by distinct NVIDIA gait clips."""

from __future__ import annotations

from pathlib import Path

from pxr import Sdf, Usd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/external/nvidia/people/HumanMotionLibrary.usd"
OUTPUT_DIR = ROOT / "assets/external/nvidia/people"
REMOTE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.1/Isaac/People/MotionLibrary/BuiltinActions"
)
ACTION_NAMES = (
    "WalkArc", "WalkForward", "WalkStart", "WalkStop", "WalkTurnInPlace", "WalkTurns"
)
VARIANTS = {
    "slow": {
        "WalkArc": "MoveWalkSlow/WalkSlowTurns.usd",
        "WalkForward": "MoveWalkSlow/WalkSlowForward.usd",
        "WalkStart": "MoveWalkSlow/WalkSlowForward.usd",
        "WalkStop": "MoveWalkSlow/WalkSlowForward.usd",
        "WalkTurnInPlace": "MoveWalkSlow/WalkSlowTurns.usd",
        "WalkTurns": "MoveWalkSlow/WalkSlowTurns.usd",
    },
    "jog": {
        "WalkArc": "MoveJog/JogTurns.usd",
        "WalkForward": "MoveJog/JogForward.usd",
        "WalkStart": "MoveJog/JogStart.usd",
        "WalkStop": "MoveJog/JogStop.usd",
        "WalkTurnInPlace": "MoveJog/JogTurns.usd",
        "WalkTurns": "MoveJog/JogTurns.usd",
    },
}


def build_variant(name: str, mapping: dict[str, str]) -> Path:
    output = OUTPUT_DIR / f"HumanMotionLibrary_{name}.usda"
    source_stage = Usd.Stage.Open(str(SOURCE), load=Usd.Stage.LoadNone)
    if source_stage is None:
        raise RuntimeError(f"Could not open {SOURCE}")
    layer = Sdf.Layer.CreateNew(str(output))
    Sdf.CopySpec(
        source_stage.GetRootLayer(), Sdf.Path("/HumanMotionLibrary"),
        layer, Sdf.Path("/HumanMotionLibrary"),
    )
    layer.defaultPrim = "HumanMotionLibrary"
    layer.Save()
    stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadNone)
    actions = stage.GetPrimAtPath("/HumanMotionLibrary/BuiltinActions")
    for group in list(actions.GetAllChildren()):
        if group.GetName() not in {"Idle", "MoveWalk"}:
            stage.RemovePrim(group.GetPath())
            continue
        keep = {"Idle"} if group.GetName() == "Idle" else set(ACTION_NAMES)
        for clip in list(group.GetAllChildren()):
            if clip.GetName() not in keep:
                stage.RemovePrim(clip.GetPath())
    for action, relative_url in mapping.items():
        prim = stage.GetPrimAtPath(f"/HumanMotionLibrary/BuiltinActions/MoveWalk/{action}")
        prim.GetPayloads().ClearPayloads()
        prim.GetPayloads().AddPayload(f"{REMOTE}/{relative_url}")
    idle = stage.GetPrimAtPath("/HumanMotionLibrary/BuiltinActions/Idle/Idle")
    idle.GetPayloads().ClearPayloads()
    idle.GetPayloads().AddPayload(f"{REMOTE}/Idle/Idle.usd")
    stage.GetRootLayer().Save()
    return output


def main() -> None:
    for name, mapping in VARIANTS.items():
        print("BEHAVIOR_VARIANT_OK", name, build_variant(name, mapping), flush=True)


if __name__ == "__main__":
    main()
