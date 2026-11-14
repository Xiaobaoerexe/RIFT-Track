#!/usr/bin/env python3
"""Create a local walking-only Behavior Motion Library from NVIDIA's full library."""

from __future__ import annotations

from pathlib import Path

from pxr import Sdf, Usd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/external/nvidia/people/HumanMotionLibrary.usd"
OUTPUT = ROOT / "assets/external/nvidia/people/HumanMotionLibrary_walk.usda"
REMOTE_BASE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.1/Isaac/People/MotionLibrary/"
)
KEEP_ACTION_GROUPS = {"Idle", "MoveWalk"}
KEEP_CLIPS = {
    "Idle": {"Idle"},
    "MoveWalk": {
        "WalkArc",
        "WalkForward",
        "WalkStart",
        "WalkStop",
        "WalkTurnInPlace",
        "WalkTurns",
    },
}


def main() -> None:
    source_stage = Usd.Stage.Open(str(SOURCE), load=Usd.Stage.LoadNone)
    if source_stage is None:
        raise RuntimeError(f"Could not open {SOURCE}")

    output_layer = Sdf.Layer.CreateNew(str(OUTPUT))
    Sdf.CopySpec(
        source_stage.GetRootLayer(),
        Sdf.Path("/HumanMotionLibrary"),
        output_layer,
        Sdf.Path("/HumanMotionLibrary"),
    )
    output_layer.defaultPrim = "HumanMotionLibrary"
    output_layer.Save()

    stage = Usd.Stage.Open(str(OUTPUT), load=Usd.Stage.LoadNone)
    actions_path = Sdf.Path("/HumanMotionLibrary/BuiltinActions")
    actions = stage.GetPrimAtPath(actions_path)
    for child in list(actions.GetAllChildren()):
        if child.GetName() not in KEEP_ACTION_GROUPS:
            stage.RemovePrim(child.GetPath())
            continue
        for clip in list(child.GetAllChildren()):
            if clip.GetName() not in KEEP_CLIPS[child.GetName()]:
                stage.RemovePrim(clip.GetPath())

    rewritten = 0
    for prim in stage.TraverseAll():
        if not prim.HasAuthoredPayloads():
            continue
        payload_op = prim.GetMetadata("payload")
        payloads = payload_op.GetAddedOrExplicitItems() if payload_op else []
        if not payloads:
            continue
        prim.GetPayloads().ClearPayloads()
        for payload in payloads:
            asset_path = payload.assetPath
            while asset_path.startswith("./"):
                asset_path = asset_path[2:]
            prim.GetPayloads().AddPayload(
                Sdf.Payload(REMOTE_BASE + asset_path, payload.primPath, payload.layerOffset)
            )
            rewritten += 1

    stage.GetRootLayer().Save()
    print(
        "MINIMAL_HUMAN_MOTION_LIBRARY_OK",
        f"groups={sorted(KEEP_ACTION_GROUPS)}",
        f"payloads={rewritten}",
        OUTPUT,
        flush=True,
    )


if __name__ == "__main__":
    main()
