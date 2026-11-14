#!/usr/bin/env python3
"""Fetch the additional CC0 materials and HDRI used by terrain dataset v2."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_hd_street_assets as assets
from fetch_outdoor_batch_assets import fetch_hdri


TEXTURE_IDS = ("leafy_grass", "rock_path")
HDRI_IDS = ("je_gray_park",)


def main() -> None:
    manifest_path = assets.ASSET_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for asset_id in TEXTURE_IDS:
        print("FETCH_TERRAIN_V2_TEXTURE", asset_id, flush=True)
        manifest.setdefault("textures", {})[asset_id] = assets.fetch_texture(asset_id)
    for asset_id in HDRI_IDS:
        print("FETCH_TERRAIN_V2_HDRI", asset_id, flush=True)
        manifest.setdefault("hdris", {})[asset_id] = fetch_hdri(asset_id)
    manifest["terrain_dataset_v2"] = {
        "textures": list(TEXTURE_IDS),
        "hdris": list(HDRI_IDS),
        "license": "CC0",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("TERRAIN_V2_ASSETS_OK", manifest_path, flush=True)


if __name__ == "__main__":
    main()
