#!/usr/bin/env python3
"""Fetch the 2K CC0 assets used by the outdoor all-terrain map batch."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_hd_street_assets as assets


TEXTURE_IDS = (
    "forest_ground_04",
    "rocky_trail",
    "muddy_tracks",
    "playground_sand",
    "snow_02",
    "coast_sand_rocks_02",
)
MODEL_IDS = (
    "boulder_01",
    "rock_09",
    "dead_tree_trunk",
    "tree_stump_01",
    "pine_tree_01",
    "modular_wooden_pier",
    "sand_rocks_small_01",
    "coast_rocks_01",
)
HDRI_IDS = (
    "forest_grove",
    "fouriesburg_mountain_midday",
    "monbachtal_riverbank",
    "dry_field",
    "snowy_hillside",
    "river_rocks",
)


def fetch_hdri(asset_id: str) -> list[dict]:
    files = assets.get_json(f"{assets.API_ROOT}/files/{asset_id}")
    record = assets.find_record(files, "hdri", "2k", "exr")
    destination = assets.ASSET_ROOT / "hdris" / asset_id / Path(record["url"]).name
    assets.download(record["url"], destination)
    return [{
        "path": str(destination.relative_to(ROOT)),
        "bytes": destination.stat().st_size,
        "sha256": assets.sha256(destination),
    }]


def main() -> None:
    manifest_path = assets.ASSET_ROOT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "source": "Poly Haven public API",
            "api_credit": "Powered by Poly Haven — https://polyhaven.com",
            "asset_license": "CC0",
            "license_url": "https://polyhaven.com/license",
            "resolution": "2k",
        }
    for section in ("models", "textures", "hdris"):
        manifest.setdefault(section, {})
    for asset_id in TEXTURE_IDS:
        print("FETCH_OUTDOOR_TEXTURE", asset_id, flush=True)
        manifest["textures"][asset_id] = assets.fetch_texture(asset_id)
    for asset_id in MODEL_IDS:
        print("FETCH_OUTDOOR_MODEL", asset_id, flush=True)
        manifest["models"][asset_id] = assets.fetch_model(asset_id)
    for asset_id in HDRI_IDS:
        print("FETCH_OUTDOOR_HDRI", asset_id, flush=True)
        manifest["hdris"][asset_id] = fetch_hdri(asset_id)
    manifest["outdoor_batch"] = {
        "textures": list(TEXTURE_IDS),
        "models": list(MODEL_IDS),
        "hdris": list(HDRI_IDS),
        "license": "CC0",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("OUTDOOR_BATCH_ASSETS_OK", manifest_path, flush=True)


if __name__ == "__main__":
    main()
