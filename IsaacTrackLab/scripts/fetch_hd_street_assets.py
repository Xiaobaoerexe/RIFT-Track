#!/usr/bin/env python3
"""Download the licensed HD assets used by the compact tracking street.

Run this script from the pinned Isaac Lab environment because it uses pxr to
read the texture dependencies authored inside Poly Haven USD model files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from pxr import Sdf, Usd


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets/external/polyhaven"
USER_AGENT = "IsaacTrackLab/1.0 (research asset preparation; Powered by Poly Haven)"
API_ROOT = "https://api.polyhaven.com"

MODEL_IDS = (
    "concrete_road_barrier",
    "covered_car",
    "fire_hydrant",
    "metal_trash_can",
    "painted_wooden_bench",
    "planter_box_01",
)

TEXTURE_IDS = (
    "asphalt_02",
    "brick_pavement_02",
    "brick_wall_001",
    "beige_wall_001",
)

HDRI_ID = "urban_street_01"


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def iter_file_records(value):
    if isinstance(value, dict):
        if "url" in value:
            yield value
        else:
            for child in value.values():
                yield from iter_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_file_records(child)


def find_record(files: dict, *parts: str) -> dict:
    value = files
    for part in parts:
        value = value[part]
    if "url" not in value:
        raise KeyError(f"Poly Haven file record has no URL at {'/'.join(parts)}")
    return value


def find_by_basename(files: dict, basename: str) -> dict:
    matches = [
        record for record in iter_file_records(files)
        if Path(record["url"]).name == basename
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one download for {basename}, got {len(matches)}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model(asset_id: str) -> list[dict]:
    files = get_json(f"{API_ROOT}/files/{asset_id}")
    model_dir = ASSET_ROOT / "models" / asset_id
    usd_record = find_record(files, "usd", "2k", "usd")
    usd_path = model_dir / Path(usd_record["url"]).name
    download(usd_record["url"], usd_path)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open downloaded model: {usd_path}")
    texture_names: set[str] = set()
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            if value and value.path:
                texture_names.add(Path(value.path).name)

    outputs = [usd_path]
    for texture_name in sorted(texture_names):
        record = find_by_basename(files, texture_name)
        destination = model_dir / "textures" / texture_name
        download(record["url"], destination)
        outputs.append(destination)
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in outputs
    ]


def fetch_texture(asset_id: str) -> list[dict]:
    files = get_json(f"{API_ROOT}/files/{asset_id}")
    texture_dir = ASSET_ROOT / "textures" / asset_id
    selections = (
        ("Diffuse", "2k", "jpg"),
        ("Rough", "2k", "jpg"),
        ("nor_gl", "2k", "jpg"),
    )
    outputs = []
    for selection in selections:
        record = find_record(files, *selection)
        destination = texture_dir / Path(record["url"]).name
        download(record["url"], destination)
        outputs.append(destination)
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in outputs
    ]


def fetch_hdri() -> list[dict]:
    files = get_json(f"{API_ROOT}/files/{HDRI_ID}")
    # Poly Haven HDRIs expose the unclipped lighting image under "hdri".
    record = find_record(files, "hdri", "2k", "exr")
    destination = ASSET_ROOT / "hdris" / HDRI_ID / Path(record["url"]).name
    download(record["url"], destination)
    return [
        {
            "path": str(destination.relative_to(ROOT)),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    ]


def main() -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source": "Poly Haven public API",
        "api_credit": "Powered by Poly Haven — https://polyhaven.com",
        "asset_license": "CC0",
        "license_url": "https://polyhaven.com/license",
        "resolution": "2k",
        "models": {},
        "textures": {},
        "hdris": {},
    }
    for asset_id in MODEL_IDS:
        print("FETCH_MODEL", asset_id, flush=True)
        manifest["models"][asset_id] = fetch_model(asset_id)
    for asset_id in TEXTURE_IDS:
        print("FETCH_TEXTURE", asset_id, flush=True)
        manifest["textures"][asset_id] = fetch_texture(asset_id)
    print("FETCH_HDRI", HDRI_ID, flush=True)
    manifest["hdris"][HDRI_ID] = fetch_hdri()
    manifest_path = ASSET_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("HD_STREET_ASSETS_OK", manifest_path, flush=True)


if __name__ == "__main__":
    main()
