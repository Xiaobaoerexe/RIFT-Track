# Training-environment upload manifest

This snapshot contains the executable project source rather than a copy of the
local 41 GB runtime installation.

## Included

- `source/isaac_track_lab`: environment, real Isaac backend, task and control code.
- `configs`: frozen D455 values, `sim2real_v1`, nominal profile and task configs.
- `scripts`: training/smoke entry points, asset builders and report validators.
- `tests`: 63 regression and sim-to-real contract tests plus the small frozen
  baseline JSON fixture required by the contract test.
- `assets/scenes`: deterministic terrain/route definitions and generated-route
  metadata. Generated USD/USDc files are rebuilt locally and are not committed.
- External-asset manifests and source filters, but no restricted binary assets.

## Deliberately excluded

- `.venv`, Isaac Sim package caches, shader caches and `third_party` checkouts.
- `outputs`, checkpoints, ONNX files, videos and offline rendered RGB-D frames.
- Generated USD/USDa/USDc stages.
- NVIDIA character and animation binaries, whose redistribution is governed by
  NVIDIA/Omniverse terms.
- Downloaded Poly Haven binaries; they are reproducibly fetched from the public
  API and their CC0 source IDs/hashes are recorded in the included manifest.

## Recreate the environment

```bash
cd IsaacTrackLab
bash scripts/bootstrap_third_party.sh
# Install Isaac Lab/Isaac Sim following the pinned IsaacLab instructions.
# From that Python environment:
python scripts/fetch_hd_street_assets.py
PYTHONPATH=source python -m unittest discover -s tests -v
```

NVIDIA People assets must be obtained through the licensed Isaac Sim asset
distribution and placed at the paths recorded in `configs/assets.json`. Do not
silently substitute unrelated characters or report missing assets as executed.

The actual smoke command is documented in `docs/SIM2REAL_V1.md`. Output paths
are non-overwriting and remain ignored by Git.
