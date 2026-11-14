#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/abc/Documents/视觉跟踪/IsaacTrackLab"
cd "${PROJECT_ROOT}/third_party/IsaacLab"
exec env \
  OMNI_KIT_ACCEPT_EULA=yes \
  PYTHONPATH="${PROJECT_ROOT}/source" \
  uv run --extra isaacsim python \
  "${PROJECT_ROOT}/scripts/browse_terrain_dataset.py" "$@"
