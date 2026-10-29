#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${TRAIN_PYTHON:-/home/abc/Documents/go2_supermimic_runtime/physx/bin/python}" -m go2_physx_cts.launch --robot go2 "$@"
