#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p runs
nohup /home/abc/Documents/go2_supermimic_runtime/physx/bin/python -u run_blind_sequence.py > runs/sequence_supervisor.log 2>&1 < /dev/null &
echo "$!" > runs/sequence_supervisor.pid
printf 'Sequence supervisor PID: %s\n' "$(cat runs/sequence_supervisor.pid)"
