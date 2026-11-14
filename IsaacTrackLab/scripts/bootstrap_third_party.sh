#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${PROJECT_ROOT}/third_party"
mkdir -p "${THIRD_PARTY_ROOT}"

checkout_dependency() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local destination="${THIRD_PARTY_ROOT}/${name}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
}

checkout_dependency \
  "IsaacLab" \
  "https://github.com/isaac-sim/IsaacLab.git" \
  "ae37b028ea415c91ea2bc32609efcd759ed2b974"
checkout_dependency \
  "unitree_ros" \
  "https://github.com/unitreerobotics/unitree_ros.git" \
  "15ebe53f4cd77e7d745ae83b9d7ff90e685e8998"

echo "Pinned third-party source checkouts are ready under ${THIRD_PARTY_ROOT}."
