#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DISTRO="${1:-humble}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

mkdir -p "${ROOT_DIR}/data"

if [[ "$#" -gt 0 ]]; then
  CMD=("$@")
else
  CMD=(bash)
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" \
  --network host \
  -v "${ROOT_DIR}:/workspace" \
  -v "${ROOT_DIR}/data:/workspace/data" \
  -w /workspace \
  "t4dataset_rosbag_converter:${DISTRO}" \
  "${CMD[@]}"
