#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "$#" -gt 0 ]]; then
  DISTROS=("$@")
else
  DISTROS=(humble jazzy)
fi

for distro in "${DISTROS[@]}"; do
  docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --build-arg "ROS_DISTRO=${distro}" \
    -t "t4dataset_rosbag_converter:${distro}" \
    "${SCRIPT_DIR}"
done
