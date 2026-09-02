#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${RUNTIME_BUILD_CONFIG:-${SCRIPT_DIR}/runtime_build.yaml}"

if [[ -f "${CONFIG_FILE}" ]]; then
  eval "$(python3 "${SCRIPT_DIR}/parse_runtime_build_yaml.py" "${CONFIG_FILE}")"
fi

ROS_DISTRO="${ROS_DISTRO:-humble}"
IMAGE_NAME="${IMAGE_NAME:-t4dataset_rosbag_converter}"
IMAGE_TAG="${IMAGE_TAG:-${IMAGE_NAME}:${ROS_DISTRO}-runtime}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"

mkdir -p "${DATA_DIR}"

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

if [[ "$#" -gt 0 ]]; then
  CMD=("$@")
else
  CMD=(bash)
fi

docker run --rm "${TTY_ARGS[@]}" \
  --network host \
  -v "${DATA_DIR}:/opt/t4_ws/data" \
  -v "${DATA_DIR}:/data" \
  -w /opt/t4_ws \
  "${IMAGE_TAG}" \
  "${CMD[@]}"
