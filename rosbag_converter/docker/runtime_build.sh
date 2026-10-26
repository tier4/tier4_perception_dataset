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

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required runtime build setting: ${name}" >&2
    echo "Set it in ${CONFIG_FILE}, or export ${name} in the environment." >&2
    exit 2
  fi
}

for setting in \
  AUTOWARE_REPO AUTOWARE_REF \
  AIP_LAUNCHER_REPO AIP_LAUNCHER_REF \
  NEBULA_REPO NEBULA_REF \
  INDIVIDUAL_PARAMS_REPO INDIVIDUAL_PARAMS_REF \
  T4_DEVKIT_REPO T4_DEVKIT_REF \
  TIER4_PERCEPTION_DATASET_REPO TIER4_PERCEPTION_DATASET_REF; do
  require_var "${setting}"
done

if CONVERTER_REF="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null)"; then
  if ! git -C "${ROOT_DIR}" diff --quiet -- .; then
    CONVERTER_REF="${CONVERTER_REF}-dirty"
  fi
else
  CONVERTER_REF="local"
fi

SSH_ARGS=()
if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
  SSH_ARGS=(--ssh default)
fi

EXTRA_BUILD_ARGS=()
if [[ -n "${DOCKER_BUILD_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_BUILD_ARGS=(${DOCKER_BUILD_EXTRA_ARGS})
fi

DOCKER_BUILDKIT=1 docker build "${SSH_ARGS[@]}" \
  "${EXTRA_BUILD_ARGS[@]}" \
  -f "${SCRIPT_DIR}/Dockerfile.runtime" \
  --target runtime \
  --build-arg "ROS_DISTRO=${ROS_DISTRO}" \
  --build-arg "AUTOWARE_REPO=${AUTOWARE_REPO}" \
  --build-arg "AUTOWARE_REF=${AUTOWARE_REF}" \
  --build-arg "AIP_LAUNCHER_REPO=${AIP_LAUNCHER_REPO}" \
  --build-arg "AIP_LAUNCHER_REF=${AIP_LAUNCHER_REF}" \
  --build-arg "NEBULA_REPO=${NEBULA_REPO}" \
  --build-arg "NEBULA_REF=${NEBULA_REF}" \
  --build-arg "INDIVIDUAL_PARAMS_REPO=${INDIVIDUAL_PARAMS_REPO}" \
  --build-arg "INDIVIDUAL_PARAMS_REF=${INDIVIDUAL_PARAMS_REF}" \
  --build-arg "T4_DEVKIT_REPO=${T4_DEVKIT_REPO}" \
  --build-arg "T4_DEVKIT_REF=${T4_DEVKIT_REF}" \
  --build-arg "TIER4_PERCEPTION_DATASET_REPO=${TIER4_PERCEPTION_DATASET_REPO}" \
  --build-arg "TIER4_PERCEPTION_DATASET_REF=${TIER4_PERCEPTION_DATASET_REF}" \
  --build-arg "CONVERTER_REF=${CONVERTER_REF}" \
  -t "${IMAGE_TAG}" \
  "${ROOT_DIR}"

echo "Built ${IMAGE_TAG}"
