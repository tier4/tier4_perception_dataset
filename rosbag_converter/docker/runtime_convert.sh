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

CONFIG_PATH="${T4_CONVERTER_CONFIG:-src/tier4_perception_dataset/rosbag_converter/docs/sample_config.yaml}"
INDIVIDUAL_PARAMS_ROOT="${T4_INDIVIDUAL_PARAMS_ROOT:-src/autoware_individual_params.j6_gen2}"
SENSOR_MODEL="${T4_SENSOR_MODEL:-aip_x2_gen2}"

if [[ "$#" -eq 0 ]]; then
  cat >&2 <<EOF
Usage:
  docker/runtime_convert.sh --input data/<bag> --output data/<output> --vehicle-id <vehicle_id> [converter args...]

Defaults added by this wrapper:
  --config ${CONFIG_PATH}
  --individual-params-root ${INDIVIDUAL_PARAMS_ROOT}
  --sensor-model ${SENSOR_MODEL}

Host data directory:
  DATA_DIR=${DATA_DIR}

Examples:
  docker/runtime_convert.sh \\
    --input data/2a1835a0-ae9f-42c2-bd1c-04f159d43d79 \\
    --output data/t4dataset_full_run \\
    --vehicle-id j6_gen2_03 \\
    --progress-interval 100 \\
    --pcd-output-format pcd \\
    --debug-timing

  DATA_DIR=/media/.../playback_less docker/runtime_convert.sh \\
    --input /data/input_bag_dir \\
    --output /data/output_dir \\
    --vehicle-id j6_gen2_03
EOF
  exit 2
fi

has_arg() {
  local flag="$1"
  shift
  for arg in "$@"; do
    if [[ "${arg}" == "${flag}" || "${arg}" == "${flag}="* ]]; then
      return 0
    fi
  done
  return 1
}

mkdir -p "${DATA_DIR}"

CONVERTER_ARGS=()
if ! has_arg "--config" "$@"; then
  CONVERTER_ARGS+=(--config "${CONFIG_PATH}")
fi
if ! has_arg "--individual-params-root" "$@"; then
  CONVERTER_ARGS+=(--individual-params-root "${INDIVIDUAL_PARAMS_ROOT}")
fi
if ! has_arg "--sensor-model" "$@"; then
  CONVERTER_ARGS+=(--sensor-model "${SENSOR_MODEL}")
fi
CONVERTER_ARGS+=("$@")

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" \
  --network host \
  -v "${DATA_DIR}:/opt/t4_ws/data" \
  -v "${DATA_DIR}:/data" \
  -w /opt/t4_ws \
  "${IMAGE_TAG}" \
  t4dataset-rosbag-converter "${CONVERTER_ARGS[@]}"
