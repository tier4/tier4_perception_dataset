#!/usr/bin/env bash
set -eo pipefail

cd "${1:-/workspace}"

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

git config --global --add safe.directory '*'

if [[ -f pyproject.toml ]]; then
  UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/t4dataset_uv_${ROS_DISTRO}}" \
    UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
    uv sync --python "$(which python3)"
fi

if [[ -f src/autoware_universe/build_depends_stable.repos ]]; then
  vcs import --skip-existing src < src/autoware_universe/build_depends_stable.repos
fi

mapfile -t ROSDEP_PATHS < <(
  colcon list --paths-only \
    --packages-up-to \
      autoware_pcl_extensions \
      autoware_pointcloud_preprocessor \
      individual_params \
      nebula_lib \
      t4dataset_rosbag_converter
)

apt-get update
rosdep install \
  --from-paths "${ROSDEP_PATHS[@]}" \
  --ignore-src \
  --rosdistro "${ROS_DISTRO}" \
  -y \
  --skip-keys "ament_python cuda blackfly_camera_driver nebula_common nebula_decoders nebula_msgs"

colcon --log-base "/tmp/t4dataset_log_${ROS_DISTRO}" build \
  --symlink-install \
  --build-base "/tmp/t4dataset_build_${ROS_DISTRO}" \
  --install-base "/tmp/t4dataset_install_${ROS_DISTRO}" \
  --packages-up-to \
    autoware_pcl_extensions \
    autoware_pointcloud_preprocessor \
    individual_params \
    nebula_lib \
    t4dataset_rosbag_converter \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DNEBULA_LIB_BUILD_PYTHON=ON

set +u
source "/tmp/t4dataset_install_${ROS_DISTRO}/setup.bash"
set -u
ros2 run t4dataset_rosbag_converter smoke_imports
ros2 run t4dataset_rosbag_converter convert_raw_rosbag --help >/tmp/t4dataset_ros_cli_help.txt
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/t4dataset_uv_${ROS_DISTRO}}" \
  uv run t4dataset-rosbag-converter --help >/tmp/t4dataset_uv_cli_help.txt
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/t4dataset_uv_${ROS_DISTRO}}" \
  uv run t4dataset-rosbag-converter \
    --config src/tier4_perception_dataset/rosbag_converter/docs/sample_config.yaml \
    --input data/sample \
    --output /tmp/t4dataset_out_${ROS_DISTRO} \
    --individual-params-root src/autoware_individual_params.j6_gen2 \
    --vehicle-id j6_gen2_72 \
    --sensor-model aip_x2_gen2 \
    --dry-run
