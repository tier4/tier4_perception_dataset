# TIER IV Perception Dataset Conversion Tools

This repository provides tools for converting from various formats like `rosbags`, `deepen`, `fastlabel` to/from T4 dataset format. These tools facilitate the preparation and transformation of perception data for autonomous driving applications.
The T4 dataset format is itself defined and maintained in [t4-devkit](https://github.com/tier4/t4-devkit).


## Table of Contents

- [TIER IV Perception Dataset Conversion Tools](#tier-iv-perception-dataset-conversion-tools)
  - [Table of Contents](#table-of-contents)
  - [Dataset Overview](#dataset-overview)
  - [Usage of T4 format dataset](#usage-of-t4-format-dataset)
  - [Usage of Conversion Tools](#usage-of-conversion-tools)
    - [Conversion Tools Overview](#conversion-tools-overview)
    - [Setup](#setup)
    - [Test](#test)
      - [Download test data](#download-test-data)
      - [Run tests](#run-tests)
    - [Pre commit](#pre-commit)

## Dataset Overview

The T4 dataset is designed to support autonomous driving research and development.  
It can include various types of perception data such as images and lidar point clouds, radar point clouds, along with annotations for tasks like object detection, segmentation, and tracking.

For detailed information about the T4 dataset format, please refer to the [T4 Dataset Format Documentation](https://github.com/tier4/t4-devkit/blob/main/docs/schema/table.md).

## Usage of T4 format dataset

In addition to the format definition, [t4-devkit](https://tier4.github.io/t4-devkit) also includes various features (visualization, io, sanity checks etc) for working with T4 datasets.  

## Usage of Conversion Tools

### Conversion Tools Overview

See [tools_overview](docs/tools_overview.md) about the converters.

### Setup

```bash
git clone git@github.com:tier4/tier4_perception_dataset.git perception_dataset
cd perception_dataset
```

Install and build ros dependencies (this step must be outside of poetry virtualenv):

```bash
source /opt/ros/${ROS_DISTRO}/setup.sh
sudo apt install -y ros-${ROS_DISTRO}-sensor-msgs-py ros-${ROS_DISTRO}-rosbag2-storage-mcap ros-${ROS_DISTRO}-radar-msgs

mkdir src -p && vcs import src < build_depends.repos
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to autoware_auto_perception_msgs autoware_perception_msgs tier4_perception_msgs autoware_sensing_msgs oxts_msgs vehicle_msgs
source ./install/setup.bash
```

As of 2024/06/10, the repository requires both `autoware_msgs` and `autoware_auto_msgs`. The above command will install both messages.
If you already have either of them, you can remove the unnecessary one from `build_depends.repos`.

Install python dependencies:

```bash
pip3 install poetry
poetry install
```

### Test

#### Download test data

- [GitHub CLI](https://github.com/cli/cli#installation)

```shell
gh release download t4-sample-0-2026-01-30 -D tests/data/t4_sample_0
unzip 'tests/data/t4_sample_0/*.zip' -d tests/data/t4_sample_0/
```

or manually download zipped data from [the release page](https://github.com/tier4/tier4_perception_dataset/releases/tag/t4-sample-0-2026-01-30) to a `test/data` directory

#### Run tests

```bash
source /opt/ros/${ROS_DISTRO}/setup.sh
source ./install/setup.bash
poetry run pytest -v tests/
```

### Pre commit

```bash
# to install hooks of formatter and linter files
pre-commit install

# to apply hooks
pre-commit run -a
```
