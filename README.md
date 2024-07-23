# TIER IV Perception Dataset Conversion Tools

This repository provides tools for converting ROS2 bag files and TIER IV (T4) dataset files, as well as the design specifications for the T4 dataset format.  
These tools facilitate the preparation and transformation of perception data for autonomous driving applications.

## Table of Contents

1. [Introduction and Purpose](#introduction-and-purpose)
2. [Dataset Overview](#dataset-overview)
3. [Usage of T4 dataset](#usage-of-t4-format-dataset)
4. [Usage of Conversion Tools](#usage-of-conversion-tools)
   - [Conversion Tools Overview](#conversion-tools-overview)
   - [Setup](#setup)
   - [Test](#test)
   - [Pre Commit](#pre-commit)

## Dataset Overview

The T4 dataset is designed to support autonomous driving research and development.  
It can include various types of perception data such as images and lidar point clouds, radar point clouds, along with annotations for tasks like object detection, segmentation, and tracking.

For detailed information about the T4 dataset format, please refer to the [T4 Dataset Format Documentation](docs/t4_format_3d_detailed.md).

## Usage of T4 format dataset

t4-devkit is a development kit for working with the T4 dataset, providing additional utilities and tools.  
Please see [t4-devkit](t4-devkit/README.md) about details.

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
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to tier4_perception_msgs pandar_msgs
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
gh release download test-data -D tests/data
unzip 'tests/data/*.zip' -d tests/data/
```

or manually download zipped data from [the release page](https://github.com/tier4/tier4_perception_dataset/releases/tag/test-data) to a `test/data` directory

#### Run tests

```bash
source /opt/ros/${ROS_DISTRO}/setup.sh
source ./install/setup.bash
poetry run pytest
```

### Pre commit

```bash
# to install hooks of formatter and linter files
pre-commit install

# to apply hooks
pre-commit run -a
```
