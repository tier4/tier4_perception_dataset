# tier4_perception_dataset

This is the data conversion tools around T4 dataset.

## Tools Overview

See [tools_overview](docs/tools_overview.md) about the converters.

## Setup

```bash
git clone git@github.com:tier4/tier4_perception_dataset.git perception_dataset
cd perception_dataset
```

install and build ros dependencies (this step must be outside of poetry virtualenv)

```bash
source /opt/ros/${ROS_DISTRO}/setup.sh
sudo apt install -y ros-${ROS_DISTRO}-sensor-msgs-py ros-${ROS_DISTRO}-rosbag2-storage-mcap ros-${ROS_DISTRO}-radar-msgs

mkdir src -p && vcs import src < build_depends.repos
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to tier4_perception_msgs
source ./install/setup.bash
```

install python dependencies

```bash
pip3 install poetry
poetry install
```

## Test

### Download test data

- [GitHub CLI](https://github.com/cli/cli#installation)

```shell
gh release download test-data -D tests/data
unzip 'tests/data/*.zip' -d tests/data/
```

or manually download zipped data from [the release page](https://github.com/tier4/tier4_perception_dataset/releases/tag/test-data) to a `test/data` directory

### Run tests

```bash
source /opt/ros/${ROS_DISTRO}/setup.sh
source ./install/setup.bash
poetry run pytest
```

## Pre commit

```bash
# to install hooks of formatter and linter files
pre-commit install

# to apply hooks
pre-commit run -a
```
