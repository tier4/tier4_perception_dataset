# T4Dataset Rosbag Converter

Offline raw rosbag to T4Dataset converter.

This subproject reads rosbag2 data programmatically through the rosbag2 API and
uses Nebula and Autoware pointcloud libraries directly from Python. It does not
require `ros2 launch`, `ros2 bag play`, or `bag_step_executor` during
conversion.

## Layout

- `t4dataset_rosbag_converter/`: converter Python package.
- `docs/`: design notes, sample config, and binding examples.
- `docker/`: reproducible runtime Docker build and run scripts.
- `scripts/`: comparison and utility scripts.
- `tests/`: focused converter tests.

## Local Usage

From this directory:

```bash
uv sync --python "$(which python3)"
source /opt/ros/humble/setup.bash
source ../../../install/setup.bash
UV_NO_SYNC=1 uv run t4dataset-rosbag-converter --help
```

Typical conversion:

```bash
UV_NO_SYNC=1 uv run t4dataset-rosbag-converter \
  --config docs/sample_config.yaml \
  --input /data/<rosbag_dir> \
  --output /data/<output_dir> \
  --individual-params-root src/autoware_individual_params.j6_gen2 \
  --vehicle-id j6_gen2_03 \
  --sensor-model aip_x2_gen2 \
  --pcd-output-format pcd \
  --debug-timing
```

## Runtime Docker

Build:

```bash
docker/runtime_build.sh
```

Convert:

```bash
docker/runtime_convert.sh \
  --input data/<rosbag_dir> \
  --output data/<output_dir> \
  --vehicle-id j6_gen2_03 \
  --pcd-output-format pcd \
  --debug-timing
```

For data outside this subproject:

```bash
DATA_DIR=/path/to/data/root docker/runtime_convert.sh \
  --input /data/<rosbag_dir> \
  --output /data/<output_dir> \
  --vehicle-id j6_gen2_03
```
