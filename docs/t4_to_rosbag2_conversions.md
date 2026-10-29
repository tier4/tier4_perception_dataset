# Tools Overview

This document describes the reverse conversion, T4 format dataset to rosbag2.
Before reading this document, we recommend referring to the [Tools Overview](./tools_overview.md) for a comprehensive description of the entire repository.

## T4 format to rosbag2

input: T4 format dataset(s) (annotated or non-annotated)  
output: rosbag2 (one bag per T4 dataset)

The converter searches `input_base` recursively for T4 datasets, i.e. directories that contain `annotation/sample_data.json`, and writes one bag per dataset to `output_base/<path relative to input_base>`.

### Usage

1. Copy `config/t4_to_rosbag2/convert_t4_to_rosbag2.yaml` and edit `input_base` / `output_base`.
2. Source ROS 2 and run the converter:

```bash
source /opt/ros/humble/setup.bash
python -m perception_dataset.convert --config config/t4_to_rosbag2/convert_t4_to_rosbag2.yaml
# if you want to overwrite an existing output bag, use --overwrite option
```

Without `--overwrite`, the conversion fails if the output directory already exists.

`config/t4_to_rosbag2/convert_t4_to_rosbag2.yaml`:

```yaml
task: convert_t4_to_rosbag2
description:
  scene: t4_to_rosbag2
conversion:
  input_base: ./data/t4_dataset
  output_base: ./data/t4_to_rosbag2
  storage_id: sqlite3 # sqlite3 or mcap
  object_topic_name: /ground_truth/objects
  tracked_object_topic_name: /ground_truth/tracked_objects
  world_frame_id: map
  base_frame_id: base_link
  # Optional overrides, see "Topic names" in t4_to_rosbag2_conversions.md
  # lidar_sensor:
  #   topic: /sensing/lidar/concatenated/pointcloud
  #   channel: LIDAR_CONCAT
  # camera_sensors:
  #   - channel: CAM_FRONT
  #     topic: /sensing/camera/camera0/image_raw/compressed
```

### Config

See the YAML above. `storage_id` selects the rosbag2 storage plugin (`sqlite3` or `mcap`); `world_frame_id` / `base_frame_id` name the TF frames.

### Topic names

By default, the LiDAR and camera topics are read from the `status.json` that the rosbag2 to T4 conversion writes next to `annotation/` and `data/`.
Set `lidar_sensor` and/or `camera_sensors` in the config to override them. They are required when `status.json` is missing (e.g. datasets that were not created by this repository).

### Output topics

| Topic                                                | Type                                           | Source                                                                           |
| ---------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------- |
| LiDAR topic                                          | `sensor_msgs/msg/PointCloud2`                  | `data/LIDAR_CONCAT/*.pcd.bin`; float32 fields `x, y, z, intensity, ring`         |
| camera topic (`.../image_raw/compressed`)            | `sensor_msgs/msg/CompressedImage`              | `data/CAM_*/*.jpg`; the JPEG bytes are copied as they are (no decode / re-encode) |
| `.../camera_info` (derived from the camera topic)    | `sensor_msgs/msg/CameraInfo`                   | `calibrated_sensor.json` intrinsics; image size from `sample_data.json`          |
| `/tf`                                                | `tf2_msgs/msg/TFMessage`                       | `ego_pose.json`; `world_frame_id` to `base_frame_id`                             |
| `/tf_static`                                         | `tf2_msgs/msg/TFMessage`                       | `calibrated_sensor.json`; `base_frame_id` to each camera frame (transient local) |
| `object_topic_name`                                  | `autoware_perception_msgs/msg/DetectedObjects` | `sample_annotation.json`, one message per sample, in `world_frame_id`            |
| `tracked_object_topic_name`                          | `autoware_perception_msgs/msg/TrackedObjects`  | same annotations; `object_id` is the instance token as UUID                      |

Message and record timestamps are the `sample_data` timestamps (objects: the `sample` timestamp). Frame ids: the LiDAR uses `base_frame_id`, a camera uses its lower-cased channel name (e.g. `cam_front`).

### Limitations

The T4 format does not keep everything from the original rosbag, so the conversion is lossy:

- Images are the undistorted JPEGs, so `CameraInfo` has zero distortion.
- Point clouds only contain `x, y, z, intensity, ring`. Per-point timestamps and other fields are not recovered.
- Topics other than the ones above (e.g. CAN, vehicle status, original frame ids) are not recovered. To get the original recording, use `input_bag/` if the dataset contains it.

### Examples

Convert a single T4 dataset:

```bash
# input_base points directly at one dataset (a directory containing annotation/ and data/, or its parent)
cat > /tmp/t4_to_rosbag2.yaml <<'EOF'
task: convert_t4_to_rosbag2
description:
  scene: t4_to_rosbag2
conversion:
  input_base: /home/autoware/data/t4_dataset/db_jpntaxigen2_v2/15f81b3f-896b-40c1-9090-97cf029bec75
  output_base: /tmp/rosbag2_out
EOF
python -m perception_dataset.convert --config /tmp/t4_to_rosbag2.yaml
# -> /tmp/rosbag2_out/0/  (the dataset's version directory "0")
```

Convert many datasets at once, writing MCAP bags:

```yaml
conversion:
  input_base: /home/autoware/data/t4_dataset/db_jpntaxigen2_v2 # every dataset below is converted
  output_base: /home/autoware/data/rosbag2_from_t4
  storage_id: mcap
```

Dataset without `status.json`, topics given explicitly:

```yaml
conversion:
  input_base: ./data/t4_dataset
  output_base: ./data/t4_to_rosbag2
  lidar_sensor:
    topic: /sensing/lidar/concatenated/pointcloud
    channel: LIDAR_CONCAT
  camera_sensors:
    - channel: CAM_FRONT
      topic: /sensing/camera/camera0/image_raw/compressed
    - channel: CAM_BACK
      topic: /sensing/camera/camera1/image_raw/compressed
```

Inspect and play back the result:

```bash
ros2 bag info /tmp/rosbag2_out/0
ros2 bag play /tmp/rosbag2_out/0 --clock
# /tf_static is stored with transient_local QoS, so it is latched on playback
```

Compare the ground-truth objects in RViz or with `ros2 topic echo`:

```bash
ros2 topic echo /ground_truth/tracked_objects --once
```
