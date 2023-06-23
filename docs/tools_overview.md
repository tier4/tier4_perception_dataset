# Tools Overview

This document is simply written about the script.

![data_collection_conversion](data_collection_conversion.drawio.svg)

## Common

Those commands below are asuumed to be run in poetry shell built in [README.md](../README.md)  
Run this to start a poetry shell.

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
source ${ROS_WORKSPACE_WITH_CUSTOM_MESSAGES}/install/setup.bash
poetry shell
```

## rosbag2 to T4 non-annotated format data

input: rosbag2

output: T4 non-annotated format data

```bash
python -m perception_dataset.convert --config config/convert_rosbag2_to_non_annotated_t4_sample.yaml
# if you want to overwrite t4-format data, use --overwrite option
```

## Deepen

### T4 format to Deepen format

input: T4 format data

output: deepen-format data

```bash
python -m perception_dataset.convert --config config/convert_t4_to_deepen_sample.yaml
```

### Download Deepen annotations

`DEEPEN_CLIENT_ID` is the `xxx` part of the URL `https://tools.deepen.ai/workspace/xxx/datasets` after logging in to Deepen.
`DEEPEN_ACCESS_TOKEN` can be obtained from [Deepen Tools](https://tools.deepen.ai/workspace/xxx/developer/tokens/developers).

```bash
export DEEPEN_CLIENT_ID='YOUR_DEEPEN_CLIENT_ID'
export DEEPEN_ACCESS_TOKEN='YOUR_DEEPEN_ACCESS_TOKEN'
python -m perception_dataset.deepen.download_annotations --config config/convert_deepen_to_t4_sample.yaml
```

### Deepen format to T4 format

input: T4 non-annotated format data + deepen annotations

output: T4 format data

```bash
python -m perception_dataset.convert --config config/convert_deepen_to_t4_sample.yaml
```

## Rosbag with objects

### Synthetic bag to T4 format

see [About Synthetic Data](about_synthetic_data.md)

input: rosbag2

output: T4 format data

#### Messages

| Topic Name                                                  | Required | Message Type                                      |
| ----------------------------------------------------------- | -------- | ------------------------------------------------- |
| `/ground_truth/filtered/objects` or `/ground_truth/objects` | o        | `autoware_perception_msgs/msg/DynamicObjectArray` |
| `/sensing/lidar/concatenated/pointcloud`                    | o        | `sensor_msgs/msg/PointCloud2`                     |
| `/tf`                                                       | o        | `tf2_msgs/msg/TFMessage`                          |
| `/tf_static`                                                | o        | `tf2_msgs/msg/TFMessage`                          |
|                                                             |          | `sensor_msgs/msg/CompressedImage`                 |
|                                                             |          | `sensor_msgs/msg/CameraInfo`                      |

#### script

```bash
python -m perception_dataset.convert --config config/rosbag2_to_t4/convert_synthetic_data.yaml
```

### Pseudo-labeled bag to T4 format

input: rosbag2

output: T4 format data

#### Messages

| Topic Name                                                                   | Required | Message Type                                                                                              |
| ---------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `/perception/object_recognition/detection/apollo/objects` or other any value | o        | `autoware_auto_perception_msgs/msg/TrackedObjects` or `autoware_auto_perception_msgs/msg/DetectedObjects` |
| `/sensing/lidar/concatenated/pointcloud` or other any value                  | o        | `sensor_msgs/msg/PointCloud2`                                                                             |
| `/tf`                                                                        | o        | `tf2_msgs/msg/TFMessage`                                                                                  |
| `/tf_static`                                                                 | o        | `tf2_msgs/msg/TFMessage`                                                                                  |
| `/sensing/camera/camera{ID}/image_rect_color/compressed`                     |          | `sensor_msgs/msg/CompressedImage`                                                                         |
| `/sensing/camera/camera{ID}/camera_info`                                     |          | `sensor_msgs/msg/CameraInfo`                                                                              |

#### script

```bash
python -m perception_dataset.convert --config config/rosbag2_to_t4/convert_synthetic_data.yaml
```
