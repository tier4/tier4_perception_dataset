# Tools for Traffic Light pseudo labeling

This document describes the conversion from a rosbag with traffic light recognition (TLR) results of Autoware to a T4 dataset with pseudo-labels for traffic light recognition.

## Common

Those commands below are asuumed to be run in poetry shell built in [README.md](../README.md)  
Run this to start a poetry shell.

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
source {AUTOWARE_WORKSPACE}/install/setup.bash
poetry shell
```

## Convert rosbag2 (with Autoware TLR result in) to T4 dataset with Autoware reuslt (T4 tlr-semi-annotated format data)

input: rosbag2

output: T4 tlr-semi-annotated format data

```bash
python -m perception_dataset.convert --config config/rosbag2_to_t4/convert_rosbag2_to_annotated_t4_tlr.yaml
# if you want to overwrite t4-format data, use --overwrite option
```

## Convert T4 dataset with Autoware reuslt (T4 tlr-semi-annotated format data) to deepen importable json format label

input: T4 tlr-semi-annotated format data

output: deepen importable tlr-semi-annotated label

```bash
python -m perception_dataset.convert --config config/convert_annotated_t4_tlr_to_deepen.yaml
```

## Convert rosbag2 and ground truth yaml file, created by tlr color labeler, to T4 dataset with color label

input: rosbag2, label yaml file (TLR color labeler result)

output: T4 format dataset

```bash
python -m perception_dataset.convert --config config/rosbag2_to_t4/convert_rosbag2_with_gt_to_annotated_t4_tlr.yaml
# if you want to overwrite t4-format data, use --overwrite option
```
