"""some implementations are from https://github.com/tier4/ros2bag_extensions/blob/main/ros2bag_extensions/ros2bag_extensions/verb/__init__.py"""

import os.path as osp
from pathlib import Path
from typing import Dict, Optional, Tuple

import builtin_interfaces.msg
import cv2
from nptyping import NDArray
import numpy as np
from rclpy.time import Time
from rosbag2_py import (
    ConverterOptions,
    Reindexer,
    SequentialReader,
    SequentialWriter,
    StorageOptions,
)
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs_py.point_cloud2
import yaml

from perception_dataset.utils.misc import unix_timestamp_to_nusc_timestamp


def get_options(
    bag_dir: str,
    storage_options: Optional[StorageOptions] = None,
    converter_options: Optional[ConverterOptions] = None,
) -> Tuple[StorageOptions, ConverterOptions]:
    storage_options = storage_options if storage_options else get_default_storage_options(bag_dir)
    converter_options = converter_options if converter_options else get_default_converter_options()
    return storage_options, converter_options


def create_reader(
    bag_dir: str,
    storage_options: Optional[StorageOptions] = None,
    converter_options: Optional[ConverterOptions] = None,
) -> SequentialReader:
    storage_options, converter_options = get_options(bag_dir, storage_options, converter_options)
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    return reader


def create_writer(bag_dir: str) -> SequentialWriter:
    storage_options = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    writer = SequentialWriter()
    writer.open(storage_options, converter_options)

    return writer


def reindex(bag_dir: str):
    storage_options = get_default_storage_options(bag_dir)
    Reindexer().reindex(storage_options)


def get_topic_type_dict(bag_dir: str) -> Dict[str, str]:
    reader = create_reader(bag_dir)

    topic_name_to_topic_type: Dict[str, str] = {}
    for topic in reader.get_all_topics_and_types():
        topic_name_to_topic_type[topic.name] = topic.type

    return topic_name_to_topic_type


def get_topic_count(bag_dir: str) -> Dict[str, int]:
    with open(osp.join(bag_dir, "metadata.yaml")) as f:
        bagfile_metadata = yaml.safe_load(f)["rosbag2_bagfile_information"]
    topic_name_to_topic_count: Dict[str, int] = {}
    for topic in bagfile_metadata["topics_with_message_count"]:
        topic_name_to_topic_count[topic["topic_metadata"]["name"]] = topic["message_count"]
    return topic_name_to_topic_count


def get_default_converter_options() -> ConverterOptions:
    return ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )


def infer_storage_id(bag_dir: str, storage_ids={".db3": "sqlite3", ".mcap": "mcap"}) -> str:
    bag_dir_path = Path(bag_dir)
    data_file = next(p for p in bag_dir_path.glob("*") if p.suffix in storage_ids)
    if data_file.suffix not in storage_ids:
        raise ValueError(f"Unsupported storage id: {data_file.suffix}")
    return storage_ids[data_file.suffix]


def get_default_storage_options(bag_dir: str) -> StorageOptions:
    storage_id = infer_storage_id(bag_dir)
    return StorageOptions(uri=bag_dir, storage_id=storage_id)


def pointcloud_msg_to_numpy(
    pointcloud_msg: PointCloud2,
) -> NDArray:
    """numpy ver. of https://github.com/ros2/common_interfaces/blob/master/sensor_msgs_py/sensor_msgs_py/point_cloud2.py#L119"""
    points_arr = np.array(
        [tuple(p) for p in sensor_msgs_py.point_cloud2.read_points(pointcloud_msg)],
        dtype=np.float32,
    )
    if len(points_arr[0]) > 5:
        points_arr = np.delete(points_arr, np.s_[5:], axis=1)
    while len(points_arr[0]) < 5:
        points_arr = np.insert(points_arr, len(points_arr[0]), -1, axis=1)
    return points_arr


def compressed_msg_to_numpy(compressed_image_msg: CompressedImage) -> NDArray:
    image_buf = np.ndarray(
        shape=(1, len(compressed_image_msg.data)), dtype=np.uint8, buffer=compressed_image_msg.data
    )
    image = cv2.imdecode(image_buf, cv2.IMREAD_ANYCOLOR)
    return image


def stamp_to_unix_timestamp(stamp: builtin_interfaces.msg.Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def unix_timestamp_to_stamp(timestamp: float) -> builtin_interfaces.msg.Time:
    sec_int = int(timestamp)
    nano_sec_int = (timestamp - sec_int) * 1e9
    return Time(seconds=sec_int, nanoseconds=nano_sec_int).to_msg()


def stamp_to_nusc_timestamp(stamp: builtin_interfaces.msg.Time) -> int:
    return unix_timestamp_to_nusc_timestamp(stamp_to_unix_timestamp(stamp))
