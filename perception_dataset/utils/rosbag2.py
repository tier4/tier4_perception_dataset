"""some implementations are from https://github.com/tier4/ros2bag_extensions/blob/main/ros2bag_extensions/ros2bag_extensions/verb/__init__.py"""

import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import builtin_interfaces.msg
import cv2
from nptyping import NDArray
import numpy as np
from radar_msgs.msg import RadarTrack, RadarTracks
from rclpy.time import Time
from rosbag2_py import (
    ConverterOptions,
    Reindexer,
    SequentialReader,
    SequentialWriter,
    StorageOptions,
)
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
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


def point_cloud2_to_array(msg: PointCloud2) -> Dict[str, NDArray]:
    """
    Convert a sensor_msgs/PointCloud2 message to a NumPy array. The fields
    in the PointCloud2 message are mapped to the fields in the NumPy array
    as follows:
    * x, y, z -> xyz
    * intensity -> intensity
    * index -> lidar_index
    * other fields are ignored
    """
    # Get the index of the "intensity" fields in the PointCloud2 message
    field_names = [field.name for field in msg.fields]

    def get_field_data(pc_data, msg, field_name, dtype_map):
        if field_name in field_names:
            field_idx = field_names.index(field_name)
            offset = msg.fields[field_idx].offset
            datatype = msg.fields[field_idx].datatype
            dtype = dtype_map.get(datatype)
            if dtype is None:
                raise ValueError(f"Unsupported {field_name} datatype: {datatype}")
            return pc_data[:, offset : offset + np.dtype(dtype).itemsize].view(dtype=dtype)
        return None

    # Mapping for PointField datatypes
    dtype_map = {
        PointField.UINT8: np.uint8,
        PointField.UINT16: np.uint16,
        PointField.FLOAT32: np.float32,
    }

    # Convert PointCloud2 data to NumPy
    pc_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    xyz = pc_data[:, :12].view(dtype=np.float32).reshape(-1, 3)


    # Extract optional fields (intensity and index) and apply the same filter
    intensity = get_field_data(pc_data, msg, "intensity", dtype_map)
    lidar_index = get_field_data(pc_data, msg, "index", dtype_map)

    # Build result dictionary
    result = {"xyz": xyz}
    if intensity is not None:
        result["intensity"] = intensity
    if lidar_index is not None:
        result["lidar_index"] = lidar_index

    return result


def pointcloud_msg_to_numpy(pointcloud_msg: PointCloud2) -> NDArray:
    """Convert ROS PointCloud2 message to numpy array using ros2-numpy."""
    NUM_DIMENSIONS = 5

    if not isinstance(pointcloud_msg, PointCloud2):
        return np.zeros((0, NUM_DIMENSIONS), dtype=np.float32)

    # Convert the PointCloud2 message to a numpy structured array
    points = point_cloud2_to_array(pointcloud_msg)

    # Extract the x, y, z coordinates and additional fields if available
    points_arr = points["xyz"]
    if "intensity" in points.keys():
        intensity = points["intensity"].astype(np.float32)
        points_arr = np.hstack((points_arr, intensity))
    if "lidar_index" in points.keys():
        lidar_index = points["lidar_index"].astype(np.float32)
        points_arr = np.hstack((points_arr, lidar_index))

    # Ensure the resulting array has exactly NUM_DIMENSIONS columns
    if points_arr.shape[1] > NUM_DIMENSIONS:
        points_arr = points_arr[:, :NUM_DIMENSIONS]
    elif points_arr.shape[1] < NUM_DIMENSIONS:
        padding = np.full(
            (points_arr.shape[0], NUM_DIMENSIONS - points_arr.shape[1]), -1, dtype=np.float32
        )
        points_arr = np.hstack((points_arr, padding))

    return points_arr


def radar_tracks_msg_to_list(radar_tracks_msg: RadarTracks) -> List[Dict[str, Any]]:
    """Convert `RadarTracks` into list.
    Each element of list is dict as shown below.

    translation (Tuple[float, float, float]): x, y, z coordinates of the centroid of the object.
    velocity (Tuple[float, float, float]): The velocity of the object in each spatial dimension.
    acceleration (Tuple[float, float, float]): The acceleration of the object in each spatial dimension.
    size (Tuple[float, float, float]): The object size in the sensor frame.
    uuid (str): A unique ID of the object generated by the radar.
    classification (int): Object classification. NO_CLASSIFICATION=0, STATIC=1, DYNAMIC=2.
    """
    radar_tracks: List[Dict[str, Any]] = []
    for track in radar_tracks_msg.tracks:
        track: RadarTrack
        translation: Tuple[float, float, float] = (
            track.position.x,
            track.position.y,
            track.position.z,
        )
        velocity: Tuple[float, float, float] = (
            track.velocity.x,
            track.velocity.y,
            track.velocity.z,
        )
        acceleration: Tuple[float, float, float] = (
            track.acceleration.x,
            track.acceleration.y,
            track.acceleration.z,
        )
        size: Tuple[float, float, float] = (track.size.x, track.size.y, track.size.z)
        obj_uuid: str = str(uuid.UUID(bytes=track.uuid.uuid.tobytes()))

        radar_tracks.append(
            {
                "translation": translation,
                "velocity": velocity,
                "acceleration": acceleration,
                "size": size,
                "uuid": obj_uuid,
                "classification": track.classification,
            }
        )
    return radar_tracks


def compressed_msg_to_numpy(compressed_image_msg: CompressedImage) -> NDArray:
    if hasattr(compressed_image_msg, "_encoding"):
        try:
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            image = np.reshape(
                np_arr, (compressed_image_msg.height, compressed_image_msg.width, 3)
            )
        except Exception as e:
            print(e)
            return None
    else:
        image_buf = np.ndarray(
            shape=(1, len(compressed_image_msg.data)),
            dtype=np.uint8,
            buffer=compressed_image_msg.data,
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
