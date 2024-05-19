"""from https://github.com/tier4/ros2bag_extensions/blob/main/ros2bag_extensions/ros2bag_extensions/verb/__init__.py"""

from typing import Any, Dict, List

import builtin_interfaces.msg
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import StorageFilter
from rosidl_runtime_py.utilities import get_message
import tf2_ros

from perception_dataset.utils.rosbag2 import create_reader, get_topic_count, get_topic_type_dict


class Rosbag2Reader:
    def __init__(self, bag_dir: str, with_world_frame_conversion: bool = False):
        self._bag_dir: str = bag_dir

        self._topic_name_to_topic_type = get_topic_type_dict(self._bag_dir)
        self._topic_name_to_topic_count = get_topic_count(self._bag_dir)
        self._is_tf_needed = with_world_frame_conversion

        #  start time in seconds
        self.start_timestamp = self._get_starting_time()
        # set the duration long enough for handling merged bag files
        self._tf_buffer = tf2_ros.BufferCore(Duration(seconds=1e9))
        self._set_tf_buffer()
        self.sensor_topic_to_frame_id: Dict[str, str] = {}
        self.camera_info: Dict[str, str] = {}
        self._set_camera_info()

    def _get_starting_time(self) -> float:
        reader = create_reader(self._bag_dir)
        first_timestamp = 0
        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            topic_type = self._topic_name_to_topic_type[topic_name]

            # fail to deserialize Marker messages
            # https://docs.ros.org/en/rolling/Releases/Release-Humble-Hawksbill.html#support-textures-and-embedded-meshes-for-marker-messages
            if topic_type.startswith("visualization_msgs"):
                continue

            try:
                msg_type = get_message(topic_type)
            except ModuleNotFoundError:
                continue
            except AttributeError:
                print("Sourced message type is differ from the one in rosbag")
                continue

            msg = deserialize_message(data, msg_type)

            # get timestamp from header.stamp
            if hasattr(msg, "header"):
                msg_stamp = msg.header.stamp
            else:
                continue
            # this might happen for some topics like "/map/vector_map", "/diagnostics_agg", ""
            if msg_stamp.sec == 0 and msg_stamp.nanosec == 0:
                continue
            msg_stamp = msg_stamp.sec + msg_stamp.nanosec / 1e9
            if first_timestamp == 0:
                first_timestamp = msg_stamp

            if abs(timestamp / 1e9 - msg_stamp) > 24 * 60 * 60:
                continue

            return msg_stamp
        return first_timestamp

    def _set_tf_buffer(self):
        """set /tf and /tf_static to tf_buffer"""
        if self._is_tf_needed and "/tf" not in self._topic_name_to_topic_type:
            raise ValueError(f"/tf is not in {self._bag_dir}")
        if "/tf_static" not in self._topic_name_to_topic_type:
            raise ValueError(f"/tf_static is not in {self._bag_dir}")
        for message in self.read_messages(topics=["/tf"]):
            for transform in message.transforms:
                self._tf_buffer.set_transform(transform, "default_authority")

        for message in self.read_messages(topics=["/tf_static"]):
            for transform in message.transforms:
                self._tf_buffer.set_transform_static(transform, "default_authority")

    def _set_camera_info(self):
        """set /camera_info to self.camera_info"""
        for topic_name, message in self.read_camera_info():
            self.camera_info[topic_name] = message

    def get_topic_count(self, topic_name: str) -> int:
        return self._topic_name_to_topic_count.get(topic_name, 0)

    def read_camera_info(self) -> Any:
        reader = create_reader(self._bag_dir)
        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            topic_type = self._topic_name_to_topic_type[topic_name]

            if "sensor_msgs/msg/" in topic_type:
                msg_type = get_message(topic_type)
                msg = deserialize_message(data, msg_type)
                if hasattr(msg, "header"):
                    msg_frame_id = msg.header.frame_id
                    self.sensor_topic_to_frame_id[topic_name] = msg_frame_id
                else:
                    continue

            if topic_type != "sensor_msgs/msg/CameraInfo":
                continue

            msg_type = get_message(topic_type)
            msg = deserialize_message(data, msg_type)
            yield topic_name, msg

    def read_messages(
        self, topics: List[str], start_time: builtin_interfaces.msg.Time = None
    ) -> Any:
        if start_time is not None:
            start_time = Time.from_msg(start_time)

        reader = create_reader(self._bag_dir)
        if len(topics) != 0:
            reader.set_filter(StorageFilter(topics=topics))

        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            topic_type = self._topic_name_to_topic_type[topic_name]

            # fails to deserialize Marker messages
            # https://docs.ros.org/en/rolling/Releases/Release-Humble-Hawksbill.html#support-textures-and-embedded-meshes-for-marker-messages
            if topic_type.startswith("visualization_msgs"):
                continue

            message = deserialize_message(data, get_message(topic_type))

            if start_time is not None:
                # FIXME(yukke42): if message is tf, message value is list
                if hasattr(message, "header"):
                    message_time = Time.from_msg(message.header.stamp)
                elif hasattr(message, "stamp"):
                    message_time = Time.from_msg(message.stamp)
                else:
                    raise AttributeError()

                if message_time < start_time:
                    continue

            yield message

    def get_transform_stamped(
        self,
        target_frame: str,
        source_frame: str,
        stamp: builtin_interfaces.msg.Time,
    ) -> TransformStamped:
        return self._tf_buffer.lookup_transform_core(target_frame, source_frame, stamp)
