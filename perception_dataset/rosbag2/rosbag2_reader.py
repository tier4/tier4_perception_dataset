"""from https://github.com/tier4/ros2bag_extensions/blob/main/ros2bag_extensions/ros2bag_extensions/verb/__init__.py"""

from typing import Any, Dict, List

import builtin_interfaces.msg
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import StorageFilter
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo
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

        self.sensor_topic_to_frame_id: Dict[str, str] = {
            topic: None
            for topic in self._topic_name_to_topic_type
            if "sensor_msgs/msg/" in self._topic_name_to_topic_type[topic]
        }
        self.camera_info: Dict[str, str] = {
            topic: None
            for topic in self._topic_name_to_topic_type
            if "sensor_msgs/msg/CameraInfo" in self._topic_name_to_topic_type[topic]
        }
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
            cam_info_available: bool = all(
                cam_info is not None for cam_info in self.camera_info.values()
            )
            frame_id_available: bool = all(
                frame_id is not None for frame_id in self.sensor_topic_to_frame_id.values()
            )
            if cam_info_available and frame_id_available:
                return
            self.camera_info[topic_name] = message

        import numpy as np

        for i in range(8):
            topic_name = f"/sensing/camera/camera{i}/camera_info"
            camera_info_msg = CameraInfo()
            camera_info_msg.header.frame_id = f"camera{i}/camera_optical_link"
            camera_info_msg.header.stamp = builtin_interfaces.msg.Time(
                sec=int(self.start_timestamp + 1), nanosec=0
            )
            camera_info_msg.distortion_model = "plumb_bob"
            camera_info_msg.r = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            camera_info_msg.width = 2880
            camera_info_msg.height = 1860
            if "camera0" in topic_name:
                camera_info_msg.k = np.array([5368.25873, 0.0, 1412.70938, 0.0, 5364.46693, 958.59729, 0.0, 0.0, 1.0])
                camera_info_msg.p = np.array([5305.15088, 0.0, 1412.64275, 0.0, 0.0, 5342.61084, 958.70113, 0.0, 0.0, 0.0, 1.0, 0.0])
                camera_info_msg.d = [-0.08849, -0.90255, 0.00075, 0.00111, 0.0]
            elif "camera1" in topic_name:
                camera_info_msg.k = np.array([1496.73395, 0.0, 1424.70018, 0.0, 1497.10726, 945.6712, 0.0, 0.0, 1.0])
                camera_info_msg.p = np.array([1015.1003418, 0.0, 1466.52248505, 0.0, 0.0, 1284.54455566, 950.87123341, 0.0, 0.0, 0.0, 1.0, 0.0])
                camera_info_msg.d = [-0.08989, -0.1186, -0.00016, -0.00007, 0.0067, 0.30995, -0.24648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif "camera2" in topic_name:
                camera_info_msg.k = np.array([1502.98471, 0.0, 1422.35349, 0.0, 1504.5042, 931.99575, 0.0, 0.0, 1.0])
                camera_info_msg.p = np.array([878.42378, 0.0, 1402.49031, 0.0, 0.0, 1258.01633, 933.10245, 0.0, 0.0, 0.0, 1.0, 0.0])
                camera_info_msg.d = [0.32864, -0.03183, 2e-05, 0.0002, 0.00267, 0.73261, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif "camera3" in topic_name:
                camera_info_msg.k = np.array([1500.05060, 0.00000, 1430.78876, 0.00000, 1499.95752, 940.95613, 0.00000, 0.00000, 1.00000])
                camera_info_msg.p = np.array([877.863525, 0.00000000, 1418.95998, 0.00000000, 0.0, 1254.34375, 945.262686, 0.00000000, 0.00000000, 0.0, 1.00000000, 0.0])
                camera_info_msg.d = [0.27430142, -0.02073177, 0.00007407, 0.00008116, 0.00128976, 0.67045534, 0.00000000, 0.00000000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif "camera4" in topic_name:
                camera_info_msg.k = np.array([5363.43736, 0.0, 1507.51202, 0.0, 5341.55785, 1076.26984, 0.0, 0.0, 1.0])
                camera_info_msg.p = np.array([5296.04052734, 0.0, 1511.62903545, 0.0, 0.0, 5311.76367188, 1077.67061308, 0.0, 0.0, 0.0, 1.0, 0.0])
                camera_info_msg.d = [-0.12858, -0.44056, 0.00123, 0.00215, 0.0]
            elif "camera5" in topic_name:
                camera_info_msg.k = np.array([1500.35853, 0.0, 1419.21658, 0.0, 1501.15968, 936.4509, 0.0, 0.0, 1.0])
                camera_info_msg.p = np.array([871.67853, 0.0, 1390.37965, 0.0, 0.0, 1251.62366, 939.62595, 0.0, 0.0, 0.0, 1.0, 0.0])
                camera_info_msg.d = [3.49528, 0.71004, 0.00028, 0.0001, -0.03621, 3.91361, 1.98308, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif "camera6" in topic_name:
                camera_info_msg.k = np.array([1543.88646, 0.0, 1455.51234, 0.0, 1542.117, 955.83491, 0.0, 0.0, 1.0])
                camera_info_msg.p = np.array([940.59991455, 0.0, 1472.20666395, 0.0, 0.0, 1302.85144043, 965.17800362, 0.0, 0.0, 0.0, 1.0, 0.0])
                camera_info_msg.d = [0.45661, -0.00186, -0.00003, -0.00015, 0.00153, 0.85654, 0.08203, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif "camera7" in topic_name:
                camera_info_msg.k = np.array([1493.89305, 0.0, 1434.05368, 0.0, 1494.11047, 938.13478, 0.0, 0.0, 1.0])
                camera_info_msg.p = np.array([870.17737, 0.0, 1421.48751, 0.0, 0.0, 1247.0332, 940.93758, 0.0, 0.0, 0.0, 1.0, 0.0])
                camera_info_msg.d = [0.45661, -0.00186, -0.00003, -0.00015, 0.00153, 0.85654, 0.08203, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            self.camera_info[topic_name] = camera_info_msg

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
