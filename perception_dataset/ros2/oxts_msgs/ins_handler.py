from enum import Enum
from typing import Any, Dict, List, Optional
import warnings

import builtin_interfaces
import builtin_interfaces.msg
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from scipy.interpolate import interp1d
from sensor_msgs.msg import Imu, NavSatFix
import tf_transformations as tf

from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.utils.rosbag2 import stamp_to_unix_timestamp, unix_timestamp_to_stamp


class LocalizeMethod(Enum):
    WITH_ODOMETRY = 0


class INSHandler:
    DEFAULT_TOPIC_MAPPING = {
        "imu": "/ins/oxts/imu",
        "imu_bias": "/ins/oxts/imu_bias",
        "lever_arm": "/ins/oxts/lever_arm",
        "nav_sat_fix": "/ins/oxts/nav_sat_fix",
        "nav_sat_ref": "/ins/oxts/nav_sat_ref",
        "ncom": "/ins/oxts/ncom",
        "odometry": "/ins/oxts/odometry",
        "path": "/ins/oxts/path",
        "velocity": "/ins/oxts/velocity",
    }

    def __init__(
        self,
        bag_dir: str,
        *,
        localize_method: LocalizeMethod = LocalizeMethod.WITH_ODOMETRY,
        topic_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self._reader = Rosbag2Reader(bag_dir=bag_dir)
        self._localize_method = localize_method
        self._topic_mapping = self.get_topic_mapping(topic_mapping=topic_mapping)

        # TODO(ktro2828): performance update
        buffer = {
            key: [msg for msg in self._reader.read_messages(topics=[topic])]
            for key, topic in self._topic_mapping.items()
        }
        self._buffer = buffer

        if self._localize_method == LocalizeMethod.WITH_ODOMETRY:
            self._localize_with_odometry()
        else:
            raise ValueError(f"Unexpected localize method: {self._localize_method}")

    @classmethod
    def get_topic_mapping(cls, topic_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        if topic_mapping is not None:
            assert set(cls.DEFAULT_TOPIC_MAPPING) == set(topic_mapping.keys())
            return topic_mapping
        return cls.DEFAULT_TOPIC_MAPPING

    def _localize_with_odometry(self) -> None:
        odometries: List[Odometry] = self._buffer["odometry"]

        reference_pose = odometries[0].pose.pose
        reference_translation = [
            reference_pose.position.x,
            reference_pose.position.y,
            reference_pose.position.z,
        ]
        reference_rotation = [
            reference_pose.orientation.x,
            reference_pose.orientation.y,
            reference_pose.orientation.z,
            reference_pose.orientation.w,
        ]

        for odometry in odometries:
            assert odometry.header.frame_id == "map"

            current_pose = odometry.pose.pose
            current_translation = [
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z,
            ]

            current_rotation = [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ]

            ref_matrix = tf.compose_matrix(
                translate=reference_translation,
                angles=tf.euler_from_quaternion(reference_rotation),
            )

            cur_matrix = tf.compose_matrix(
                translate=current_translation,
                angles=tf.euler_from_quaternion(current_rotation),
            )

            relative_matrix = tf.concatenate_matrices(tf.inverse_matrix(ref_matrix), cur_matrix)
            relative_translation = tf.translation_from_matrix(relative_matrix)
            relative_rotation = tf.quaternion_from_matrix(relative_matrix)

            transform = TransformStamped()
            transform.header = odometry.header
            transform.child_frame_id = "base_link"
            transform.transform.translation.x = relative_translation[0]
            transform.transform.translation.y = relative_translation[1]
            transform.transform.translation.z = relative_translation[2]
            transform.transform.rotation.x = relative_rotation[0]
            transform.transform.rotation.y = relative_rotation[1]
            transform.transform.rotation.z = relative_rotation[2]
            transform.transform.rotation.w = relative_rotation[3]

            self._reader._tf_buffer.set_transform(transform, "default_authority")

    def get_closest_msg(self, key: str, stamp: builtin_interfaces.msg.Time) -> Any:
        assert key in self._buffer
        messages: List[Any] = self._buffer[key]
        timestamp = stamp_to_unix_timestamp(stamp)
        return min(
            messages, key=lambda x: abs(stamp_to_unix_timestamp(x.header.stamp) - timestamp)
        )

    def get_odometries(self) -> List[Odometry]:
        return self._buffer["odometry"]

    def get_imus(self) -> List[Imu]:
        return self._buffer["imu"]

    def get_nav_sat_fixes(self) -> List[NavSatFix]:
        return self._buffer["nav_sat_fix"]

    def get_interpolated_nav_sat_fixes(
        self, query_stamps: List[builtin_interfaces.msg.Time]
    ) -> List[NavSatFix]:
        """Interpolate NavSatFix using linear interpolation.

        Args:
            query_stamps (List[builtin_interfaces.msg.Time]): List of query stamps.

        Returns:
            List[NavSatFix]: List of interpolated messages.

        Warnings:
            If the value in `query_stamps` is out of range of the observed timestamps,
            the interpolation accuracy can decrease.
        """
        # convert Times to unix timestamps
        query_timestamps = [stamp_to_unix_timestamp(stamp) for stamp in query_stamps]

        observed = self.get_nav_sat_fixes()

        timestamps = []
        latitudes = []
        longitudes = []
        altitudes = []
        for msg in observed:
            timestamps.append(stamp_to_unix_timestamp(msg.header.stamp))
            latitudes.append(msg.latitude)
            longitudes.append(msg.longitude)
            altitudes.append(msg.altitude)

        if min(query_timestamps) < min(timestamps) or max(timestamps) < max(query_timestamps):
            warnings.warn(
                "The value in `query_timestamps` is out of range of the observed timestamps, "
                "it can decrease interpolation accuracy."
            )

        # fitting
        func_latitude = interp1d(timestamps, latitudes, fill_value="extrapolate")
        func_longitude = interp1d(timestamps, longitudes, fill_value="extrapolate")
        func_altitude = interp1d(timestamps, altitudes, fill_value="extrapolate")

        # interpolate
        interp_latitudes = func_latitude(query_timestamps)
        interp_longitudes = func_longitude(query_timestamps)
        interp_altitudes = func_altitude(query_timestamps)

        output: List[NavSatFix] = []
        for latitude, longitude, altitude, timestamp in zip(
            interp_latitudes,
            interp_longitudes,
            interp_altitudes,
            query_timestamps,
            strict=True,
        ):
            msg = NavSatFix()
            msg.header.frame_id = observed[0].header.frame_id
            msg.header.stamp = unix_timestamp_to_stamp(timestamp)
            msg.status = observed[0].status
            msg.position_covariance = observed[0].position_covariance
            msg.position_covariance_type = observed[0].position_covariance_type
            msg.latitude = latitude
            msg.longitude = longitude
            msg.altitude = altitude

            output.append(msg)

        return output

    def get_transform_stamped(
        self,
        target_frame: str,
        source_frame: str,
        stamp: builtin_interfaces.msg.Time,
    ) -> TransformStamped:
        return self._reader.get_transform_stamped(
            target_frame=target_frame,
            source_frame=source_frame,
            stamp=stamp,
        )
