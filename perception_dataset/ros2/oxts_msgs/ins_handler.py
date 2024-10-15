from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
import numpy as np
from scipy.interpolate import interp1d
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Header
import tf_transformations as tf

from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.utils.rosbag2 import stamp_to_unix_timestamp, unix_timestamp_to_stamp


@dataclass
class EgoState:
    header: Header
    child_frame_id: str
    translation: Vector3
    rotation: Quaternion
    twist: Twist
    accel: Vector3


class EgoStateBuffer:
    def __init__(self) -> None:
        self._buffer: list[EgoState] = []

    def set_state(self, ego_state: EgoState) -> None:
        self._buffer.append(ego_state)

    def lookup_state(self, stamp: RosTime) -> EgoState:
        return self.interpolate_state(stamp)

    def interpolate_state(
        self, query_stamp: Union[RosTime | Sequence[RosTime]]
    ) -> Union[EgoState | List[EgoState]]:
        """Return interpolated ego status(es).

        Args:
            query_stamp (Union[RosTime | Sequence[RosTime]]):
                Sequence of ROS timestamps or a single timestamp.

        Returns:
            Union[EgoState | List[EgoState]]: If input stamp is a sequential, returns a list of `EgoState`s.
                Otherwise returns a single `EgoState`.
        """
        if isinstance(query_stamp, RosTime):
            query_stamps = [query_stamp]
            is_single_query = True
        else:
            query_stamps = query_stamp
            is_single_query = False

        timestamps: List[float] = []
        translations: List[Tuple[float]] = []
        rotations = []
        twists = []
        accelerations: List[Tuple[float, float, float]] = []
        for state in self._buffer:
            pass

        return


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
        # buffer to store all messages
        buffer = {
            key: [msg for msg in self._reader.read_messages(topics=[topic])]
            for key, topic in self._topic_mapping.items()
        }
        self._buffer = buffer

        # buffer to store EgoStates
        self._ego_buffer = EgoStateBuffer()

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

        if len(odometries) == 0:
            return

        # reference_pose = odometries[0].pose.pose
        # reference_translation = [
        #     reference_pose.position.x,
        #     reference_pose.position.y,
        #     reference_pose.position.z,
        # ]
        # reference_rotation = [
        #     reference_pose.orientation.x,
        #     reference_pose.orientation.y,
        #     reference_pose.orientation.z,
        #     reference_pose.orientation.w,
        # ]

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

            # ref_matrix = tf.compose_matrix(
            #     translate=reference_translation,
            #     angles=tf.euler_from_quaternion(reference_rotation),
            # )

            cur_matrix = tf.compose_matrix(
                translate=current_translation,
                angles=tf.euler_from_quaternion(current_rotation),
            )

            # relative_matrix = tf.concatenate_matrices(tf.inverse_matrix(ref_matrix), cur_matrix)
            # relative_translation = tf.translation_from_matrix(relative_matrix)
            # relative_rotation = tf.quaternion_from_matrix(relative_matrix)

            imu: Imu = self.get_closest_msg(key="imu", stamp=odometry.header.stamp)

            # NOTE: rotate acceleration from IMU coords to map coords
            # b_H_m = relative_matrix  # base_link -> map
            b_H_m = cur_matrix  # base_link -> map
            i_H_b = tf.compose_matrix(
                angles=tf.euler_from_quaternion(
                    [
                        imu.orientation.x,
                        imu.orientation.y,
                        imu.orientation.z,
                        imu.orientation.w,
                    ]
                )
            )  # imu -> base_link

            i_H_m = tf.concatenate_matrices(i_H_b, b_H_m)  # imu -> map
            # rotate acceleration
            relative_acceleration = np.matmul(
                i_H_m[:3, :3],
                [
                    imu.linear_acceleration.x,
                    imu.linear_acceleration.y,
                    imu.linear_acceleration.z,
                ],
            )

            # ego_state = EgoState(
            #     header=odometry.header,
            #     child_frame_id="base_link",
            #     translation=Vector3(
            #         x=relative_translation[0],
            #         y=relative_translation[1],
            #         z=relative_translation[2],
            #     ),
            #     rotation=Quaternion(
            #         x=relative_rotation[0],
            #         y=relative_rotation[1],
            #         z=relative_rotation[2],
            #         w=relative_rotation[3],
            #     ),
            #     twist=odometry.twist.twist,
            #     accel=Vector3(
            #         x=relative_acceleration[0],
            #         y=relative_acceleration[1],
            #         z=relative_acceleration[2],
            #     ),
            # )
            ego_state = EgoState(
                header=odometry.header,
                child_frame_id="base_link",
                translation=Vector3(
                    x=current_translation[0],
                    y=current_translation[1],
                    z=current_translation[2],
                ),
                rotation=Quaternion(
                    x=current_rotation[0],
                    y=current_rotation[1],
                    z=current_rotation[2],
                    w=current_rotation[3],
                ),
                twist=odometry.twist.twist,
                accel=Vector3(
                    x=relative_acceleration[0],
                    y=relative_acceleration[1],
                    z=relative_acceleration[2],
                ),
            )

            self._ego_buffer.set_state(ego_state)

    def get_closest_msg(self, key: str, stamp: RosTime) -> Any:
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

    def get_interpolated_nav_sat_fixes(self, query_stamps: List[RosTime]) -> List[NavSatFix]:
        """Interpolate NavSatFix using linear interpolation.

        Args:
            query_stamps (List[RosTime]): List of query stamps.

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

    def get_ego_state(self, stamp: RosTime) -> EgoState:
        """Return a `EgoState` interpolated by the specified `stamp`.

        Args:
            stamp (RosTime): ROS timestamp.

        Returns:
            EgoState: Corresponding ego state.
        """
        return self._ego_buffer.lookup_state(stamp)
