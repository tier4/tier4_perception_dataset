from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import Header

from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.utils.rosbag2 import stamp_to_unix_timestamp, unix_timestamp_to_stamp


@dataclass
class EgoState:
    """A dataclass to represent a set of ego status.

    Attributes:
        header (Header): Message header.
        translation (Vector3): 3D position in global coordinates.
        rotation (Quaternion): Quaternion in global coordinates.
        twist (Twist): Twist in local coordinates.
            Linear velocities are in (m/s), and angular velocities in (rad/s).
        accel (Vector3): Acceleration in local coordinates in (m/s2).
    """

    header: Header
    translation: Vector3
    rotation: Quaternion
    twist: Twist
    accel: Vector3


class EgoStateBuffer:
    """A buffer class to store `EgoState`s."""

    def __init__(self) -> None:
        self._buffer: list[EgoState] = []

        # Cache of interpolation functions
        self._interp_functions: Optional[Dict[str, Callable]] = None

    def set_state(self, ego_state: EgoState) -> None:
        """Set a ego state to the buffer.

        Args:
            ego_state (EgoState): Ego state.
        """
        self._buffer.append(ego_state)

    def _load_interpolate_functions(self) -> Dict[str, Callable]:
        """Load interpolate functions for each ego state field.

        Returns:
            Dict[str, Callable]: Key-value mapping of interpolations for each field.
        """
        timestamps: List[float] = []
        translations: List[Tuple[float, ...]] = []
        rotations: List[Tuple[float, ...]] = []
        linear_vel: List[Tuple[float, ...]] = []
        angular_vel: List[Tuple[float, ...]] = []
        accelerations: List[Tuple[float, ...]] = []
        for state in self._buffer:
            timestamps.append(stamp_to_unix_timestamp(state.header.stamp))
            translations.append((state.translation.x, state.translation.y, state.translation.z))
            rotations.append(
                (
                    state.rotation.w,
                    state.rotation.x,
                    state.rotation.y,
                    state.rotation.z,
                )
            )
            linear_vel.append(
                (
                    state.twist.linear.x,
                    state.twist.linear.y,
                    state.twist.linear.z,
                )
            )
            angular_vel.append(
                (
                    state.twist.angular.x,
                    state.twist.angular.y,
                    state.twist.angular.z,
                )
            )

            accelerations.append(
                (
                    state.accel.x,
                    state.accel.y,
                    state.accel.z,
                )
            )

        translations = np.asarray(translations).reshape(-1, 3)
        linear_vel = np.asarray(linear_vel).reshape(-1, 3)
        angular_vel = np.asarray(angular_vel).reshape(-1, 3)
        accelerations = np.asanyarray(accelerations).reshape(-1, 3)

        interp_func_x = interp1d(
            timestamps,
            translations[:, 0],
            fill_value="extrapolate",
        )
        interp_func_y = interp1d(
            timestamps,
            translations[:, 1],
            fill_value="extrapolate",
        )
        interp_func_z = interp1d(
            timestamps,
            translations[:, 2],
            fill_value="extrapolate",
        )

        interp_func_rot = Slerp(timestamps, Rotation.from_quat(rotations))

        interp_func_lvx = interp1d(
            timestamps,
            linear_vel[:, 0],
            fill_value="extrapolate",
        )
        interp_func_lvy = interp1d(
            timestamps,
            linear_vel[:, 1],
            fill_value="extrapolate",
        )
        interp_func_lvz = interp1d(
            timestamps,
            linear_vel[:, 2],
            fill_value="extrapolate",
        )

        interp_func_avx = interp1d(
            timestamps,
            angular_vel[:, 0],
            fill_value="extrapolate",
        )
        interp_func_avy = interp1d(
            timestamps,
            angular_vel[:, 1],
            fill_value="extrapolate",
        )
        interp_func_avz = interp1d(
            timestamps,
            angular_vel[:, 2],
            fill_value="extrapolate",
        )

        interp_func_ax = interp1d(
            timestamps,
            accelerations[:, 0],
            fill_value="extrapolate",
        )
        interp_func_ay = interp1d(
            timestamps,
            accelerations[:, 1],
            fill_value="extrapolate",
        )
        interp_func_az = interp1d(
            timestamps,
            accelerations[:, 2],
            fill_value="extrapolate",
        )

        return {
            "x": interp_func_x,
            "y": interp_func_y,
            "z": interp_func_z,
            "rot": interp_func_rot,
            "linear_vx": interp_func_lvx,
            "linear_vy": interp_func_lvy,
            "linear_vz": interp_func_lvz,
            "angular_vx": interp_func_avx,
            "angular_vy": interp_func_avy,
            "angular_vz": interp_func_avz,
            "ax": interp_func_ax,
            "ay": interp_func_ay,
            "az": interp_func_az,
        }

    def lookup_state(self, stamp: RosTime) -> EgoState:
        """Lookup a ego state corresponding to ROS timestamp.

        Args:
            stamp (RosTime): _description_

        Returns:
            EgoState: _description_
        """
        return self.interpolate_state(stamp)

    def interpolate_state(
        self,
        query_stamp: Union[RosTime | Sequence[RosTime]],
    ) -> Union[EgoState | List[EgoState]]:
        """Return interpolated ego status(es).

        Args:
            query_stamp (Union[RosTime | Sequence[RosTime]]):
                Sequence of ROS timestamps or a single timestamp.

        Returns:
            Union[EgoState | List[EgoState]]: If input stamp is a sequential, returns a list of `EgoState`s.
                Otherwise returns a single `EgoState`.
        """
        if self._interp_functions is None:
            self._interp_functions = self._load_interpolate_functions()

        if isinstance(query_stamp, RosTime):
            query_timestamps = [stamp_to_unix_timestamp(query_stamp)]
            is_single_query = True
        else:
            query_timestamps = [stamp_to_unix_timestamp(stamp) for stamp in query_stamp]
            is_single_query = False

        interp_x = self._interp_functions["x"](query_timestamps)
        interp_y = self._interp_functions["y"](query_timestamps)
        interp_z = self._interp_functions["z"](query_timestamps)

        interp_rotations = self._interp_functions["rot"](query_timestamps).as_quat().reshape(-1, 4)

        interp_lvx = self._interp_functions["linear_vx"](query_timestamps)
        interp_lvy = self._interp_functions["linear_vy"](query_timestamps)
        interp_lvz = self._interp_functions["linear_vz"](query_timestamps)

        interp_avx = self._interp_functions["angular_vx"](query_timestamps)
        interp_avy = self._interp_functions["angular_vy"](query_timestamps)
        interp_avz = self._interp_functions["angular_vz"](query_timestamps)

        interp_ax = self._interp_functions["ax"](query_timestamps)
        interp_ay = self._interp_functions["ay"](query_timestamps)
        interp_az = self._interp_functions["az"](query_timestamps)

        interp_translations = np.stack([interp_x, interp_y, interp_z], axis=-1)
        interp_twists = np.stack(
            [interp_lvx, interp_lvy, interp_lvz, interp_avx, interp_avy, interp_avz],
            axis=-1,
        )
        interp_accelerations = np.stack([interp_ax, interp_ay, interp_az], axis=-1)

        output: List[EgoState] = []
        for (
            translation,
            rotation,
            twist,
            accel,
            timestamp,
        ) in zip(
            interp_translations,
            interp_rotations,
            interp_twists,
            interp_accelerations,
            query_timestamps,
            strict=True,
        ):
            header = Header()
            header.frame_id = "map"
            header.stamp = unix_timestamp_to_stamp(timestamp)

            twist_msg = Twist()
            twist_msg.linear.x = twist[0]
            twist_msg.linear.y = twist[1]
            twist_msg.linear.z = twist[2]
            twist_msg.angular.x = twist[3]
            twist_msg.angular.y = twist[4]
            twist_msg.angular.z = twist[5]

            ego_state = EgoState(
                header=header,
                translation=Vector3(
                    x=translation[0],
                    y=translation[1],
                    z=translation[2],
                ),
                rotation=Quaternion(
                    w=rotation[0],
                    x=rotation[1],
                    y=rotation[2],
                    z=rotation[3],
                ),
                twist=twist_msg,
                accel=Vector3(
                    x=accel[0],
                    y=accel[1],
                    z=accel[2],
                ),
            )

            output.append(ego_state)

        return output[0] if is_single_query else output


class LocalizeMethod(Enum):
    WITH_ODOMETRY = 0


class INSHandler:
    DEFAULT_TOPIC_MAPPING = {
        "imu": "/ins/oxts/imu",
        "nav_sat_fix": "/ins/oxts/nav_sat_fix",
        "odometry": "/ins/oxts/odometry",
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

        # buffer to store all messages
        buffer = {
            key: [msg for msg in self._reader.read_messages(topics=[topic])]  # noqa
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
            current_twist = odometry.twist.twist

            # acceleration from imu
            # TODO: update with imu bias
            imu: Imu = self.get_closest_msg(key="imu", stamp=odometry.header.stamp)
            current_acceleration = [
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            ]

            if True:
                # TODO: tentative correction. If the INS coordinates are fixed, remove this correction.
                r_current = Rotation.from_quat(current_rotation)
                roll, pitch, yaw = r_current.as_euler("xyz")
                yaw = yaw - np.pi / 2
                roll = roll + np.pi
                r_new = (
                    Rotation.from_euler("z", yaw)
                    * Rotation.from_euler("y", pitch)
                    * Rotation.from_euler("x", roll)
                )
                current_rotation = r_new.as_quat()
                current_twist.angular.z = -current_twist.angular.z
                current_acceleration[1] = -current_acceleration[1]

            ego_state = EgoState(
                header=odometry.header,
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
                twist=current_twist,
                accel=Vector3(
                    x=current_acceleration[0],
                    y=current_acceleration[1],
                    z=current_acceleration[2],
                ),
            )

            self._ego_buffer.set_state(ego_state)

    def get_closest_msg(self, key: str, stamp: RosTime) -> Any:
        assert key in self._buffer
        messages: List[Any] = self._buffer[key]
        timestamp = stamp_to_unix_timestamp(stamp)
        return min(
            messages,
            key=lambda x: abs(stamp_to_unix_timestamp(x.header.stamp) - timestamp),
        )

    def get_odometries(self) -> List[Odometry]:
        return self._buffer["odometry"]

    def get_imus(self) -> List[Imu]:
        return self._buffer["imu"]

    def lookup_nav_sat_fixes(self, stamp: RosTime) -> Optional[NavSatFix]:
        return self.interpolate_nav_sat_fixes(stamp)

    def interpolate_nav_sat_fixes(
        self, query_stamp: Union[RosTime, List[RosTime]]
    ) -> Optional[Union[NavSatFix | List[NavSatFix]]]:
        """Interpolate NavSatFix.

        Args:
            query_stamp (RosTime | List[RosTime]): Query stamp(s).

        Returns:
            Optional[Union[NavSatFix, List[NavSatFix]]]: Interpolated message(s).
                Note that it returns `None`, if there is no observed NavSatFix messages.

        Warnings:
            If the value in `query_stamps` is out of range of the observed timestamps,
            the interpolation accuracy can decrease.
        """
        # convert Times to unix timestamps
        if isinstance(query_stamp, RosTime):
            query_stamps = [query_stamp]
            is_single_query = True
        else:
            query_stamps = query_stamp
            is_single_query = False

        query_timestamps = [stamp_to_unix_timestamp(stamp) for stamp in query_stamps]

        observed = self._buffer["nav_sat_fix"]

        timestamps = []
        geo_coordinates: List[Tuple[float, ...]] = []
        for msg in observed:
            if msg.status.status == NavSatStatus.STATUS_NO_FIX:
                continue
            timestamps.append(stamp_to_unix_timestamp(msg.header.stamp))
            geo_coordinates.append((msg.latitude, msg.longitude, msg.altitude))

        if len(timestamps) == 0:
            return None

        if min(query_timestamps) < min(timestamps) or max(timestamps) < max(query_timestamps):
            warnings.warn(
                "The value in `query_timestamps` is out of range of the observed timestamps, "
                "it can decrease interpolation accuracy."
            )

        geo_coordinates = np.asarray(geo_coordinates).reshape(-1, 3)
        interp_func_lat = interp1d(timestamps, geo_coordinates[:, 0], fill_value="extrapolate")
        interp_func_lon = interp1d(timestamps, geo_coordinates[:, 1], fill_value="extrapolate")
        interp_func_alt = interp1d(timestamps, geo_coordinates[:, 2], fill_value="extrapolate")

        # interpolate
        interp_lat = interp_func_lat(query_timestamps)
        interp_lon = interp_func_lon(query_timestamps)
        interp_alt = interp_func_alt(query_timestamps)
        interp_geo_coords = np.stack([interp_lat, interp_lon, interp_alt], axis=-1)

        output: List[NavSatFix] = []
        for coord, timestamp in zip(interp_geo_coords, query_timestamps, strict=True):
            msg = NavSatFix()
            msg.header.frame_id = observed[0].header.frame_id
            msg.header.stamp = unix_timestamp_to_stamp(timestamp)
            msg.status = observed[0].status
            msg.position_covariance = observed[0].position_covariance
            msg.position_covariance_type = observed[0].position_covariance_type
            msg.latitude = coord[0]
            msg.longitude = coord[1]
            msg.altitude = coord[2]

            output.append(msg)

        return output[0] if is_single_query else output

    def get_ego_state(self, stamp: RosTime) -> EgoState:
        """Return a `EgoState` interpolated by the specified `stamp`.

        Args:
            stamp (RosTime): ROS timestamp.

        Returns:
            EgoState: Corresponding ego state.
        """
        return self._ego_buffer.lookup_state(stamp)
