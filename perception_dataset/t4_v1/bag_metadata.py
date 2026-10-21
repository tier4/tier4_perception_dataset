from __future__ import annotations

from collections import defaultdict
import json
from typing import Any
import uuid

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import StorageFilter
from rosidl_runtime_py.utilities import get_message
from t4_devkit.schema.v1 import Hardware

from perception_dataset.utils.rosbag2 import create_reader, get_topic_type_dict

__all__ = ["BagMetadata"]


class BagMetadata:
    """1.0 metadata read from the rosbags of a recording: the dataset's `input_bag/` and any
    extra bags (a filtered slice of the original log, the chunk where the route was set).

    Each method returns raw 1.0 values (no tokens); `notes` says what could not be used.
    """

    def __init__(self, bag_dirs: list[str]) -> None:
        self.notes: list[str] = []
        # topic -> [(stamp [us], message)], in stamp order
        self.messages: dict[str, list[tuple[int, Any]]] = defaultdict(list)
        # a message in several bags (input_bag and a slice of the original log) is read once
        self._seen: set[tuple[str, int, int]] = set()
        for bag_dir in bag_dirs:
            self._read(bag_dir)
        for messages in self.messages.values():
            messages.sort(key=lambda item: item[0])

    def _read(self, bag_dir: str) -> None:
        """Read the backfill topics and every `camera_info` from a bag."""
        backfill_topics = {
            "/localization/kinematic_state",
            "/sensing/imu/imu_data",
            "/sensing/gnss/septentrio/nav_sat_fix",
            "/vehicle/status/velocity_status",
            "/vehicle/status/steering_status",
            "/vehicle/status/gear_status",
            "/vehicle/status/turn_indicators_status",
            "/vehicle/status/hazard_lights_status",
            "/planning/mission_planning/route",
            "/api/routing/state",
            "/diagnostics",
            "/webauto/vehicle_info",
        }
        types = get_topic_type_dict(bag_dir)
        topics = [t for t in types if t in backfill_topics or t.endswith("/camera_info")]
        reader = create_reader(bag_dir)
        reader.set_filter(StorageFilter(topics=topics))
        classes = {}
        while reader.has_next():
            topic, data, bag_time = reader.read_next()
            key = (topic, bag_time, hash(data))
            if key in self._seen:
                continue
            self._seen.add(key)
            if topic not in classes:
                classes[topic] = get_message(types[topic])
            message = deserialize_message(data, classes[topic])
            if topic == "/planning/mission_planning/route":
                # a route reloaded by the mission planner keeps its old header stamp: use the
                # time it was published
                stamp = bag_time // 1000
            else:
                stamp = self._stamp(message, bag_time)
            self.messages[topic].append((stamp, message))

    @staticmethod
    def _stamp(message: Any, bag_time: int) -> int:
        """Return the message stamp [us]: `header.stamp`, `stamp`, else the bag time."""
        stamp = getattr(getattr(message, "header", None), "stamp", getattr(message, "stamp", None))
        if stamp is None or (stamp.sec == 0 and stamp.nanosec == 0):
            return bag_time // 1000
        return _us(stamp.sec, stamp.nanosec)

    def _nearest(self, topic: str, timestamps: list[int], tolerance: int) -> list[Any]:
        """Return the message of a topic nearest each timestamp [us], None if none within
        `tolerance` [us]."""
        messages = self.messages.get(topic, [])
        if not messages:
            return [None] * len(timestamps)
        stamps = np.array([stamp for stamp, _ in messages])
        found = []
        for timestamp in timestamps:
            after = int(np.searchsorted(stamps, timestamp))
            neighbors = [i for i in (after - 1, after) if 0 <= i < len(stamps)]
            i = min(neighbors, key=lambda i: abs(stamps[i] - timestamp))
            found.append(messages[i][1] if abs(stamps[i] - timestamp) <= tolerance else None)
        return found

    def ego_pose_fields(self, timestamps: list[int]) -> list[dict[str, Any]]:
        """Return `twist`, `covariance`, `acceleration` and `geocoordinate` at each ego pose time.

        - `twist`, `covariance`: `/localization/kinematic_state` (vehicle frame, map pose).
        - `acceleration`: `/sensing/imu/imu_data` in `base_link`, gravity included.
        - `geocoordinate`: `/sensing/gnss/septentrio/nav_sat_fix` with a fix.

        A value is None when no message is within 50 ms (GNSS: 200 ms).

        Args:
            timestamps (list[int]): Ego pose times [us].

        Returns:
            The fields of each ego pose, in order.
        """
        odometries = self._nearest("/localization/kinematic_state", timestamps, 50_000)
        imus = self._nearest("/sensing/imu/imu_data", timestamps, 50_000)
        fixes = self._nearest("/sensing/gnss/septentrio/nav_sat_fix", timestamps, 200_000)
        imu_frames = {imu.header.frame_id for imu in imus if imu is not None}
        if imu_frames - {"base_link"}:
            self.notes.append(f"acceleration left empty: imu_data is in {sorted(imu_frames)}")
            imus = [None] * len(timestamps)

        fields = []
        for odometry, imu, fix in zip(odometries, imus, fixes):
            row = {"twist": None, "covariance": None, "acceleration": None, "geocoordinate": None}
            if odometry is not None:
                linear, angular = odometry.twist.twist.linear, odometry.twist.twist.angular
                row["twist"] = [linear.x, linear.y, linear.z, angular.z, angular.y, angular.x]
                row["covariance"] = list(odometry.pose.covariance)
            if imu is not None:
                accel = imu.linear_acceleration
                row["acceleration"] = [accel.x, accel.y, accel.z]
            if fix is not None and fix.status.status >= 0:
                row["geocoordinate"] = [fix.latitude, fix.longitude, fix.altitude]
            fields.append(row)
        return fields

    def vehicle_states(self, timestamps: list[int]) -> list[dict[str, Any]]:
        """Return a 1.0 `vehicle_state` row (no token) at each sample time, from the nearest
        velocity, steering, gear, turn indicator and hazard report within 100 ms.

        Args:
            timestamps (list[int]): Sample times [us].

        Returns:
            The rows, in order; samples without a velocity report have none.
        """

        def shift_state(gear: int) -> str:
            # GearReport: NONE 0, NEUTRAL 1, DRIVE 2-19, REVERSE 20-21, PARK 22, LOW 23-24
            for state, last in (("NONE", 0), ("NEUTRAL", 1), ("DRIVE", 19), ("REVERSE", 21)):
                if gear <= last:
                    return state
            return {22: "PARK", 23: "LOW", 24: "LOW"}.get(gear, "unknown")

        def nearest(name: str) -> list[Any]:
            return self._nearest(f"/vehicle/status/{name}", timestamps, 100_000)

        velocities, steerings = nearest("velocity_status"), nearest("steering_status")
        gears = nearest("gear_status")
        turns, hazards = nearest("turn_indicators_status"), nearest("hazard_lights_status")
        rows = []
        for timestamp, velocity, steering, gear, turn, hazard in zip(
            timestamps, velocities, steerings, gears, turns, hazards
        ):
            if velocity is None:
                continue
            indicators = None
            if turn is not None and hazard is not None:
                # TurnIndicatorsReport: DISABLE 1, ENABLE_LEFT 2, ENABLE_RIGHT 3
                # HazardLightsReport: DISABLE 1, ENABLE 2
                indicators = {
                    "left": "on" if turn.report == 2 else "off",
                    "right": "on" if turn.report == 3 else "off",
                    "hazard": "on" if hazard.report == 2 else "off",
                }
            rows.append(
                {
                    "timestamp": timestamp,
                    "accel_pedal": None,
                    "brake_pedal": None,
                    "steer_pedal": None,
                    "steering_tire_angle": (
                        steering.steering_tire_angle if steering is not None else None
                    ),
                    "steering_wheel_angle": None,
                    "shift_state": shift_state(gear.report) if gear is not None else None,
                    "indicators": indicators,
                    "longitudinal_velocity": velocity.longitudinal_velocity,
                    "lateral_velocity": velocity.lateral_velocity,
                    "heading_rate": velocity.heading_rate,
                    "source": "vehicle_interface",
                }
            )
        return rows

    def routes(self) -> list[dict[str, Any]]:
        """Return a 1.0 `route` row (no token, log or map) for each route set.

        A route is valid from its stamp until the next route, or until `/api/routing/state`
        becomes UNSET (1) or ARRIVED (3); None if neither happens in the bags.
        """
        routes = self.messages.get("/planning/mission_planning/route", [])
        ends = [
            stamp
            for stamp, state in self.messages.get("/api/routing/state", [])
            if state.state in (1, 3)
        ]
        rows = []
        for i, (valid_from, route) in enumerate(routes):
            candidates = [stamp for stamp in ends if stamp > valid_from]
            if i + 1 < len(routes):
                candidates.append(routes[i + 1][0])
            rows.append(
                {
                    "valid_from": valid_from,
                    "valid_until": min(candidates, default=None),
                    "route_uuid": str(uuid.UUID(bytes=bytes(route.uuid.uuid))),
                    "start_pose": _pose(route.start_pose),
                    "goal_pose": _pose(route.goal_pose),
                    "segments": [
                        {
                            "preferred_primitive_id": segment.preferred_primitive.id,
                            "primitive_ids": [primitive.id for primitive in segment.primitives],
                        }
                        for segment in route.segments
                    ],
                    "allow_modification": route.allow_modification,
                    "source": "mission_planner",
                    "provenance": "own_bag",
                }
            )
        return rows

    def lidar_hardware(self) -> dict[str, Hardware]:
        """Return the hardware of each LiDAR, by position (e.g. `left_upper`).

        Read from the driver's `/diagnostics` status: name `/sensing/lidar/<position>/<node>: ...`,
        hardware id `<model>: <serial>` (e.g. `OT128: OT3BCC53943BCC56`).
        """
        makes = {"hesai": "Hesai"}
        hardware = {}
        for _, array in self.messages.get("/diagnostics", []):
            for status in array.status:
                parts = status.name.split("/")
                if (
                    len(parts) < 5
                    or parts[1:3] != ["sensing", "lidar"]
                    or ": " not in status.hardware_id
                ):
                    continue
                model, serial = status.hardware_id.split(": ", 1)
                make = next((m for key, m in makes.items() if key in parts[4]), None)
                hardware[parts[3]] = Hardware(
                    make=make, model=model, serial_number=serial, firmware_version=None
                )
        return hardware

    def distortion_models(self) -> dict[str, str]:
        """Return the `camera_info` distortion model of each camera, by name (e.g. `camera0`)."""
        return {
            topic.split("/")[-2]: messages[0][1].distortion_model
            for topic, messages in self.messages.items()
            if topic.endswith("/camera_info")
        }

    def vehicle_info(self) -> dict[str, Any] | None:
        """Return the first `/webauto/vehicle_info` (vehicle, map and firmware), or None."""
        messages = self.messages.get("/webauto/vehicle_info")
        return json.loads(messages[0][1].data) if messages else None


def _us(sec: int, nanosec: int) -> int:
    """Return a ROS time as Unix time [us]."""
    return sec * 1_000_000 + nanosec // 1000


def _pose(pose: Any) -> dict[str, list[float]]:
    """Return a ROS pose as a 1.0 `RoutePose`."""
    position, orientation = pose.position, pose.orientation
    return {
        "translation": [position.x, position.y, position.z],
        "rotation": [orientation.w, orientation.x, orientation.y, orientation.z],
    }
