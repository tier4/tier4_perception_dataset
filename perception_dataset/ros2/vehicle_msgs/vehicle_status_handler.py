from typing import Any, Dict, List, Optional

import builtin_interfaces.msg
from vehicle_msgs.msg import (
    ActuationStatusStamped,
    GearReport,
    HazardLightsReport,
    SteeringReport,
    SteeringWheelStatusStamped,
    TurnIndicatorsReport,
    VelocityReport,
)

from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.utils.rosbag2 import stamp_to_unix_timestamp


class VehicleStatusHandler:
    """A class to handle ROS2 messages related to vehicle status."""

    DEFAULT_TOPIC_MAPPING = {
        "actuation_status": "/vehicle/status/actuation_status",
        "control_mode": "/vehicle/status/control_mode",
        "door_status": "/vehicle/status/door_status",
        "gear_status": "/vehicle/status/gear_status",
        "hazard_lights_status": "/vehicle/status/hazard_lights_status",
        "steering_status": "/vehicle/status/steering_status",
        "steering_wheel_status": "/vehicle/status/steering_wheel_status",
        "turn_indicators_status": "/vehicle/status/turn_indicators_status",
        "velocity_status": "/vehicle/status/velocity_status",
    }

    def __init__(self, bag_dir: str, *, topic_mapping: Optional[Dict[str, str]] = None) -> None:
        self._reader = Rosbag2Reader(bag_dir=bag_dir)
        self._topic_mapping = self.get_topic_mapping(topic_mapping=topic_mapping)

        # TODO(ktro2828): performance update
        buffer = {
            key: [msg for msg in self._reader.read_messages(topics=[topic])]
            for key, topic in self._topic_mapping.items()
        }
        self._buffer = buffer

    @classmethod
    def get_topic_mapping(cls, topic_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        if topic_mapping is not None:
            assert set(cls.DEFAULT_TOPIC_MAPPING) == set(topic_mapping.keys())
            return topic_mapping
        return cls.DEFAULT_TOPIC_MAPPING

    def get_closest_msg(self, key: str, stamp: builtin_interfaces.msg.Time) -> Any:
        assert key in self._buffer
        messages: List[Any] = self._buffer[key]
        timestamp = stamp_to_unix_timestamp(stamp)
        return min(
            messages, key=lambda x: abs(stamp_to_unix_timestamp(x.header.stamp) - timestamp)
        )

    def get_actuation_statuses(self) -> List[ActuationStatusStamped]:
        return self._buffer["actuation_status"]

    def get_gear_reports(self) -> List[GearReport]:
        return self._buffer["gear_status"]

    def get_steering_reports(self) -> List[SteeringReport]:
        return self._buffer["steering_status"]

    def get_steering_wheel_statuses(self) -> List[SteeringWheelStatusStamped]:
        return self._buffer["steering_wheel_status"]

    def get_turn_indicators_reports(self) -> List[TurnIndicatorsReport]:
        return self._buffer["turn_indicators_status"]

    def get_hazard_lights_reports(self) -> List[HazardLightsReport]:
        return self._buffer["hazard_lights_status"]

    def get_velocity_reports(self) -> List[VelocityReport]:
        return self._buffer["velocity_status"]
