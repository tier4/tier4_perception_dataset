from typing import Any, Dict, List, Optional

from autoware_vehicle_msgs.msg import (
    GearReport,
    HazardLightsReport,
    SteeringReport,
    TurnIndicatorsReport,
    VelocityReport,
)
import builtin_interfaces.msg
from tier4_vehicle_msgs.msg import (
    ActuationStatusStamped,
    SteeringWheelStatusStamped,
)

from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.utils.rosbag2 import stamp_to_unix_timestamp


class VehicleStatusHandler:
    """A class to handle ROS2 messages related to vehicle status."""

    # Field name[str] -> Topic name[str]
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

    # GearReport[uint8] -> Shift[str]
    GEAR2SHIFT: Dict[int, str] = {
        0: "NONE",
        1: "NEUTRAL",
        2: "DRIVE",
        3: "DRIVE",
        4: "DRIVE",
        5: "DRIVE",
        6: "DRIVE",
        7: "DRIVE",
        8: "DRIVE",
        9: "DRIVE",
        10: "DRIVE",
        11: "DRIVE",
        12: "DRIVE",
        13: "DRIVE",
        14: "DRIVE",
        15: "DRIVE",
        16: "DRIVE",
        17: "DRIVE",
        18: "DRIVE",
        19: "DRIVE",
        20: "REVERSE",
        21: "REVERSE",
        22: "PARK",
        23: "LOW",
        24: "LOW",
    }

    # TurnIndicatorsReport[uint8] -> Dict[str, str]
    INDICATOR_MAPPING = {
        0: {"left": "off", "right": "off", "hazard": "off"},
        1: {"left": "on", "right": "off", "hazard": "off"},
        2: {"left": "off", "right": "on", "hazard": "off"},
        3: {"left": "off", "right": "off", "hazard": "on"},
    }

    def __init__(self, bag_dir: str, *, topic_mapping: Optional[Dict[str, str]] = None) -> None:
        """Construct a new object.

        Args:
            bag_dir (str): Directory path of rosbag.
            topic_mapping (Optional[Dict[str, str]], optional): Mapping of a field name to a topic name.
                If `None`, `DEFAULT_TOPIC_MAPPING` will be used. Defaults to None.
        """
        self._reader = Rosbag2Reader(bag_dir=bag_dir)
        self._topic_mapping = self.get_topic_mapping(topic_mapping=topic_mapping)

        buffer = {
            key: [msg for msg in self._reader.read_messages(topics=[topic])]  # noqa
            for key, topic in self._topic_mapping.items()
        }
        self._buffer = buffer

    @classmethod
    def get_topic_mapping(cls, topic_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Return a mapping of a field name to a topic name.
        If `topic_mapping` is not specified, `DEFAULT_TOPIC_MAPPING` will be returned.

        Args:
            topic_mapping (Optional[Dict[str, str]], optional): Custom mapping. Defaults to None.

        Returns:
            Dict[str, str]: Topic mapping.
        """
        if topic_mapping is not None:
            assert set(cls.DEFAULT_TOPIC_MAPPING) == set(topic_mapping.keys())
            return topic_mapping
        return cls.DEFAULT_TOPIC_MAPPING

    def gear_to_shift(self, gear: int) -> str:
        """Convert the value of the gear report to the shift state using `GEAR2SHIFT`.

        Args:
            gear (int): Value of `GearReport.report`.

        Returns:
            str: Shift state.
        """
        return self.GEAR2SHIFT[gear]

    def indicator_to_state(self, indicator: int) -> Dict[str, str]:
        """Convert the value of the indicator report to a mapping.

        Args:
            indicator (int): Value of `TurnIndicatorsReport.report`.

        Returns:
            Dict[str, str]: Mapping value for each indicator as following format:
                - keys: [left, right, hazard]
                - values: "on" or "off"
        """
        return self.INDICATOR_MAPPING[indicator]

    def get_closest_msg(self, key: str, stamp: builtin_interfaces.msg.Time) -> Any:
        assert key in self._buffer
        messages: List[Any] = self._buffer[key]
        timestamp = stamp_to_unix_timestamp(stamp)

        def time_diff(x) -> float:
            if hasattr(x, "header"):
                return abs(stamp_to_unix_timestamp(x.header.stamp) - timestamp)
            elif hasattr(x, "stamp"):
                return abs(stamp_to_unix_timestamp(x.stamp) - timestamp)
            else:
                raise ValueError("Unexpected message type")

        return min(messages, key=lambda x: time_diff(x))

    def get_actuation_statuses(self) -> List[ActuationStatusStamped]:
        """Return a list of `ActuationStatusStamped`s.

        Returns:
            List[ActuationStatusStamped]: List of messages.
        """
        return self._buffer["actuation_status"]

    def get_gear_reports(self) -> List[GearReport]:
        """Return a list of `GearReport`s.

        Returns:
            List[GearReport]: List of messages.
        """
        return self._buffer["gear_status"]

    def get_steering_reports(self) -> List[SteeringReport]:
        """Return a list of `SteeringReport`s.

        Returns:
            List[SteeringReport]: List of messages.
        """
        return self._buffer["steering_status"]

    def get_steering_wheel_statuses(self) -> List[SteeringWheelStatusStamped]:
        """Return a list of `SteeringWheelStatusStamped`s.

        Returns:
            List[SteeringWheelStatusStamped]: List of messages.
        """
        return self._buffer["steering_wheel_status"]

    def get_turn_indicators_reports(self) -> List[TurnIndicatorsReport]:
        """Return a list of `TurnIndicatorsReport`s.

        Returns:
            List[TurnIndicatorsReport]: List of messages.
        """
        return self._buffer["turn_indicators_status"]

    def get_hazard_lights_reports(self) -> List[HazardLightsReport]:
        """Return a list of `HazardLightsReport`s

        Returns:
            List[HazardLightsReport]: List of messages.
        """
        return self._buffer["hazard_lights_status"]

    def get_velocity_reports(self) -> List[VelocityReport]:
        """Return a list of `VelocityReport`s.

        Returns:
            List[VelocityReport]: List of messages.
        """
        return self._buffer["velocity_status"]
