from __future__ import annotations

import json
from typing import Any, Dict, Optional

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class VehicleStateRecord(AbstractRecord):
    def __init__(
        self,
        timestamp: int,
        accel_pedal: Optional[float],
        brake_pedal: Optional[float],
        steer_pedal: Optional[float],
        steering_tire_angle: Optional[float],
        steering_wheel_angle: Optional[float],
        shift_state: Optional[str],
        indicators: Optional[Dict[str, str]],
        additional_info: Optional[Dict[str, Any]],
    ):
        super().__init__()

        if shift_state is not None:
            assert shift_state in (
                "PARK",
                "REVERSE",
                "NEUTRAL",
                "HIGH",
                "FORWARD",
                "LOW",
                "NONE",
            ), f"Got unexpected shift state: {shift_state}"
        if indicators is not None:
            assert {"left", "right", "hazard"} == set(indicators.keys())
            assert {"on", "off"} >= set(indicators.values())
        if additional_info is not None:
            assert {"speed"} == set(additional_info.keys())

        self.timestamp: int = timestamp
        self.accel_pedal: Optional[float] = accel_pedal
        self.brake_pedal: Optional[float] = brake_pedal
        self.steer_pedal: Optional[float] = steer_pedal
        self.steering_tire_angle: Optional[float] = steering_tire_angle
        self.steering_wheel_angle: Optional[float] = steering_wheel_angle
        self.shift_state: Optional[str] = shift_state
        self.indicators: Optional[Dict[str, str]] = indicators
        self.additional_info: Optional[Dict[str, Any]] = additional_info

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token": self.token,
            "timestamp": self.timestamp,
            "accel_pedal": self.accel_pedal,
            "brake_pedal": self.brake_pedal,
            "steer_pedal": self.steer_pedal,
            "steering_tire_angle": self.steering_tire_angle,
            "steering_wheel_angle": self.steering_wheel_angle,
            "shift_state": self.shift_state,
            "indicators": self.indicators,
            "additional_info": self.additional_info,
        }
        return d


class VehicleStateTable(AbstractTable[VehicleStateRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#vehicle_statejson"""

    FILENAME = "vehicle_state" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> VehicleStateRecord:
        return VehicleStateRecord(**kwargs)

    @classmethod
    def from_json(cls, filepath: str) -> VehicleStateTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls()
        for item in items:
            record = VehicleStateRecord(
                timestamp=item["timestamp"],
                accel_pedal=item.get("accel_pedal"),
                brake_pedal=item.get("brake_pedal"),
                steer_pedal=item.get("steer_pedal"),
                steering_tire_angle=item.get("steering_tire_angle"),
                steering_wheel_angle=item.get("steering_wheel_angle"),
                shift_state=item.get("shift_state"),
                indicators=item.get("indicators"),
                additional_info=items.get("additional_info"),
            )
            record.token = item["token"]
            table.set_record_to_table(record)

        return table
