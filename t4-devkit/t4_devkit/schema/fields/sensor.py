from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import sys
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase

if sys.version_info < (3, 11):

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


__all__ = ("Sensor", "SensorModality", "SensorChannel")


class SensorModality(StrEnum):
    LIDAR = "lidar"
    CAMERA = "camera"
    RADAR = "radar"


class SensorChannel(StrEnum):
    CAM_BACK_LEFT = "CAM_BACK_LEFT"
    CAM_FRONT = "CAM_FRONT"
    CAM_FRONT_RIGHT = "CAM_FRONT_RIGHT"
    CAM_BACK_RIGHT = "CAM_BACK_RIGHT"
    CAM_BACK = "CAM_BACK"
    CAM_FRONT_LEFT = "CAM_FRONT_LEFT"
    CAM_TRAFFIC_LIGHT_NEAR = "CAM_TRAFFIC_LIGHT_NEAR"
    CAM_TRAFFIC_LIGHT_FAR = "CAM_TRAFFIC_LIGHT_FAR"
    LIDAR_TOP = "LIDAR_TOP"
    LIDAR_CONCAT = "LIDAR_CONCAT"
    RADAR_FRONT = "RADAR_FRONT"
    RADAR_FRONT_RIGHT = "RADAR_FRONT_RIGHT"
    RADAR_FRONT_LEFT = "RADAR_FRONT_LEFT"
    RADAR_BACK = "RADAR_BACK"
    RADAR_BACK_LEFT = "RADAR_BACK_LEFT"
    RADAR_BACK_RIGHT = "RADAR_BACK_RIGHT"

    @property
    def modality(self) -> SensorModality:
        if "CAM" in self:
            return SensorModality.CAMERA
        elif "LIDAR" in self:
            return SensorModality.LIDAR
        elif "RADAR" in self:
            return SensorModality.RADAR
        else:
            raise ValueError(f"Cannot find modality for {self.value}")


@dataclass(frozen=True)
class Sensor(SchemaBase):
    token: str
    channel: SensorChannel
    modality: SensorModality

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        token: str = record["token"]
        channel = SensorChannel(record["channel"])
        modality = SensorModality(record["modality"])
        return cls(token=token, channel=channel, modality=modality)
