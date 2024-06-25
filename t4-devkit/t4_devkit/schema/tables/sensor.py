from dataclasses import dataclass
from enum import Enum
import sys
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if sys.version_info < (3, 11):

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


__all__ = ("Sensor", "SensorModality", "SensorChannel")


class SensorModality(StrEnum):
    """An enum to represent sensor modalities."""

    LIDAR = "lidar"
    CAMERA = "camera"
    RADAR = "radar"


class SensorChannel(StrEnum):
    """An enum to represent sensor channels."""

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


@dataclass
@SCHEMAS.register(SchemaName.SENSOR)
class Sensor(SchemaBase):
    """A dataclass to represent schema table of `sensor.json`."""

    token: str
    channel: SensorChannel
    modality: SensorModality

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        objs: list[Self] = []
        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            channel = SensorChannel(record["channel"])
            modality = SensorModality(record["modality"])
            objs.append(cls(token=token, channel=channel, modality=modality))
        return objs
