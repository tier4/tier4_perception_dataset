from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import sys
from typing import Any

from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if sys.version_info < (3, 11):

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


__all__ = ("Sensor", "SensorModality")


class SensorModality(StrEnum):
    """An enum to represent sensor modalities.

    Attributes:
        LIDAR: Lidar sensor.
        CAMERA: Camera sensor.
        RADAR: Radar sensor.
    """

    LIDAR = "lidar"
    CAMERA = "camera"
    RADAR = "radar"


@dataclass
@SCHEMAS.register(SchemaName.SENSOR)
class Sensor(SchemaBase):
    """A dataclass to represent schema table of `sensor.json`.

    Attributes:
        token (str): Unique record identifier.
        channel (str): Sensor channel name.
        modality (SensorModality): Sensor modality.

    Shortcuts:
    ---------
        first_sd_token (str): The first sample data token corresponding to its sensor channel.
    """

    token: str
    channel: str
    modality: SensorModality

    # shortcuts
    first_sd_token: str = field(init=False)

    @staticmethod
    def shortcuts() -> tuple[str] | None:
        return ("first_sd_token",)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        token: str = data["token"]
        channel = data["channel"]
        modality = SensorModality(data["modality"])

        return cls(token=token, channel=channel, modality=modality)
