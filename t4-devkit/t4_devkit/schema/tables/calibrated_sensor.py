from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from pyquaternion import Quaternion
from t4_devkit.common.io import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if TYPE_CHECKING:
    from t4_devkit.typing import (
        AccelerationType,
        CamDistortionType,
        CamIntrinsicType,
        RotationType,
        TranslationType,
        VelocityType,
    )

__all__ = ("CalibratedSensor",)


@dataclass
@SCHEMAS.register(SchemaName.CALIBRATED_SENSOR)
class CalibratedSensor(SchemaBase):
    """A dataclass to represent schema table of `calibrated_sensor.json`."""

    token: str
    sensor_token: str
    translation: TranslationType
    rotation: RotationType
    camera_intrinsic: CamIntrinsicType
    camera_distortion: CamDistortionType
    velocity: VelocityType | None = field(default=None)
    acceleration: AccelerationType | None = field(default=None)

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        objs: list[Self] = []
        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            sensor_token: str = record["sensor_token"]
            translation = np.array(record["translation"])
            velocity = np.array(record["velocity"]) if record.get("velocity") else None
            acceleration = np.array(record["acceleration"]) if record.get("velocity") else None
            rotation = Quaternion(record["rotation"])
            camera_intrinsic = np.array(record["camera_intrinsic"])
            camera_distortion = np.array(record["camera_distortion"])
            objs.append(
                cls(
                    token=token,
                    sensor_token=sensor_token,
                    translation=translation,
                    velocity=velocity,
                    acceleration=acceleration,
                    rotation=rotation,
                    camera_intrinsic=camera_intrinsic,
                    camera_distortion=camera_distortion,
                )
            )
        return objs
