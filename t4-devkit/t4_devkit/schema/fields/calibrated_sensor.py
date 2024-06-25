from dataclasses import dataclass
from typing import Any

import numpy as np
from pyquaternion import Quaternion
from t4_devkit.common import load_json
from t4_devkit.typing import (
    AccelerationType,
    CamDistortionType,
    CamIntrinsicType,
    RotationType,
    TranslationType,
    VelocityType,
)
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("CalibratedSensor",)


@dataclass(frozen=True)
class CalibratedSensor(SchemaBase):
    token: str
    sensor_token: str
    translation: TranslationType
    velocity: VelocityType
    acceleration: AccelerationType
    rotation: RotationType
    camera_intrinsic: CamIntrinsicType
    camera_distortion: CamDistortionType

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        token: str = record["token"]
        sensor_token: str = record["sensor_token"]
        translation = np.array(record["translation"])
        velocity = np.array(record["velocity"]) if record.get("velocity") else None
        acceleration = np.array(record["acceleration"]) if record.get("velocity") else None
        rotation = Quaternion(record["rotation"])
        camera_intrinsic = np.array(record["camera_intrinsic"])
        camera_distortion = np.array(record["camera_distortion"])
        return cls(
            token=token,
            sensor_token=sensor_token,
            translation=translation,
            velocity=velocity,
            acceleration=acceleration,
            rotation=rotation,
            camera_intrinsic=camera_intrinsic,
            camera_distortion=camera_distortion,
        )
