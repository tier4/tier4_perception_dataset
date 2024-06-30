from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pyquaternion import Quaternion
from t4_devkit.common.io import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if TYPE_CHECKING:
    from t4_devkit.typing import CamDistortionType, CamIntrinsicType, RotationType, TranslationType

__all__ = ("CalibratedSensor",)


@dataclass
@SCHEMAS.register(SchemaName.CALIBRATED_SENSOR)
class CalibratedSensor(SchemaBase):
    """A dataclass to represent schema table of `calibrated_sensor.json`.

    Attributes:
    ----------
        token (str): Unique record identifier.
        sensor_token (str): Foreign key pointing to the sensor type.
        translation (TranslationType): Coordinates system origin given as [x, y, z] in [m].
        rotation (RotationType): Coordinates system orientation given as quaternion [w, x, y, z].
        camera_intrinsic (CamIntrinsicType): 3x3 camera intrinsic matrix. Empty for sensors that are not cameras.
        camera_distortion (CamDistortionType): Camera distortion array. Empty for sensors that are not cameras.
    """

    token: str
    sensor_token: str
    translation: TranslationType
    rotation: RotationType
    camera_intrinsic: CamIntrinsicType
    camera_distortion: CamDistortionType

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        objs: list[Self] = []
        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            sensor_token: str = record["sensor_token"]
            translation = np.array(record["translation"])
            rotation = Quaternion(record["rotation"])
            camera_intrinsic = np.array(record["camera_intrinsic"])
            camera_distortion = np.array(record["camera_distortion"])
            objs.append(
                cls(
                    token=token,
                    sensor_token=sensor_token,
                    translation=translation,
                    rotation=rotation,
                    camera_intrinsic=camera_intrinsic,
                    camera_distortion=camera_distortion,
                )
            )
        return objs
