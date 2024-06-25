from dataclasses import dataclass
from typing import Any

import numpy as np
from pyquaternion import Quaternion
from t4_devkit.common import load_json
from t4_devkit.typing import (
    AccelerationType,
    RotationType,
    SizeType,
    TranslationType,
    VelocityType,
)
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("SampleAnnotation",)


@dataclass
class SampleAnnotation(SchemaBase):
    token: str
    sample_token: str
    instance_token: str
    attribute_tokens: list[str]
    visibility_token: str
    translation: TranslationType
    velocity: VelocityType
    acceleration: AccelerationType
    size: SizeType
    rotation: RotationType
    num_lidar_pts: int
    num_radar_pts: int
    next: str  # noqa: A003
    prev: str

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        token: str = record["token"]
        instance_token: str = record["instance_token"]
        attribute_tokens: list[str] = record["attribute_tokens"]
        visibility_token: str = record["visibility_token"]
        translation = np.array(record["translation"])
        velocity = np.array(record["velocity"]) if record.get("velocity") else None
        acceleration = np.array(record["acceleration"]) if record.get("acceleration") else None
        size = np.array(record["size"])
        rotation = Quaternion(record["rotation"])
        num_lidar_pts: int = record["num_lidar_pts"]
        num_radar_pts: int = record["num_radar_pts"]
        next_: str = record["next"]
        prev: str = record["prev"]
        return cls(
            token=token,
            instance_token=instance_token,
            attribute_tokens=attribute_tokens,
            visibility_token=visibility_token,
            translation=translation,
            velocity=velocity,
            acceleration=acceleration,
            size=size,
            rotation=rotation,
            num_lidar_pts=num_lidar_pts,
            num_radar_pts=num_radar_pts,
            next=next_,
            prev=prev,
        )
