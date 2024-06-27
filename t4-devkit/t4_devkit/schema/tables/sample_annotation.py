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
        RotationType,
        SizeType,
        TranslationType,
        VelocityType,
    )

__all__ = ("SampleAnnotation",)


@dataclass
@SCHEMAS.register(SchemaName.SAMPLE_ANNOTATION)
class SampleAnnotation(SchemaBase):
    """A dataclass to represent schema table of `sample_annotation.json`."""

    token: str
    sample_token: str
    instance_token: str
    attribute_tokens: list[str]
    visibility_token: str
    translation: TranslationType
    size: SizeType
    rotation: RotationType
    num_lidar_pts: int
    num_radar_pts: int
    next: str  # noqa: A003
    prev: str
    velocity: VelocityType | None = field(default=None)
    acceleration: AccelerationType | None = field(default=None)

    # shortcuts
    category_name: str = field(init=False)

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        objs: list[Self] = []
        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            sample_token: str = record["sample_token"]
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
            objs.append(
                cls(
                    token=token,
                    sample_token=sample_token,
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
            )
        return objs
