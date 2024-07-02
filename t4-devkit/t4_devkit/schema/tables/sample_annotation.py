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
    """A dataclass to represent schema table of `sample_annotation.json`.

    Attributes:
        token (str): Unique record identifier.
        sample_token (str): Foreign key pointing the sample.
        instance_token (str): Foreign key pointing the object instance.
        attribute_tokens (list[str]): Foreign keys. List of attributes for this annotation.
        visibility_token (str): Foreign key pointing the object visibility.
        translation (TranslationType): Bounding box location given as [x, y, z] in [m].
        size (SizeType): Bounding box size given as [width, length, height] in [m].
        rotation (RotationType): Bounding box orientation given as quaternion [w, x, y, z].
        num_lidar_pts (int): Number of lidar points in this box.
        num_radar_pts (int): Number of radar points in this box.
        next (str): Foreign key pointing the annotation that follows this in time.
            Empty if this is the last annotation for this object.
        prev (str): Foreign key pointing the annotation that precedes this in time.
            Empty if this the first annotation for this object.
        velocity (VelocityType | None, optional): Bounding box velocity given as
            [vx, vy, vz] in [m/s].
        acceleration (AccelerationType | None, optional): Bonding box acceleration
            given as [ax, ay, av] in [m/s^2].

    Shortcuts:
    ---------
        category_name (str): Category name. This should be set after instantiated.
    """

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
