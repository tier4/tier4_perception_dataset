from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pyquaternion import Quaternion
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if TYPE_CHECKING:
    from t4_devkit.typing import RotationType, TranslationType

__all__ = ("EgoPose",)


@dataclass
@SCHEMAS.register(SchemaName.EGO_POSE)
class EgoPose(SchemaBase):
    """A dataclass to represent schema table of `ego_pose.json`.

    Attributes:
        token (str): Unique record identifier.
        translation (TranslationType): Coordinate system origin given as [x, y, z] in [m].
        rotation (RotationType): Coordinate system orientation given as quaternion [w, x, y, z].
        timestamp (int): Unix time stamp.
    """

    token: str
    translation: TranslationType
    rotation: RotationType
    timestamp: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        token: str = data["token"]
        translation = np.array(data["translation"])
        rotation = Quaternion(data["rotation"])
        timestamp: int = data["timestamp"]

        return cls(token=token, translation=translation, rotation=rotation, timestamp=timestamp)
