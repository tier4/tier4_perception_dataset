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
    from t4_devkit.typing import RotationType, TranslationType

__all__ = ("EgoPose",)


@dataclass
@SCHEMAS.register(SchemaName.EGO_POSE)
class EgoPose(SchemaBase):
    """A dataclass to represent schema table of `ego_pose.json`."""

    token: str
    translation: TranslationType
    rotation: RotationType
    timestamp: int

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        objs: list[Self] = []
        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            translation = np.array(record["translation"])
            rotation = Quaternion(record["rotation"])
            timestamp: int = record["timestamp"]
            objs.append(
                cls(token=token, translation=translation, rotation=rotation, timestamp=timestamp)
            )
        return objs
