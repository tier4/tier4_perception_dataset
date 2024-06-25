from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyquaternion import Quaternion
from t4_devkit.common import load_json
from t4_devkit.typing import RotationType, TranslationType
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("EgoPose",)


@dataclass(frozen=True)
class EgoPose(SchemaBase):
    token: str
    translation: TranslationType
    rotation: RotationType
    timestamp: int

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        token: str = record["token"]
        translation = np.array(record["translation"])
        rotation = Quaternion(record["rotation"])
        timestamp: int = record["timestamp"]
        return cls(token=token, translation=translation, rotation=rotation, timestamp=timestamp)
