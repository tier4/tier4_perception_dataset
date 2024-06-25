from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if TYPE_CHECKING:
    from t4_devkit.typing import KeypointType

__all__ = ("Keypoint",)


@dataclass
@SCHEMAS.register(SchemaName.KEYPOINT)
class Keypoint(SchemaBase):
    """A dataclass to represent schema table of `keypoint.json`."""

    token: str
    sample_data_token: str
    instance_token: str
    visibility_token: str
    keypoint_tokens: list[str]
    keypoints: KeypointType
    num_keypoints: int

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        objs: list[Self] = []
        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            sample_data_token: str = record["sample_data_token"]
            instance_token: str = record["instance_token"]
            visibility_token: str = record["visibility_token"]
            keypoint_tokens: list[str] = record["keypoint_tokens"]
            keypoints = np.array(record["keypoints"])
            num_keypoints: int = record["num_keypoints"]
            objs.append(
                cls(
                    token=token,
                    sample_data_token=sample_data_token,
                    instance_token=instance_token,
                    visibility_token=visibility_token,
                    keypoint_tokens=keypoint_tokens,
                    keypoints=keypoints,
                    num_keypoints=num_keypoints,
                )
            )
        return objs
