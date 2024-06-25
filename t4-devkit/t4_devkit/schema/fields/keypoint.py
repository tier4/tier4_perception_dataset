from dataclasses import dataclass
from typing import Any

import numpy as np
from t4_devkit.common import load_json
from t4_devkit.typing import KeypointType
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("Keypoint",)


@dataclass(frozen=True)
class Keypoint(SchemaBase):
    token: str
    sample_data_token: str
    instance_token: str
    visibility_token: str
    keypoint_tokens: list[str]
    keypoints: KeypointType
    num_keypoints: int

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        token: str = record["token"]
        sample_data_token: str = record["sample_data_token"]
        instance_token: str = record["instance_token"]
        visibility_token: str = record["visibility_token"]
        keypoint_tokens: list[str] = record["keypoint_tokens"]
        keypoints = np.array(record["keypoints"])
        num_keypoints: int = record["num_keypoints"]
        return cls(
            token=token,
            sample_data_token=sample_data_token,
            instance_token=instance_token,
            visibility_token=visibility_token,
            keypoint_tokens=keypoint_tokens,
            keypoints=keypoints,
            num_keypoints=num_keypoints,
        )
