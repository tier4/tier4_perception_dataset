from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
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
    """A dataclass to represent schema table of `keypoint.json`.

    Attributes:
        token (str): Unique record identifier.
        sample_data_token (str): Foreign key pointing to the sample data, which must be a keyframe image.
        instance_token (str): Foreign key pointing to the instance.
        category_tokens (list[str]): Foreign key pointing to keypoints categories.
        keypoints (KeypointType): Annotated keypoints. Given as a list of [x, y].
        num_keypoints (int): The number of keypoints to be annotated.
    """

    token: str
    sample_data_token: str
    instance_token: str
    category_tokens: list[str]
    keypoints: KeypointType
    num_keypoints: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        token: str = data["token"]
        sample_data_token: str = data["sample_data_token"]
        instance_token: str = data["instance_token"]
        category_tokens: list[str] = data["category_tokens"]
        keypoints = np.array(data["keypoints"])
        num_keypoints: int = data["num_keypoints"]

        return cls(
            token=token,
            sample_data_token=sample_data_token,
            instance_token=instance_token,
            category_tokens=category_tokens,
            keypoints=keypoints,
            num_keypoints=num_keypoints,
        )
