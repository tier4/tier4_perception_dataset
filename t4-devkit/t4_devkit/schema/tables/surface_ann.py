from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if TYPE_CHECKING:
    from t4_devkit.typing import MaskType

__all__ = ("SurfaceAnn",)


@dataclass
@SCHEMAS.register(SchemaName.SURFACE_ANN)
class SurfaceAnn(SchemaBase):
    """A dataclass to represent schema table of `surface_ann.json`.

    Attributes:
        token (str): Unique record identifier.
        sample_data_token (str): Foreign key pointing to the sample data, which must be a keyframe image.
        category_token (str): Foreign key pointing to the surface category.
        mask (MaskType): Segmentation mask using the COCO format.
    """

    token: str
    sample_data_token: str
    category_token: str
    mask: MaskType

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)
