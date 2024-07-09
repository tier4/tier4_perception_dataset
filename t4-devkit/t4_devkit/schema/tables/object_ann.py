from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if TYPE_CHECKING:
    from t4_devkit.typing import MaskType, RoiType

__all__ = ("ObjectAnn",)


@dataclass
@SCHEMAS.register(SchemaName.OBJECT_ANN)
class ObjectAnn(SchemaBase):
    """A dataclass to represent schema table of `object_ann.json`.

    Attributes:
        token (str): Unique record identifier.
        sample_data_token (str): Foreign key pointing to the sample data, which must be a keyframe image.
        instance_token (str): Foreign key pointing to the instance.
        category_token (str): Foreign key pointing to the object category.
        attribute_tokens (list[str]): Foreign keys. List of attributes for this annotation.
        bbox (RoiType): Annotated bounding box. Given as [xmin, ymin, xmax, ymax].
        mask (MaskType): Instance mask using the COCO format.
    """

    token: str
    sample_data_token: str
    instance_token: str
    category_token: str
    attribute_tokens: list[str]
    bbox: RoiType
    mask: MaskType

    # shortcuts
    category_name: str = field(init=False)

    @staticmethod
    def shortcuts() -> tuple[str]:
        return ("category_name",)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
