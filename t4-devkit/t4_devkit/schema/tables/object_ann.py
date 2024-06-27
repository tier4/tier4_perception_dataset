from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from t4_devkit.common.io import load_json
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
    """A dataclass to represent schema table of `object_ann.json`."""

    token: str
    sample_data_token: str
    instance_token: str
    category_token: str
    attribute_tokens: list[str]
    bbox: RoiType
    mask: MaskType

    # shortcuts
    category_name: str = field(init=False)

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        record_list: list[dict[str, Any]] = load_json(filepath)
        return [cls(**record) for record in record_list]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
