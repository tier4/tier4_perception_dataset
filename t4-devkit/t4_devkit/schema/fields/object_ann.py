from dataclasses import dataclass
from typing import Any

from t4_devkit.common import load_json
from t4_devkit.typing import BboxType, MaskType
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("ObjectAnn",)


@dataclass(frozen=True)
class ObjectAnn(SchemaBase):
    token: str
    sample_data_token: str
    instance_token: str
    category_token: str
    attribute_tokens: list[str]
    bbox: BboxType
    mask: MaskType

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        return cls(**record)
