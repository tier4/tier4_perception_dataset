from dataclasses import dataclass
from typing import Any

from t4_devkit.common import load_json
from t4_devkit.typing import MaskType
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("SurfaceAnn",)


@dataclass(frozen=True)
class SurfaceAnn(SchemaBase):
    token: str
    sample_data_token: str
    category_token: str
    mask: MaskType

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        return cls(**record)
