from dataclasses import dataclass
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("Instance",)


@dataclass(frozen=True)
class Instance(SchemaBase):
    token: str
    category_token: str
    instance_name: str
    nbr_annotations: int
    first_annotation_token: str
    last_annotation_token: str

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        return cls(**record)
