from dataclasses import dataclass

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("Category",)


@dataclass(frozen=True)
class Category(SchemaBase):
    token: str
    name: str
    description: str

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record = load_json(filepath)
        return cls(**record)
