from dataclasses import dataclass
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("Map",)


@dataclass(frozen=True)
class Map(SchemaBase):
    token: str
    log_tokens: list[str]
    category: str
    filename: str

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        return cls(**record)
