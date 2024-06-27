from dataclasses import dataclass
from typing import Any

from t4_devkit.common.io import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

__all__ = ("Map",)


@dataclass
@SCHEMAS.register(SchemaName.MAP)
class Map(SchemaBase):
    """A dataclass to represent schema table of `map.json`."""

    token: str
    log_tokens: list[str]
    category: str
    filename: str

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        record_list: list[dict[str, Any]] = load_json(filepath)
        return [cls(**record) for record in record_list]
