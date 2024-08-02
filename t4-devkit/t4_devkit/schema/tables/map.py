from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

__all__ = ("Map",)


@dataclass
@SCHEMAS.register(SchemaName.MAP)
class Map(SchemaBase):
    """A dataclass to represent schema table of `map.json`.

    Attributes:
        token (str): Unique record identifier.
        log_tokens (str): Foreign keys pointing the log tokens.
        category (str): Map category.
        filename (str): Relative path to the file with the map mask.
    """

    token: str
    log_tokens: list[str]
    category: str
    filename: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)
