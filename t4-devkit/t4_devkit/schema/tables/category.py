from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

__all__ = ("Category",)


@dataclass
@SCHEMAS.register(SchemaName.CATEGORY)
class Category(SchemaBase):
    """A dataclass to represent schema table of `category.json`.

    Attributes:
        token (str): Unique record identifier.
        name (str): Category name.
        description (str): Category description.
    """

    token: str
    name: str
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)
