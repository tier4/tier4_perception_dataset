from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

__all__ = ("Attribute",)


@dataclass
@SCHEMAS.register(SchemaName.ATTRIBUTE)
class Attribute(SchemaBase):
    """A dataclass to represent schema table of `attribute.json`.

    Attributes:
        token (str): Unique record identifier.
        name (str): Attribute name.
        description (str): Attribute description.
    """

    token: str
    name: str
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)
