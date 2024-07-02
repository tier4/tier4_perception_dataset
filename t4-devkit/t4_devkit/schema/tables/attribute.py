from dataclasses import dataclass
from typing import Any

from t4_devkit.common.io import load_json
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
    def from_json(cls, filepath: str) -> list[Self]:
        record_list: list[dict[str, Any]] = load_json(filepath)
        return [cls(**record) for record in record_list]
