from dataclasses import dataclass
from typing import Any

from t4_devkit.common.io import load_json
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
    def from_json(cls, filepath: str) -> list[Self]:
        record_list: list[dict[str, Any]] = load_json(filepath)
        return [cls(**record) for record in record_list]
