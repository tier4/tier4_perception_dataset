from dataclasses import dataclass
from typing import Any

from t4_devkit.common.io import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

__all__ = ("Instance",)


@dataclass
@SCHEMAS.register(SchemaName.INSTANCE)
class Instance(SchemaBase):
    """A dataclass to represent schema table of `instance.json`."""

    token: str
    category_token: str
    instance_name: str
    nbr_annotations: int
    first_annotation_token: str
    last_annotation_token: str

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        record_list: list[dict[str, Any]] = load_json(filepath)
        return [cls(**record) for record in record_list]
