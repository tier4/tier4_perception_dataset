from dataclasses import dataclass
from typing import Any

from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

__all__ = ("Instance",)


@dataclass
@SCHEMAS.register(SchemaName.INSTANCE)
class Instance(SchemaBase):
    """A dataclass to represent schema table of `instance.json`.

    Attributes:
        token (str): Unique record identifier.
        category_token (str): Foreign key pointing to the object category.
        instance_name (str): Dataset name and instance ID defined in annotation tool.
        nbr_annotations (int): Number of annotations of this instance.
        first_annotation_token (str): Foreign key pointing to the first annotation of this instance.
        last_annotation_token (str): Foreign key pointing to the last annotation of this instance.
    """

    token: str
    category_token: str
    instance_name: str
    nbr_annotations: int
    first_annotation_token: str
    last_annotation_token: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)
