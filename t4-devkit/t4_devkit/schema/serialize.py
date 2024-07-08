from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from pyquaternion import Quaternion

if TYPE_CHECKING:
    from .tables import SchemaTable


__all__ = ("serialize_schemas", "serialize_schema")


def serialize_schemas(data: list[SchemaTable]) -> list[dict]:
    """Serialize a list of schema dataclasses into list of dict.

    Args:
        data (list[SchemaTable]): List of schema dataclasses.

    Returns:
        Serialized list of dict data.
    """
    return [serialize_schema(d) for d in data]


def serialize_schema(data: SchemaTable) -> dict:
    """Serialize a schema dataclass into dict.

    Args:
        data (SchemaTable): Schema dataclass.

    Returns:
        Serialized dict data.
    """
    return asdict(data, dict_factory=_schema_as_dict_factory)


def _schema_as_dict_factory(data: SchemaTable) -> dict:
    """A factory to convert schema dataclass field to dict data.

    Args:
        data (SchemaTable): Some data of dataclass field.

    Returns:
        Converted dict data.
    """

    def _convet_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, Quaternion):
            return value.q.tolist()
        elif isinstance(value, Enum):
            return value.value
        return value

    return {k: _convet_value(v) for k, v in data}
