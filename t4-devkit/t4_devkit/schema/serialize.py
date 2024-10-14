from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Sequence

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
    dict_factory = partial(_schema_as_dict_factory, excludes=data.shortcuts())
    return asdict(data, dict_factory=dict_factory)


def _schema_as_dict_factory(
    data: list[tuple[str, Any]], *, excludes: Sequence[str] | None = None
) -> dict:
    """A factory to convert schema dataclass field to dict data.

    Args:
        data (list[tuple[str, Any]]): Some data of dataclass field.
        excludes (Sequence[str] | None, optional): Sequence of field names to be excluded.

    Returns:
        Converted dict data.
    """

    def _convert_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, Quaternion):
            return value.q.tolist()
        elif isinstance(value, Enum):
            return value.value
        return value

    return (
        {k: _convert_value(v) for k, v in data}
        if excludes is None
        else {k: _convert_value(v) for k, v in data if k not in excludes}
    )
