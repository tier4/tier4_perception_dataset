from __future__ import annotations

from typing import TYPE_CHECKING

from .tables import SCHEMAS

if TYPE_CHECKING:
    from .name import SchemaName
    from .tables import SchemaTable

__all__ = ("build_schema",)


def build_schema(name: str | SchemaName, filepath: str) -> list[SchemaTable]:
    """Build schema dataclass from json file path.

    Args:
        name (str | SchemaName): Name of schema table.
        filepath (str): Path to json file.

    Returns:
        List of schema dataclasses.
    """
    return SCHEMAS.build_from_json(name, filepath)
