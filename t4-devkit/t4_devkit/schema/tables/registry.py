from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable, KeysView

from ..name import SchemaName

if TYPE_CHECKING:
    from .base import SchemaTable

__all__ = ("SCHEMAS",)


class SchemaRegistry:
    """A manager class to register schema tables."""

    def __init__(self) -> None:
        self.__schemas: dict[str, SchemaTable] = {}

    @property
    def schemas(self) -> dict[str, SchemaTable]:
        return self.__schemas

    def __len__(self) -> int:
        return len(self.__schemas)

    def __contains__(self, key: str | SchemaName) -> bool:
        if isinstance(key, SchemaName):
            key = key.value

        return key in self.keys()

    def get(self, key: str | SchemaName) -> SchemaTable:
        if isinstance(key, SchemaName):
            key = key.value

        if key not in self.__schemas:
            raise KeyError(f"{key} has not registered yet.")
        return self.__schemas[key]

    def keys(self) -> KeysView[str]:
        return self.__schemas.keys()

    def register(self, name: SchemaName, *, force: bool = False) -> Callable:
        if not isinstance(name, SchemaName):
            raise TypeError(f"name must be SchemaName, but got {type(name)}.")

        def _register_decorator(obj: object) -> object:
            self._add_module(obj, name=name, force=force)
            return obj

        return _register_decorator

    def _add_module(self, module: object, name: SchemaName, *, force: bool = False) -> None:
        if not inspect.isclass(module):
            raise TypeError(f"module must be a class, but got {type(module)}.")

        if not force and name in self:
            raise KeyError(f"{name.value} has already been registered.")

        self.__schemas[name.value] = module

    def build_from_json(self, key: str | SchemaName, filepath: str) -> list[SchemaTable]:
        """Build schema dataclass from json.

        Args:
            key (str): Name of schema field.
            filepath (str): Path to schema json file.

        Returns:
            Instantiated dataclass.
        """
        if isinstance(key, SchemaName):
            key = key.value

        schema: SchemaTable = self.get(key)
        return schema.from_json(filepath)


SCHEMAS = SchemaRegistry()
