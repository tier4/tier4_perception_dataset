from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from t4_devkit.common.io import load_json
from typing_extensions import Self

__all__ = ("SchemaBase", "SchemaTable")


@dataclass
class SchemaBase(ABC):
    """Abstract base dataclass of schema tables."""

    token: str

    @staticmethod
    def shortcuts() -> tuple[str, ...] | None:
        """Return a sequence of shortcut field names.

        Returns:
            Returns None if there is no shortcut. Otherwise, returns sequence of shortcut field names.
        """
        return None

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        """Construct dataclass from json file.

        Args:
            filepath (str): Filepath to json.

        Returns:
            List of instantiated schema dataclasses.
        """
        records: list[dict[str, Any]] = load_json(filepath)
        return [cls.from_dict(data) for data in records]

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct dataclass from dict.

        Args:
            data (dict[str, Any]): Dict data.

        Returns:
            Instantiated schema dataclass.
        """
        ...


SchemaTable = TypeVar("SchemaTable", bound=SchemaBase)
