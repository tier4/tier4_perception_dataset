from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from typing_extensions import Self

__all__ = ("SchemaBase", "SchemaTable")


@dataclass
class SchemaBase(ABC):
    """Abstract base dataclass of schema tables."""

    token: str

    @classmethod
    @abstractmethod
    def from_json(cls, filepath: str) -> list[Self]:
        """Construct dataclass from json file.

        Args:
            filepath (str): Filepath to json.

        Returns:
            List of instantiated schema dataclasses.
        """
        ...


SchemaTable = TypeVar("SchemaTable", bound=SchemaBase)
