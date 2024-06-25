from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Self


@dataclass(frozen=True)
class SchemaBase(ABC):
    """

    Args:
        ABC (_type_): _description_
    """

    @classmethod
    @abstractmethod
    def from_json(cls, filepath: str) -> Self:
        """Construct dataclass from json file.

        Args:
            filepath (str): Filepath to json.

        Returns:
            Self: Instantiated schema dataclass.
        """
        ...
