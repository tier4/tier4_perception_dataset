from dataclasses import dataclass
import sys
from typing import Any
import warnings

from t4_devkit.common.io import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum

__all__ = ("Visibility", "VisibilityLevel")


class VisibilityLevel(StrEnum):
    """An enum to represent visibility levels."""

    FULL = "full"
    MOST = "most"
    PARTIAL = "partial"
    NONE = "none"
    UNAVAILABLE = "unavailable"

    @classmethod
    def from_value(cls, level: str) -> Self:
        """Load member from its value."""
        if level not in cls.__members__.values():
            return cls._from_alias(level)
        return cls(level)

    @staticmethod
    def _from_alias(level: str) -> Self:
        """Load member from alias format of level.

        Args:
        ----
            level (str): Level of visibility.
        """
        if level == "v0-40":
            return VisibilityLevel.NONE
        elif level == "v40-60":
            return VisibilityLevel.PARTIAL
        elif level == "v60-80":
            return VisibilityLevel.MOST
        elif level == "v80-100":
            return VisibilityLevel.FULL
        else:
            warnings.warn(
                f"level: {level} is not supported, Visibility.UNAVAILABLE will be assigned."
            )
            return VisibilityLevel.UNAVAILABLE


@dataclass
@SCHEMAS.register(SchemaName.VISIBILITY)
class Visibility(SchemaBase):
    """A dataclass to represent schema table of `visibility.json`."""

    token: str
    level: VisibilityLevel
    description: str

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        objs: list[Self] = []
        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            level = VisibilityLevel.from_value(record["level"])
            description: str = record["description"]
            objs.append(cls(token=token, level=level, description=description))
        return objs
