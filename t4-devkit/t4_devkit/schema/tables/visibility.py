from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any
import warnings

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
    """An enum to represent visibility levels.

    Attributes:
        FULL: No occlusion for the object.
        MOST: Object is occluded, but by less than 50%.
        PARTIAL: Object is occluded, but by more than 50%.
        NONE: Object is 90-100% occluded and no points/pixels are visible in the label.
        UNAVAILABLE: Visibility level is not specified.
    """

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
    """A dataclass to represent schema table of `visibility.json`.

    Attributes:
        token (str): Unique record identifier.
        level (VisibilityLevel): Visibility level.
        description (str): Description of visibility level.
    """

    token: str
    level: VisibilityLevel
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        token: str = data["token"]
        level = VisibilityLevel.from_value(data["level"])
        description: str = data["description"]

        return cls(token=token, level=level, description=description)
