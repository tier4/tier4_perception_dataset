from dataclasses import dataclass
import sys
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum

__all__ = ("Visibility", "VisibilityLevel")


class VisibilityLevel(StrEnum):
    FULL = "full"
    MOST = "most"
    PARTIAL = "partial"
    NONE = "none"


@dataclass(frozen=True)
class Visibility(SchemaBase):
    token: str
    level: VisibilityLevel
    description: str

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        token: str = record["token"]
        level = VisibilityLevel(record["level"])
        description: str = record["description"]
        return cls(token=token, level=level, description=description)
