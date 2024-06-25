from dataclasses import dataclass
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("Sample",)


@dataclass(frozen=True)
class Sample(SchemaBase):
    token: str
    timestamp: int
    scene_token: str
    next: str  # noqa: A003
    prev: str

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        return cls(**record)
