from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from t4_devkit.common.io import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if TYPE_CHECKING:
    from .sensor import SensorChannel

__all__ = ("Sample",)


@dataclass
@SCHEMAS.register(SchemaName.SAMPLE)
class Sample(SchemaBase):
    """A dataclass to represent schema table of `sample.json`."""

    token: str
    timestamp: int
    scene_token: str
    next: str  # noqa: A003
    prev: str

    # shortcuts
    data: dict[SensorChannel, str] = field(default_factory=dict, init=False)
    ann_3ds: list[str] = field(default_factory=list, init=False)
    ann_2ds: list[str] = field(default_factory=list, init=False)

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        record_list: list[dict[str, Any]] = load_json(filepath)
        return [cls(**record) for record in record_list]
