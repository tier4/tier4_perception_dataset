from dataclasses import dataclass, field
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

__all__ = ("Log",)


@dataclass
@SCHEMAS.register(SchemaName.LOG)
class Log(SchemaBase):
    """A dataclass to represent schema table of `log.json`."""

    token: str
    logfile: str
    vehicle: str
    data_captured: str
    location: str

    # shortcuts
    map_token: str = field(init=False)

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        record_list: list[dict[str, Any]] = load_json(filepath)
        return [cls(**record) for record in record_list]
