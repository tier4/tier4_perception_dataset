from dataclasses import dataclass
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase

__all__ = ("Scene",)


@dataclass(frozen=True)
class Scene(SchemaBase):
    token: str
    name: str
    description: str
    log_token: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        return cls(**record)
