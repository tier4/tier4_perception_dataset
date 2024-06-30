from dataclasses import dataclass
from typing import Any

from t4_devkit.common.io import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

__all__ = ("Scene",)


@dataclass
@SCHEMAS.register(SchemaName.SCENE)
class Scene(SchemaBase):
    """A dataclass to represent schema table of `scene.json`.

    Attributes:
    ----------
        token (str): Unique record identifier.
        name (str): Short string identifier.
        description (str): Longer description for the scene.
        log_token (str): Foreign key pointing to log from where the data was extracted.
        nbr_samples (int): Number of samples in the scene.
        first_sample_token (str): Foreign key pointing to the first sample in scene.
        last_sample_token (str): Foreign key pointing to the last sample in scene.
    """

    token: str
    name: str
    description: str
    log_token: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        record_list: list[dict[str, Any]] = load_json(filepath)
        return [cls(**record) for record in record_list]
