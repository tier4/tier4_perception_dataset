from typing import Any, Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class SceneRecord(AbstractRecord):
    def __init__(
        self,
        name: str,
        description: str,
        log_token: str,
        nbr_samples: int = 0,
        first_sample_token: str = "",
        last_sample_token: str = "",
    ):
        super().__init__()

        self.name: str = name
        self.description: str = description
        self.log_token: str = log_token
        self.nbr_samples: int = nbr_samples
        self.first_sample_token: str = first_sample_token
        self.last_sample_token: str = last_sample_token

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token": self.token,
            "name": self.name,
            "description": self.description,
            "log_token": self.log_token,
            "nbr_samples": self.nbr_samples,
            "first_sample_token": self.first_sample_token,
            "last_sample_token": self.last_sample_token,
        }
        return d


class SceneTable(AbstractTable[SceneRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#scenejson"""

    FILENAME = "scene" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> SceneRecord:
        return SceneRecord(**kwargs)
