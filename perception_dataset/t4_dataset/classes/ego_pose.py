from typing import Any, Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class EgoPoseRecord(AbstractRecord):
    def __init__(
        self,
        translation: Dict[str, float],
        rotation: Dict[str, float],
        timestamp: int,
    ):
        super().__init__()

        assert {"x", "y", "z"} == set(translation.keys())
        assert {"w", "x", "y", "z"} == set(rotation.keys())

        self.translation: Dict[str, float] = translation
        self.rotation: Dict[str, float] = rotation
        self.timestamp: int = timestamp

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token": self.token,
            "translation": [
                self.translation["x"],
                self.translation["y"],
                self.translation["z"],
            ],
            "rotation": [
                self.rotation["w"],
                self.rotation["x"],
                self.rotation["y"],
                self.rotation["z"],
            ],
            "timestamp": self.timestamp,
        }
        return d


class EgoPoseTable(AbstractTable):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#ego_posejson"""

    FILENAME = "ego_pose" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> EgoPoseRecord:
        return EgoPoseRecord(**kwargs)
