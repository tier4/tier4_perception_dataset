from typing import Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class VisibilityRecord(AbstractRecord):
    def __init__(self, level: str, description: str):
        super().__init__()
        self._level: str = level
        self._description: str = description

    def to_dict(self) -> Dict[str, str]:
        d = {
            "token": self.token,
            "level": self._level,
            "description": self._description,
        }
        return d


class VisibilityTable(AbstractTable):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#visibilityjson"""

    FILENAME = "visibility" + EXTENSION_ENUM.JSON.value

    def __init__(self, level_to_description: Dict[str, str], default_value: str):
        super().__init__()
        self._level_to_token: Dict[str, str] = {}
        self._level_to_description: Dict[str, str] = level_to_description
        self._description_default_value: str = default_value

    def _to_record(self, level: str, description: str) -> VisibilityRecord:
        record = VisibilityRecord(
            level=level,
            description=description,
        )
        return record

    def get_token_from_level(self, level: str) -> str:
        if level in self._level_to_token:
            token = self._level_to_token[level]
        else:
            description = self._level_to_description.get(level, self._description_default_value)
            token = self.insert_into_table(level=level, description=description)
            self._level_to_token[level] = token

        return token
