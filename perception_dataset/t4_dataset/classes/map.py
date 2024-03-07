from typing import Any, Dict, List

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class MapRecord(AbstractRecord):
    def __init__(
        self,
        log_tokens: List[str],
        category: str,
        filename: str,
    ):
        super().__init__()

        self.log_tokens: List[str] = log_tokens
        self.category: str = category
        self.filename: str = filename

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token": self.token,
            "log_tokens": self.log_tokens,
            "category": self.category,
            "filename": self.filename,
        }
        return d


class MapTable(AbstractTable[MapRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#mapjson"""

    FILENAME = "map" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> MapRecord:
        return MapRecord(**kwargs)
