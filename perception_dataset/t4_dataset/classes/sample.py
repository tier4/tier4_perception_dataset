from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class SampleRecord(AbstractRecord):
    def __init__(
        self,
        timestamp: int,
        scene_token: str,
        next_token: str = "",
        prev_token: str = "",
    ):
        super().__init__()

        self.timestamp: int = timestamp
        self.scene_token: str = scene_token
        self.next: str = next_token
        self.prev: str = prev_token

    def to_dict(self):
        d = {
            "token": self.token,
            "timestamp": self.timestamp,
            "scene_token": self.scene_token,
            "next": self.next,
            "prev": self.prev,
        }
        return d


class SampleTable(AbstractTable[SampleRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#samplejson"""

    FILENAME = "sample" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> SampleRecord:
        return SampleRecord(**kwargs)
