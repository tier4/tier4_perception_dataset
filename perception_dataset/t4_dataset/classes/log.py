from typing import Any, Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class LogRecord(AbstractRecord):
    def __init__(
        self,
        logfile: str,
        vehicle: str,
        data_captured: str,
        location: str,
    ):
        super().__init__()

        self.logfile: str = logfile
        self.vehicle: str = vehicle
        self.data_captured: str = data_captured
        self.location: str = location

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token": self.token,
            "logfile": self.logfile,
            "vehicle": self.vehicle,
            "data_captured": self.data_captured,
            "location": self.location,
        }
        return d


class LogTable(AbstractTable):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#logjson"""

    FILENAME = "log" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> LogRecord:
        return LogRecord(**kwargs)
