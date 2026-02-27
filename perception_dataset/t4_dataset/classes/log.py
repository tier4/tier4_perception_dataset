from __future__ import annotations

import json
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


class LogTable(AbstractTable[LogRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#logjson"""

    FILENAME = "log" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> LogRecord:
        # date_capturedとdata_capturedの両方に対応
        if "date_captured" in kwargs and "data_captured" not in kwargs:
            kwargs["data_captured"] = kwargs.pop("date_captured")
        return LogRecord(**kwargs)

    @classmethod
    def from_json(cls, filepath: str) -> LogTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls()
        for item in items:
            # date_capturedとdata_capturedの両方に対応
            captured = item.get("data_captured") or item.get("date_captured")
            record = LogRecord(
                logfile=item["logfile"],
                vehicle=item["vehicle"],
                data_captured=captured,
                location=item["location"],
            )
            record.token = item["token"]
            table.set_record_to_table(record)

        return table
