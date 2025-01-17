from __future__ import annotations

import json
from typing import Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class LidarSegRecord(AbstractRecord):
    """A lidarseg record."""

    def __init__(self, sample_data_token: str, filename: str):
        super().__init__()
        self.sample_data_token = sample_data_token
        self.filename: str = filename

    def to_dict(self) -> Dict[str, str]:
        d = {
            "token": self.token,
            "sample_data_token": self.sample_data_token,
            "filename": self.filename,
        }
        return d


class LidarSegTable(AbstractTable[LidarSegRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#lidarsegjson"""

    FILENAME = "lidarseg" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, sample_data_token: str, filename: str) -> LidarSegRecord:
        record = LidarSegRecord(sample_data_token=sample_data_token, filename=filename)
        return record

    @classmethod
    def from_json(
        cls,
        filepath: str,
    ) -> LidarSegTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls()
        for item in items:
            record = LidarSegRecord(
                sample_data_token=item["sample_data_token"], description=item["filename"]
            )

            record.token = item["token"]
            table.set_record_to_table(record)

        return table
