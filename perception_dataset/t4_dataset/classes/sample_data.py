from __future__ import annotations

import json

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class SampleDataRecord(AbstractRecord):
    def __init__(
        self,
        sample_token: str,
        ego_pose_token: str,
        calibrated_sensor_token: str,
        filename: str,
        fileformat: str,
        timestamp: int,
        is_key_frame: bool,
        width: int = 0,
        height: int = 0,
        next_token: str = "",
        prev_token: str = "",
        is_valid: bool = True,
    ):
        super().__init__()

        self.sample_token: str = sample_token
        self.ego_pose_token: str = ego_pose_token
        self.calibrated_sensor_token: str = calibrated_sensor_token
        self.filename: str = filename
        self.fileformat: str = fileformat
        self.width: int = width
        self.height: int = height
        self.timestamp: int = timestamp
        self.is_key_frame: bool = is_key_frame
        self.next: str = next_token
        self.prev: str = prev_token
        self._is_valid: bool = is_valid

    def to_dict(self):
        d = {
            "token": self.token,
            "sample_token": self.sample_token,
            "ego_pose_token": self.ego_pose_token,
            "calibrated_sensor_token": self.calibrated_sensor_token,
            "filename": self.filename,
            "fileformat": self.fileformat,
            "width": self.width,
            "height": self.height,
            "timestamp": self.timestamp,
            "is_key_frame": self.is_key_frame,
            "next": self.next,
            "prev": self.prev,
            "is_valid": self._is_valid,
        }
        return d


class SampleDataTable(AbstractTable[SampleDataRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#sample_datajson"""

    FILENAME = "sample_data" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> SampleDataRecord:
        return SampleDataRecord(**kwargs)

    @classmethod
    def from_json(cls, filepath: str) -> SampleDataTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls()
        for item in items:
            record = SampleDataRecord(
                sample_token=item["sample_token"],
                ego_pose_token=item["ego_pose_token"],
                calibrated_sensor_token=item["calibrated_sensor_token"],
                filename=item["filename"],
                fileformat=item["fileformat"],
                timestamp=item["timestamp"],
                is_key_frame=item["is_key_frame"],
                width=item["width"],
                height=item["height"],
                next_token=item["next"],
                prev_token=item["prev"],
                is_valid=item["is_valid"],
            )
            record.token = item["token"]
            table.set_record_to_table(record)

        return table
