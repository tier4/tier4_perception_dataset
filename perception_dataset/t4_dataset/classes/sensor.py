from __future__ import annotations

import json
from typing import Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class SensorRecord(AbstractRecord):
    def __init__(
        self,
        channel: str,
        modality: str,
    ):
        super().__init__()

        self._channel: str = channel
        self._modality: str = modality

    def to_dict(self):
        d = {
            "token": self.token,
            "channel": self._channel,
            "modality": self._modality,
        }
        return d


class SensorTable(AbstractTable[SensorRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#sensorjson"""

    FILENAME = "sensor" + EXTENSION_ENUM.JSON.value

    def __init__(self, channel_to_modality: Dict[str, str]):
        super().__init__()
        self._channel_to_modality: Dict[str, str] = channel_to_modality
        self._channel_to_token: Dict[str, str] = {}

    def _to_record(self, **kwargs) -> SensorRecord:
        return SensorRecord(**kwargs)

    def get_token_from_channel(self, channel: str):
        if channel in self._channel_to_token:
            token = self._channel_to_token[channel]
        else:
            modality = self._channel_to_modality[channel]
            token = self.insert_into_table(channel=channel, modality=modality)
            self._channel_to_token[channel] = token

        return token

    @classmethod
    def from_json(cls, filepath: str, channel_to_modality: Dict[str, str]) -> SensorTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls(channel_to_modality=channel_to_modality)
        for item in items:
            record = SensorRecord(channel=item["channel"], modality=item["modality"])
            record.token = item["token"]
            table.set_record_to_table(record)

        return table
