from __future__ import annotations

import json
from typing import Dict, List

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class ObjectAnnRecord(AbstractRecord):
    def __init__(
        self,
        sample_data_token: str,
        instance_token: str,
        category_token: str,
        attribute_tokens: str,
        bbox: List[float],
        mask: Dict[str, any],
    ):
        super().__init__()

        assert len(bbox) == 4

        self._sample_data_token: str = sample_data_token
        self._instance_token: str = instance_token
        self._category_token: str = category_token
        self._attribute_tokens: List[str] = attribute_tokens
        self._bbox: List[float] = bbox
        self._mask: Dict[str, any] = mask

    def to_dict(self):
        d = {
            "token": self.token,
            "sample_data_token": self._sample_data_token,
            "instance_token": self._instance_token,
            "category_token": self._category_token,
            "attribute_tokens": self._attribute_tokens,
            "bbox": [
                self._bbox[0],
                self._bbox[1],
                self._bbox[2],
                self._bbox[3],
            ],
            "mask": self._mask,
        }
        return d


class ObjectAnnTable(AbstractTable[ObjectAnnRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#sample_annotationjson"""

    FILENAME = "object_ann" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(
        self,
        sample_data_token: str,
        instance_token: str,
        category_token: str,
        attribute_tokens: str,
        bbox: List[float],
        mask: Dict[str, any],
    ):
        record = ObjectAnnRecord(
            sample_data_token=sample_data_token,
            instance_token=instance_token,
            category_token=category_token,
            attribute_tokens=attribute_tokens,
            bbox=bbox,
            mask=mask,
        )
        return record

    @classmethod
    def from_json(cls, filepath: str) -> ObjectAnnTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls()
        for item in items:
            record = ObjectAnnRecord(
                sample_data_token=item["sample_data_token"],
                instance_token=item["instance_token"],
                category_token=item["category_token"],
                attribute_tokens=item["attribute_tokens"],
                bbox=item["bbox"],
                mask=item["mask"],
            )
            record.token = item["token"]
            table.set_record_to_table(record)

        return table
