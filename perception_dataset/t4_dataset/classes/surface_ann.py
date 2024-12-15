from __future__ import annotations

import json
from typing import Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class SurfaceAnnRecord(AbstractRecord):
    def __init__(
        self,
        category_token: str,
        mask: Dict[str, any],
        sample_data_token: str,
        automatic_annotation: bool,
    ):
        super().__init__()

        self._category_token: str = category_token
        self._mask: Dict[str, any] = mask
        self._sample_data_token: str = sample_data_token
        self.automatic_annotation: bool = automatic_annotation

    def to_dict(self):
        d = {
            "token": self.token,
            "category_token": self._category_token,
            "mask": self._mask,
            "sample_data_token": self._sample_data_token,
            "automatic_annotation": self.automatic_annotation,
        }
        return d


class SurfaceAnnTable(AbstractTable[SurfaceAnnRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#sample_annotationjson"""

    FILENAME = "surface_ann" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(
        self,
        category_token: str,
        mask: Dict[str, any],
        sample_data_token: str,
        automatic_annotation: bool,
    ):
        record = SurfaceAnnRecord(
            category_token=category_token,
            mask=mask,
            sample_data_token=sample_data_token,
            automatic_annotation=automatic_annotation,
        )
        return record

    @classmethod
    def from_json(cls, filepath: str) -> SurfaceAnnTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls()
        for item in items:
            record = SurfaceAnnRecord(
                category_token=item["category_token"],
                mask=item["mask"],
                sample_data_token=item["sample_data_token"],
                automatic_annotation=item["automatic_annotation"],
            )
            record.token = item["token"]
            table.set_record_to_table(record)

        return table
