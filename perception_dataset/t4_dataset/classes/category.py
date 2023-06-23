from __future__ import annotations

import json
from typing import Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class CategoryRecord(AbstractRecord):
    def __init__(self, name: str, description: str, index: int):
        super().__init__()
        self.name: str = name
        self.description: str = description
        self.index: int = index

    def to_dict(self) -> Dict[str, str]:
        d = {
            "token": self.token,
            "name": self.name,
            "description": self.description,
            "index": self.index,
        }
        return d


class CategoryTable(AbstractTable[CategoryRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#categoryjson"""

    FILENAME = "category" + EXTENSION_ENUM.JSON.value

    def __init__(self, name_to_description: Dict[str, str], default_value: str):
        super().__init__()
        self._name_to_token: Dict[str, str] = {}
        self._name_to_description: Dict[str, str] = name_to_description
        self._description_default_value: str = default_value
        self._index = 1  # Index starts from 1 where 0 reserved for unpainted labels

    def _to_record(self, name: str, description: str):
        record = CategoryRecord(name=name, description=description, index=self._index)
        # Index increment by one
        self._index += 1
        return record

    def get_token_from_name(self, name: str) -> str:
        if name in self._name_to_token:
            token = self._name_to_token[name]
        else:
            description = self._name_to_description.get(name, self._description_default_value)
            token = self.insert_into_table(name=name, description=description)
            self._name_to_token[name] = token

        return token

    def get_index_from_token(self, token: str) -> int:
        """Retrieve index from a token."""
        record = self.select_record_from_token(token=token)
        assert "index" in record, "Index doesn't find in the category record!"
        return record["index"]

    @classmethod
    def from_json(
        cls,
        filepath: str,
        name_to_description: Dict[str, str],
        default_value: str,
    ) -> CategoryTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls(name_to_description=name_to_description, default_value=default_value)
        index_counter = 1  # Index starts from 1 where 0 reserved for unpainted labels
        for item in items:
            index = item.get(item["index"], index_counter)
            record = CategoryRecord(
                name=item["name"], description=item["description"], index=index
            )

            record.token = item["token"]
            table.set_record_to_table(record)
            index_counter += 1

        return table
