from typing import Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class AttributeRecord(AbstractRecord):
    def __init__(self, name: str, description: str):
        super().__init__()
        self.name: str = name
        self.description: str = description

    def to_dict(self) -> Dict[str, str]:
        d = {
            "token": self.token,
            "name": self.name,
            "description": self.description,
        }
        return d


class AttributeTable(AbstractTable[AttributeRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#attributejson"""

    FILENAME = "attribute" + EXTENSION_ENUM.JSON.value

    def __init__(self, name_to_description: Dict[str, str], default_value: str):
        super().__init__()
        self._name_to_token: Dict[str, str] = {}
        self._name_to_description: Dict[str, str] = name_to_description
        self._description_default_value: str = default_value

    def _to_record(self, name: str, description: str) -> AttributeRecord:
        record = AttributeRecord(
            name=name,
            description=description,
        )
        return record

    def get_token_from_name(self, name: str) -> str:
        if name in self._name_to_token:
            token = self._name_to_token[name]
        else:
            description = self._name_to_description.get(name, self._description_default_value)
            token = self.insert_into_table(name=name, description=description)
            self._name_to_token[name] = token

        return token
