from typing import Any, Dict

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class InstanceRecord(AbstractRecord):
    def __init__(self, category_token: str, instance_name: str = ""):
        super().__init__()
        self._category_token: str = category_token
        self._instance_name: str = instance_name
        self._nbr_annotations: int = 0
        self._first_annotation_token: str = ""
        self._last_annotation_token: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token": self.token,
            "category_token": self._category_token,
            "instance_name": self._instance_name,
            "nbr_annotations": self._nbr_annotations,
            "first_annotation_token": self._first_annotation_token,
            "last_annotation_token": self._last_annotation_token,
        }
        return d

    def set_annotation_info(
        self, nbr_annotations: int, first_annotation_token: str, last_annotation_token: str
    ):
        self._nbr_annotations = nbr_annotations
        self._first_annotation_token = first_annotation_token
        self._last_annotation_token = last_annotation_token


class InstanceTable(AbstractTable):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#instancejson"""

    FILENAME = "instance" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()
        self._id_to_token: Dict[str, str] = {}

    def _to_record(self, category_token: str, instance_name: str = ""):
        return InstanceRecord(category_token=category_token, instance_name=instance_name)

    def get_token_from_id(self, instance_id: str, category_token: str, dataset_name: str) -> str:
        if instance_id in self._id_to_token and False:
            token = self._id_to_token[instance_id]
        else:
            token = self.insert_into_table(
                category_token=category_token, instance_name=dataset_name + "::" + str(instance_id)
            )
            self._id_to_token[instance_id] = token

        return token
