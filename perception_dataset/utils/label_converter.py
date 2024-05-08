from abc import ABC
from typing import Dict, List, Union

import yaml

from perception_dataset.constants import LABEL_PATH_ENUM


class BaseConverter(ABC):
    def __init__(self, label_path: Union[str, LABEL_PATH_ENUM]) -> None:
        super().__init__()
        self.label_map: Dict[str, str] = self.__init_label_map(str(label_path))

    @staticmethod
    def __init_label_map(label_path: str) -> Dict[str, str]:
        with open(label_path, "r") as f:
            label_info: Dict[str, List[str]] = yaml.safe_load(f)

        label_map: Dict[str, str] = {}
        for key, names in label_info.items():
            label_map.update({name: key for name in names})
        return label_map

    def convert_label(self, label: str) -> str:
        return label
        return self.label_map[label]


class LabelConverter(BaseConverter):
    def __init__(
        self,
        label_path: Union[str, LABEL_PATH_ENUM] = LABEL_PATH_ENUM.OBJECT_LABEL,
        attribute_path: Union[str, LABEL_PATH_ENUM] = LABEL_PATH_ENUM.ATTRIBUTE,
    ) -> None:
        super().__init__(label_path)
        self.attribute_map: Dict[str, str] = self.__init_attribute_map(str(attribute_path))

    @staticmethod
    def __init_attribute_map(attribute_path: str) -> Dict[str, str]:
        with open(attribute_path, "r") as f:
            attribute_info: Dict[str, Dict[str, List[str]]] = yaml.safe_load(f)

        attribute_map: Dict[str, str] = {}
        for key, attributes in attribute_info.items():
            for attr, names in attributes.items():
                attribute_map.update({name: f"{key}.{attr}" for name in names})
        return attribute_map

    def convert_attribute(
        self,
        attribute: str,
    ) -> str:
        return attribute


class TrafficLightLabelConverter(BaseConverter):
    def __init__(
        self,
        label_path: Union[str, LABEL_PATH_ENUM] = LABEL_PATH_ENUM.TRAFFIC_LIGHT_LABEL,
    ) -> None:
        super().__init__(label_path)
