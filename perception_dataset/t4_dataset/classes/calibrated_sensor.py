from typing import Any, Dict, List

import numpy as np

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class CalibratedSensorRecord(AbstractRecord):
    def __init__(
        self,
        sensor_token: str,
        translation: Dict[str, float],
        rotation: Dict[str, float],
        camera_intrinsic: List[List[float]],
        camera_distortion: Dict[str, float],
    ):
        super().__init__()

        assert {"x", "y", "z"} == set(translation.keys())
        assert {"w", "x", "y", "z"} == set(rotation.keys())
        assert len(camera_intrinsic) == 0 or np.array(camera_intrinsic).shape == (3, 3)

        self.sensor_token: str = sensor_token
        self.translation: Dict[str, float] = translation
        self.rotation: Dict[str, float] = rotation
        self.camera_intrinsic: List[List[float]] = camera_intrinsic
        self.camera_distortion: List[float] = camera_distortion

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token": self.token,
            "sensor_token": self.sensor_token,
            "translation": [
                self.translation["x"],
                self.translation["y"],
                self.translation["z"],
            ],
            "rotation": [
                self.rotation["w"],
                self.rotation["x"],
                self.rotation["y"],
                self.rotation["z"],
            ],
            "camera_intrinsic": self.camera_intrinsic,
            "camera_distortion": self.camera_distortion,
        }
        return d


class CalibratedSensorTable(AbstractTable[CalibratedSensorRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#calibrated_sensorjson"""

    FILENAME = "calibrated_sensor" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> CalibratedSensorRecord:
        return CalibratedSensorRecord(**kwargs)
