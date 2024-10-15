from typing import Any, Dict, Optional

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class EgoPoseRecord(AbstractRecord):
    def __init__(
        self,
        translation: Dict[str, float],
        rotation: Dict[str, float],
        timestamp: int,
        twist: Optional[Dict[str, float]],
        acceleration: Optional[Dict[str, float]],
        geocoordinate: Optional[Dict[str, float]],
    ):
        super().__init__()

        assert {"x", "y", "z"} == set(translation.keys())
        assert {"w", "x", "y", "z"} == set(rotation.keys())

        if twist is not None:
            assert {"vx", "vy", "vz", "yaw_rate", "pitch_rate", "roll_rate"} == set(twist.keys())
        if acceleration is not None:
            assert {"ax", "ay", "az"} == set(acceleration.keys())
        if geocoordinate is not None:
            assert {"latitude", "longitude", "altitude"} == set(geocoordinate.keys())

        self.translation: Dict[str, float] = translation
        self.rotation: Dict[str, float] = rotation
        self.timestamp: int = timestamp
        self.twist: Optional[Dict[str, float]] = twist
        self.acceleration: Optional[Dict[str, float]] = acceleration
        self.geocoordinate = geocoordinate

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token": self.token,
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
            "timestamp": self.timestamp,
            "twist": (
                [
                    self.twist["vx"],
                    self.twist["vy"],
                    self.twist["vz"],
                    self.twist["yaw_rate"],
                    self.twist["pitch_rate"],
                    self.twist["roll_rate"],
                ]
                if self.twist is not None
                else None
            ),
            "acceleration": (
                [
                    self.acceleration["ax"],
                    self.acceleration["ay"],
                    self.acceleration["az"],
                ]
                if self.acceleration is not None
                else None
            ),
            "geocoordinate": (
                [
                    self.geocoordinate["latitude"],
                    self.geocoordinate["longitude"],
                    self.geocoordinate["altitude"],
                ]
                if self.geocoordinate is not None
                else None
            ),
        }
        return d


class EgoPoseTable(AbstractTable[EgoPoseRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#ego_posejson"""

    FILENAME = "ego_pose" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> EgoPoseRecord:
        return EgoPoseRecord(**kwargs)
