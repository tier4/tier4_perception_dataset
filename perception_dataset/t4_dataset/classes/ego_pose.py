from __future__ import annotations

import json
from typing import Any, Dict, Optional

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class EgoPoseRecord(AbstractRecord):
    def __init__(
        self,
        translation: Dict[str, float],
        rotation: Dict[str, float],
        timestamp: int,
        twist: Optional[Dict[str, float]] = None,
        acceleration: Optional[Dict[str, float]] = None,
        geocoordinate: Optional[Dict[str, float]] = None,
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

    @classmethod
    def from_json(cls, filepath: str) -> EgoPoseTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls()
        for item in items:
            record = EgoPoseRecord(
                translation={
                    "x": item["translation"][0],
                    "y": item["translation"][1],
                    "z": item["translation"][2],
                },
                rotation={
                    "w": item["rotation"][0],
                    "x": item["rotation"][1],
                    "y": item["rotation"][2],
                    "z": item["rotation"][3],
                },
                timestamp=item["timestamp"],
                twist=(
                    {
                        "vx": item["twist"][0],
                        "vy": item["twist"][1],
                        "vz": item["twist"][2],
                        "yaw_rate": item["twist"][3],
                        "pitch_rate": item["twist"][4],
                        "roll_rate": item["twist"][5],
                    }
                    if item.get("twist") is not None
                    else None
                ),
                acceleration=(
                    {
                        "ax": item["acceleration"][0],
                        "ay": item["acceleration"][1],
                        "az": item["acceleration"][2],
                    }
                    if item.get("acceleration") is not None
                    else None
                ),
                geocoordinate=(
                    {
                        "latitude": item["geocoordinate"][0],
                        "longitude": item["geocoordinate"][1],
                        "altitude": item["geocoordinate"][2],
                    }
                    if item.get("geocoordinate") is not None
                    else None
                ),
            )
            record.token = item["token"]
            table.set_record_to_table(record)

        return table
