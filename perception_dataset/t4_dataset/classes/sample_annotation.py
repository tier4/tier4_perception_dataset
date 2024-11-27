from __future__ import annotations

import json
from typing import Dict, List, Optional

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class SampleAnnotationRecord(AbstractRecord):
    def __init__(
        self,
        sample_token: str,
        instance_token: str,
        attribute_tokens: List[str],
        visibility_token: str,
        translation: Dict[str, float],
        velocity: Optional[Dict[str, float]],
        acceleration: Optional[Dict[str, float]],
        size: Dict[str, float],
        rotation: Dict[str, float],
        num_lidar_pts: int,
        num_radar_pts: int,
    ):
        super().__init__()

        assert {"x", "y", "z"} == set(translation.keys())
        assert {"width", "length", "height"} == set(size.keys())
        assert {"w", "x", "y", "z"} == set(rotation.keys())
        if velocity is not None:
            assert {"x", "y", "z"} == set(velocity.keys())
        if acceleration is not None:
            assert {"x", "y", "z"} == set(acceleration.keys())

        self._sample_token: str = sample_token
        self._instance_token: str = instance_token
        self._attribute_tokens: List[str] = attribute_tokens
        self._visibility_token: str = visibility_token
        self._translation: Dict[str, float] = translation
        self._velocity: Optional[Dict[str, float]] = velocity
        self._acceleration: Optional[Dict[str, float]] = acceleration
        self._size: Dict[str, float] = size
        self._rotation: Dict[str, float] = rotation
        self._num_lidar_pts: int = num_lidar_pts
        self._num_radar_pts: int = num_radar_pts
        self._next: str = ""
        self._prev: str = ""

    @property
    def next_token(self):
        return self._next

    @next_token.setter
    def next_token(self, value: str):
        self._next = value

    @property
    def prev_token(self):
        return self._prev

    @prev_token.setter
    def prev_token(self, value: str):
        self._prev = value

    def to_dict(self):
        d = {
            "token": self.token,
            "sample_token": self._sample_token,
            "instance_token": self._instance_token,
            "attribute_tokens": self._attribute_tokens,
            "visibility_token": self._visibility_token,
            "translation": [
                self._translation["x"],
                self._translation["y"],
                self._translation["z"],
            ],
            "velocity": (
                [
                    self._velocity["x"],
                    self._velocity["y"],
                    self._velocity["z"],
                ]
                if self._velocity is not None
                else self._velocity
            ),
            "acceleration": (
                [
                    self._acceleration["x"],
                    self._acceleration["y"],
                    self._acceleration["z"],
                ]
                if self._acceleration is not None
                else self._acceleration
            ),
            "size": [
                self._size["width"],
                self._size["length"],
                self._size["height"],
            ],
            "rotation": [
                self._rotation["w"],
                self._rotation["x"],
                self._rotation["y"],
                self._rotation["z"],
            ],
            "num_lidar_pts": self._num_lidar_pts,
            "num_radar_pts": self._num_radar_pts,
            "next": self._next,
            "prev": self._prev,
        }
        return d


class SampleAnnotationTable(AbstractTable[SampleAnnotationRecord]):
    """https://github.com/tier4/tier4_perception_dataset/blob/main/docs/t4_format_3d_detailed.md#sample_annotationjson"""

    FILENAME = "sample_annotation" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(
        self,
        sample_token: str,
        instance_token: str,
        attribute_tokens: str,
        visibility_token: str,
        translation: Dict[str, float],
        velocity: Optional[Dict[str, float]],
        acceleration: Optional[Dict[str, float]],
        size: Dict[str, float],
        rotation: Dict[str, float],
        num_lidar_pts: int,
        num_radar_pts: int,
    ):
        record = SampleAnnotationRecord(
            sample_token=sample_token,
            instance_token=instance_token,
            attribute_tokens=attribute_tokens,
            visibility_token=visibility_token,
            translation=translation,
            velocity=velocity,
            acceleration=acceleration,
            size=size,
            rotation=rotation,
            num_lidar_pts=num_lidar_pts,
            num_radar_pts=num_radar_pts,
        )
        return record

    @classmethod
    def from_json(cls, filepath: str) -> SampleAnnotationTable:
        with open(filepath) as f:
            items = json.load(f)

        table = cls()
        for item in items:
            record = SampleAnnotationRecord(
                sample_token=item["sample_token"],
                instance_token=item["instance_token"],
                attribute_tokens=item["attribute_tokens"],
                visibility_token=item["visibility_token"],
                translation={
                    "x": item["translation"][0],
                    "y": item["translation"][1],
                    "z": item["translation"][2],
                },
                velocity=(
                    {
                        "x": item["velocity"][0],
                        "y": item["velocity"][1],
                        "z": item["velocity"][2],
                    }
                    if item.get("velocity") is not None
                    else None
                ),
                acceleration=(
                    {
                        "x": item["acceleration"][0],
                        "y": item["acceleration"][1],
                        "z": item["acceleration"][2],
                    }
                    if item.get("acceleration") is not None
                    else None
                ),
                size={
                    "width": item["size"][0],
                    "length": item["size"][1],
                    "height": item["size"][2],
                },
                rotation={
                    "w": item["rotation"][0],
                    "x": item["rotation"][1],
                    "y": item["rotation"][2],
                    "z": item["rotation"][3],
                },
                num_lidar_pts=item["num_lidar_pts"],
                num_radar_pts=item["num_radar_pts"],
            )
            record.token = item["token"]
            table.set_record_to_table(record)

        return table
