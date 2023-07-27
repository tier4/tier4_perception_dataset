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
        velocity: Dict[str, Optional[float]],
        acceleration: Dict[str, Optional[float]],
        size: Dict[str, float],
        rotation: Dict[str, float],
        num_lidar_pts: int,
        num_radar_pts: int,
    ):
        super().__init__()

        assert {"x", "y", "z"} == set(translation.keys())
        assert {"width", "length", "height"} == set(size.keys())
        assert {"w", "x", "y", "z"} == set(rotation.keys())

        self._sample_token: str = sample_token
        self._instance_token: str = instance_token
        self._attribute_tokens: List[str] = attribute_tokens
        self._visibility_token: str = visibility_token
        self._translation: Dict[str, float] = translation
        self._velocity: Dict[str, Optional[float]] = velocity
        self._acceleration: Dict[str, Optional[float]] = acceleration
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
            "velocity": [
                self._velocity["x"],
                self._velocity["y"],
                self._velocity["z"],
            ],
            "acceleration": [
                self._acceleration["x"],
                self._acceleration["y"],
                self._acceleration["z"],
            ],
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


class SampleAnnotationTable(AbstractTable):
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
        velocity: Dict[str, Optional[float]],
        acceleration: Dict[str, Optional[float]],
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
