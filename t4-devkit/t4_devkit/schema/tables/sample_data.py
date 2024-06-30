from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import TYPE_CHECKING, Any

from t4_devkit.common.io import load_json
from typing_extensions import Self

from .base import SchemaBase
from .registry import SCHEMAS
from ..name import SchemaName

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum

if TYPE_CHECKING:
    from .sensor import SensorChannel, SensorModality

__all__ = ("SampleData", "FileFormat")


class FileFormat(StrEnum):
    """An enum to represent file formats.

    Attributes:
    ----------
        JPG: JPG format for image data.
        PNG: PNG format for image data.
        PCD: PCD format for pointcloud data.
        BIN: BIN format.
        PCDBIN: PCD.BIN format for pointcloud data.
    """

    JPG = "jpg"
    PNG = "png"
    PCD = "pcd"
    BIN = "bin"
    PCDBIN = "pcd.bin"

    @staticmethod
    def is_member(item: str) -> bool:
        """Indicate whether the input item is the one of members of FileFormat.

        Args:
        ----
            item (str):

        Returns:
        -------
            bool: Return True if the item is included.
        """
        return item in FileFormat.values()

    @staticmethod
    def values() -> list[str]:
        """Return a list of values of members.

        Returns:
        -------
            list[str]: List of values.
        """
        return [v.value for v in FileFormat]

    def as_ext(self) -> str:
        """Return the value as file extension.

        Returns:
        -------
            str: File extension.
        """
        return f".{self.value}"


@dataclass
@SCHEMAS.register(SchemaName.SAMPLE_DATA)
class SampleData(SchemaBase):
    """A class to represent schema table of `sample_data.json`.

    Attributes:
    ----------
        token (str): Unique record identifier.
        sample_token (str): Foreign key pointing the sample.
        ego_pose_token (str): Foreign key pointing the ego_pose.
        calibrated_sensor_token (str): Foreign key pointing the calibrated_sensor.
        filename (str): Relative path to data-blob on disk.
        fileformat (FileFormat): Data file format.
        width (int): If the sample data is an image, this is the image width in [px].
        height (int): If the sample data is an image, this is the image height in [px].
        timestamp (int): Unix time stamp.
        is_key_frame (bool): True if sample_data is part of key frame else, False.
        next (str): Foreign key pointing the sample_data that follows this in time.
            Empty if end of scene.
        prev (str): Foreign key pointing the sample_data that precedes this in time.
            Empty if start of scene.
        is_valid (bool): True if this data is valid, else False. Invalid data should be ignored.

    Shortcuts:
    ---------
        modality (SensorModality): Sensor modality. This should be set after instantiated.
        channel (SensorChannel): Sensor channel. This should be set after instantiated.
    """

    token: str
    sample_token: str
    ego_pose_token: str
    calibrated_sensor_token: str
    filename: str
    fileformat: FileFormat
    width: int
    height: int
    timestamp: int
    is_key_frame: bool
    next: str  # noqa: A003
    prev: str
    is_valid: bool

    # shortcuts
    modality: SensorModality = field(init=False)
    channel: SensorChannel = field(init=False)

    @classmethod
    def from_json(cls, filepath: str) -> list[Self]:
        objs: list[Self] = []
        record_list: list[dict[str, Any]] = load_json(filepath)
        for record in record_list:
            token: str = record["token"]
            sample_token: str = record["sample_token"]
            ego_pose_token: str = record["ego_pose_token"]
            calibrated_sensor_token: str = record["calibrated_sensor_token"]
            filename: str = record["filename"]
            fileformat = FileFormat(record["fileformat"])
            width: int = record["width"]
            height: int = record["height"]
            timestamp: int = record["timestamp"]
            is_key_frame: bool = record["is_key_frame"]
            next_: str = record["next"]
            prev: str = record["prev"]
            is_valid: bool = record.get("is_valid", True)
            objs.append(
                cls(
                    token=token,
                    sample_token=sample_token,
                    ego_pose_token=ego_pose_token,
                    calibrated_sensor_token=calibrated_sensor_token,
                    filename=filename,
                    fileformat=fileformat,
                    width=width,
                    height=height,
                    timestamp=timestamp,
                    is_key_frame=is_key_frame,
                    next=next_,
                    prev=prev,
                    is_valid=is_valid,
                )
            )
        return objs
