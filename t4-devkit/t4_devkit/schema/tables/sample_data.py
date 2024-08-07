from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import TYPE_CHECKING, Any

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
    from .sensor import SensorModality

__all__ = ("SampleData", "FileFormat")


class FileFormat(StrEnum):
    """An enum to represent file formats.

    Attributes:
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
            item (str): Any file format name.

        Returns:
            Return True if the item is included.
        """
        return item in FileFormat.values()

    @staticmethod
    def values() -> list[str]:
        """Return a list of values of members.

        Returns:
            List of values.
        """
        return [v.value for v in FileFormat]

    def as_ext(self) -> str:
        """Return the value as file extension.

        Returns:
            File extension.
        """
        return f".{self.value}"


@dataclass
@SCHEMAS.register(SchemaName.SAMPLE_DATA)
class SampleData(SchemaBase):
    """A class to represent schema table of `sample_data.json`.

    Attributes:
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
        channel (str): Sensor channel. This should be set after instantiated.
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
    channel: str = field(init=False)

    @staticmethod
    def shortcuts() -> tuple[str, str]:
        return ("modality", "channel")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        token: str = data["token"]
        sample_token: str = data["sample_token"]
        ego_pose_token: str = data["ego_pose_token"]
        calibrated_sensor_token: str = data["calibrated_sensor_token"]
        filename: str = data["filename"]
        fileformat = FileFormat(data["fileformat"])
        width: int = data["width"]
        height: int = data["height"]
        timestamp: int = data["timestamp"]
        is_key_frame: bool = data["is_key_frame"]
        next_: str = data["next"]
        prev: str = data["prev"]
        is_valid: bool = data.get("is_valid", True)

        return cls(
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
