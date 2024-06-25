from dataclasses import dataclass
import sys
from typing import Any

from t4_devkit.common import load_json
from typing_extensions import Self

from .base import SchemaBase

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum

__all__ = ("SampleData", "FileFormat")


class FileFormat(StrEnum):
    JPG = "jpg"
    PNG = "png"
    PCD = "pcd"
    BIN = "bin"
    PCDBIN = "pcd.bin"

    @staticmethod
    def is_member(item: str) -> bool:
        """Indicate whether the input item is the one of members of FileFormat.

        Args:
            item (str):

        Returns:
            bool: Return True if the item is included.
        """
        return item in FileFormat.values()

    @staticmethod
    def values() -> list[str]:
        """Return a list of values of members.

        Returns:
            list[str]: List of values.
        """
        return [v.value for v in FileFormat]

    def as_ext(self) -> str:
        """Return the value as file extension.

        Returns:
            str: File extension.
        """
        return f".{self.value}"


@dataclass(frozen=True)
class SampleData(SchemaBase):
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

    @classmethod
    def from_json(cls, filepath: str) -> Self:
        record: dict[str, Any] = load_json(filepath)
        token: str = record["token"]
        sample_token: str = record["sample_token"]
        ego_pose_token: str = record["ego_pose_token"]
        calibrated_sensor_token: str = record["calibrated_sensor_token"]
        filename: str = record["filename"]
        fileformat = FileFormat(record["fileformat"])
        width: int = record["width"]
        height: int = record["heght"]
        timestamp: int = record["timestamp"]
        is_key_frame: bool = record["is_key_frame"]
        next_: str = record["next"]
        prev: str = record["prev"]
        is_valid: bool = record["is_valid"]
        return cls(
            token=token,
            sample_token=sample_token,
            ego_pose_token=ego_pose_token,
            calibrated_sensor_token=calibrated_sensor_token,
            filename=filename,
            fileformat=fileformat,
            widht=width,
            height=height,
            timestamp=timestamp,
            is_key_frame=is_key_frame,
            next=next_,
            prev=prev,
            is_valid=is_valid,
        )
