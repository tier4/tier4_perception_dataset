from __future__ import annotations

from enum import Enum
import sys

if sys.version_info < (3, 11):

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


__all__ = ("SchemaName",)


class SchemaName(StrEnum):
    """Represents names of schema files."""

    CATEGORY = "category"
    ATTRIBUTE = "attribute"
    VISIBILITY = "visibility"
    INSTANCE = "instance"
    SENSOR = "sensor"
    CALIBRATED_SENSOR = "calibrated_sensor"
    EGO_POSE = "ego_pose"
    LOG = "log"
    SCENE = "scene"
    SAMPLE = "sample"
    SAMPLE_DATA = "sample_data"
    SAMPLE_ANNOTATION = "sample_annotation"
    MAP = "map"
    OBJECT_ANN = "object_ann"  # optional
    SURFACE_ANN = "surface_ann"  # optional
    KEYPOINT = "keypoint"  # optional

    def is_optional(self) -> bool:
        """Indicates if this schema name is optional.

        Returns:
            bool: Return True if this schema is optional.
        """
        return self in (SchemaName.OBJECT_ANN, SchemaName.SURFACE_ANN, SchemaName.KEYPOINT)
