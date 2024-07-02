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
    """An enum to represent schema filenames.

    Attributes:
        ATTRIBUTE: Property of an instance that can change while the category remains the same.
        CALIBRATED_SENSOR: Definition of a particular sensor as calibrated on a vehicle.
        CATEGORY: Object categories.
        EGO_POSE: Ego vehicle pose at at particular timestamp.
        INSTANCE: An object instance.
        LOG: Information about the log from which the data aws extracted.
        MAP: Map data that is stored as binary semantic masks from a top-down view.
        SAMPLE: A sample is an annotated keyframe at specific Hz.
        SAMPLE_ANNOTATION: A bounding box defining the position of an object seen in a sample.
        SAMPLE_DATA: A sensor data e.g. image, pointcloud or radar return.
        SCENE: A scene is a specific long sequence of consecutive frames extracted from a log.
        SENSOR: A specific sensor type.
        VISIBILITY: The visibility of instance is the fraction of annotation visible in all images.
        OBJECT_ANN (optional): The annotation of a foreground object in an image.
        SURFACE_ANN (optional): The annotation of a background object in an image.
        KEYPOINT (optional): The annotation of pose keypoints of an object in an image.
    """

    ATTRIBUTE = "attribute"
    CALIBRATED_SENSOR = "calibrated_sensor"
    CATEGORY = "category"
    EGO_POSE = "ego_pose"
    INSTANCE = "instance"
    LOG = "log"
    MAP = "map"
    SAMPLE = "sample"
    SAMPLE_ANNOTATION = "sample_annotation"
    SAMPLE_DATA = "sample_data"
    VISIBILITY = "visibility"
    SENSOR = "sensor"
    SCENE = "scene"
    OBJECT_ANN = "object_ann"  # optional
    SURFACE_ANN = "surface_ann"  # optional
    KEYPOINT = "keypoint"  # optional

    @property
    def filename(self) -> str:
        """Return the annotation json filename.

        Returns:
            Annotation json filename.
        """
        return f"{self.value}.json"

    def is_optional(self) -> bool:
        """Indicates if this schema name is optional.

        Returns:
            Return True if this schema is optional.
        """
        return self in (SchemaName.OBJECT_ANN, SchemaName.SURFACE_ANN, SchemaName.KEYPOINT)
