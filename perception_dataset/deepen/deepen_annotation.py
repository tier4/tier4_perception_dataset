from __future__ import annotations

from abc import ABC
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from numbers import Number
import re
from typing import Any, Dict, List, Optional, TypeVar

from typing_extensions import Self

__all__ = ["DeepenAnnotation", "DeepenAnnotationLike"]


class LabelType(str, Enum):
    BBOX_3D = "3d_bbox"
    BBOX_2D = "box"
    SEGMENTATION_2D = "2d_segmentation"


class LabelFormat(str, Enum):
    POLYGON = "polygon"
    PAINTING = "painting"


@dataclass
class LabelInfo:
    label_type: LabelType
    label_format: LabelFormat

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        return cls(**data)


@dataclass
class DeepenAnnotation(ABC):
    """
    Represents a single segmentation annotation in Deepen format.

    Attributes:
        dataset_id (str): The ID of the dataset.
        file_id (str): The file identifier, e.g., "0.pcd".
        label_category_id (str): The category of the label, e.g., "car".
        label_id (str): The unique identifier of the label, e.g., "car:1".
        label_type (str): The type of the label, e.g., "3d_bbox", "box", "2d_segmentation".
        sensor_id (str): The identifier of the sensor, e.g., "lidar", "camera1".
        labeller_email (str): The email of the labeller. Defaults to "default@tier4.jp".
        attributes (Optional[Dict[str, Any]]): Additional attributes of the annotation, e.g.,
            {
                "state": "moving",
                "occlusion": "none",
                "cycle_state": "with_rider"
            }.
        three_d_bbox (Optional[Dict[str, Any]]): The 3D bounding box data. e.g.,
            {
                "cx": ...,
                "cy": ...,
                "cz": ...,
                "h": ...,
                "l": ...,
                "w": ...,
                "quaternion": {"x": ..., "y": ..., "z": ..., "w": 0}
            }.
        box (Optional[List[float]]): The 2D bounding box data, e.g., [corner_x, corner_y, width, height].
        two_d_mask (Optional[str]): Run-length encoding (RLE) of the 2D mask.

    Note:
        Exactly one of `three_d_bbox`, `two_d_box`, or `two_d_mask` must be provided.
    """

    dataset_id: str
    file_id: str
    label_category_id: str
    label_id: str
    label_type: str
    sensor_id: str
    labeller_email: str = "default@tier4.jp"
    attributes: Optional[Dict[str, Any]] = field(default_factory=dict)
    three_d_bbox: Optional[Dict[str, Any]] = None
    box: Optional[List[float]] = None
    two_d_mask: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Validates the annotation data after initialization.
        Ensures that exactly one of three_d_bbox, two_d_box, or two_d_mask is provided
        and that the provided annotation contains the required data.
        """

        def _check_provided_annotations_exists() -> None:
            # Ensures that exactly one of three_d_bbox, two_d_box, or two_d_mask is provided.
            provided_annotations = (
                self.three_d_bbox is not None,
                self.box is not None,
                self.two_d_mask is not None,
            )
            assert any(
                provided_annotations
            ), "At least, one of three_d_bbox, two_d_box, or two_d_mask must be provided."

        def _check_label_id() -> None:
            """
            Checks if label_id follows the format '{label_category_id}:{int}'.
            Raises:
                ValueError: If label_id does not match the required format.
            """
            pattern = rf"^{re.escape(self.label_category_id)}:\d+$"
            if not re.match(pattern, self.label_id):
                raise ValueError(
                    f"label_id '{self.label_id}' must follow the format '{{label_category_id}}:{{int}}'"
                )

        def _check_three_d_bbox() -> None:
            """
            Checks if three_d_bbox contains the required keys:
            'cx', 'cy', 'cz', 'h', 'l', 'w', and 'quaternion' with keys 'x', 'y', 'z', 'w'.

            Raises:
                AssertionError: If any required keys are missing.
            """
            if self.three_d_bbox is None:
                return

            required_keys = ("cx", "cy", "cz", "h", "l", "w", "quaternion")
            missing_keys = [key for key in required_keys if key not in self.three_d_bbox]
            assert not missing_keys, f"three_d_bbox is missing keys: {missing_keys}"

            # Check quaternion
            quaternion = self.three_d_bbox["quaternion"]
            quaternion_keys = ("x", "y", "z", "w")
            missing_quaternion_keys = [key for key in quaternion_keys if key not in quaternion]
            assert (
                not missing_quaternion_keys
            ), f"quaternion is missing keys: {missing_quaternion_keys}"

        def _check_two_d_box() -> None:
            """
            Checks if two_d_box has four elements: [corner_x, corner_y, width, height].

            Raises:
                AssertionError: If the list does not have exactly four numerical elements.
            """
            if self.box is None:
                return

            assert isinstance(self.box, list), "two_d_box must be a list."
            assert len(self.box) == 4, "two_d_box must be a list of four elements."
            assert all(
                isinstance(value, Number) and value >= 0 for value in self.box
            ), "two_d_box elements must be numbers and greater than 0."

        def _check_two_d_mask() -> None:
            """
            Checks if two_d_mask is a non-empty string representing an RLE (Run-Length Encoding).
            Args:
                two_d_mask (str): The RLE string to validate.
            Raises:
                AssertionError: If two_d_mask is not a valid non-empty string.
            """
            if self.two_d_mask is None:
                return

            assert isinstance(
                self.two_d_mask["counts"], str
            ), "two_d_mask['counts'] must be a string."
            assert self.two_d_mask[
                "counts"
            ].strip(), "two_d_mask['counts'] must not be an empty string."

        # Ensures that exactly one of three_d_bbox, two_d_box, or two_d_mask is provided.
        _check_provided_annotations_exists()

        # Checks if label_id follows the format '{label_category_id}:{int}'
        _check_label_id()

        # Checks three_d_box keys
        _check_three_d_bbox()

        # Checks two_d_box keys
        _check_two_d_box()

        # Checks two_d_mask keys
        _check_two_d_mask()

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_file(
        cls,
        ann_file: str,
        *,
        as_dict: bool = True,
    ) -> List[DeepenAnnotationLike | Dict[str, Any]]:
        """Load annotations from file(s).

        Args:
            ann_file (str): Annotation file (.json).
            as_dict (bool, optional): Whether to output objects as dict or its instance.
                Defaults to True.

        Returns:
            List[DeepenAnnotationLike | Dict[str, Any]]: List of annotations or dicts.
        """
        with open(ann_file, "r") as f:
            data = json.load(f)

        labels: List[Dict[str, Any]] = data["labels"]

        output = []
        for label in labels:
            if as_dict:
                output.append(label)
            else:
                output.append(DeepenAnnotation(**label))
        return output


DeepenAnnotationLike = TypeVar("DeepenAnnotationLike", bound=DeepenAnnotation)
