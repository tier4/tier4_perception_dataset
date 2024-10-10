from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from typing_extensions import Self


@dataclass
class DeepenAnnotation:
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

        two_d_box (Optional[List[float]]): The 2D bounding box data, e.g., [corner_x, corner_y, width, height].

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
    two_d_box: Optional[List[float]] = None
    two_d_mask: Optional[str] = None

    def __post_init__(self):
        """
        Validates the annotation data after initialization.
        Ensures that exactly one of three_d_bbox, two_d_box, or two_d_mask is provided
        and that the provided annotation contains the required data.
        """

        def _check_provided_annotations_exists(
            three_d_bbox: Dict[str, Any], two_d_box: List[float], two_d_mask: str
        ):
            # Ensures that exactly one of three_d_bbox, two_d_box, or two_d_mask is provided.
            provided_annotations = [
                three_d_bbox is not None,
                two_d_box is not None,
                two_d_mask is not None,
            ]
            num_provided = sum(provided_annotations)
            assert (
                num_provided == 1
            ), "Exactly one of three_d_bbox, two_d_box, or two_d_mask must be provided."

        def _check_label_id(label_category_id: str, label_id: str):
            """
            Checks if label_id follows the format '{label_category_id}:{int}'.

            Raises:
                ValueError: If label_id does not match the required format.
            """
            pattern = rf"^{re.escape(label_category_id)}:\d+$"
            if not re.match(pattern, label_id):
                raise ValueError(
                    f"label_id '{label_id}' must follow the format '{{label_category_id}}:{{int}}'"
                )

        def _check_three_d_bbox(three_d_bbox: Dict[str, Any]):
            """
            Checks if three_d_bbox contains the required keys:
            'cx', 'cy', 'cz', 'h', 'l', 'w', and 'quaternion' with keys 'x', 'y', 'z', 'w'.

            Args:
                three_d_bbox (Dict[str, Any]): The 3D bounding box data to validate.

            Raises:
                AssertionError: If any required keys are missing.
            """
            required_keys = ["cx", "cy", "cz", "h", "l", "w", "quaternion"]
            missing_keys = [key for key in required_keys if key not in three_d_bbox]
            assert not missing_keys, f"three_d_bbox is missing keys: {missing_keys}"

            # Check quaternion
            quaternion = three_d_bbox["quaternion"]
            quaternion_keys = ["x", "y", "z", "w"]
            missing_quaternion_keys = [key for key in quaternion_keys if key not in quaternion]
            assert (
                not missing_quaternion_keys
            ), f"quaternion is missing keys: {missing_quaternion_keys}"

        def _check_two_d_box(two_d_box: List[float]):
            """
            Checks if two_d_box has four elements: [corner_x, corner_y, width, height].

            Args:
                two_d_box (List[float]): The 2D bounding box data to validate.

            Raises:
                AssertionError: If the list does not have exactly four numerical elements.
            """
            assert isinstance(two_d_box, list), "two_d_box must be a list."
            assert len(two_d_box) == 4, "two_d_box must be a list of four elements."
            for value in two_d_box:
                assert isinstance(value, (int, float)), "two_d_box elements must be numbers."

        def _check_two_d_mask(two_d_mask: str):
            """
            Checks if two_d_mask is a non-empty string representing an RLE (Run-Length Encoding).

            Args:
                two_d_mask (str): The RLE string to validate.

            Raises:
                AssertionError: If two_d_mask is not a valid non-empty string.
            """
            assert isinstance(two_d_mask, str), "two_d_mask must be a string."
            assert two_d_mask.strip(), "two_d_mask must not be an empty string."

        # Ensures that exactly one of three_d_bbox, two_d_box, or two_d_mask is provided.
        _check_provided_annotations_exists(self.three_d_bbox, self.two_d_box, self.two_d_mask)
        # Checks if label_id follows the format '{label_category_id}:{int}'
        _check_label_id(self.label_category_id, self.label_id)

        if self.three_d_bbox is not None:
            _check_three_d_bbox(self.three_d_bbox)
        elif self.two_d_box is not None:
            _check_two_d_box(self.two_d_box)
        elif self.two_d_mask is not None:
            _check_two_d_mask(self.two_d_mask)

    def to_dict(self) -> Dict[str, Any]:
        # Converts the dataclass instance to a dictionary.
        return {
            "dataset_id": self.dataset_id,
            "file_id": self.file_id,
            "label_category_id": self.label_category_id,
            "label_id": self.label_id,
            "label_type": self.label_type,
            "labeller_email": self.labeller_email,
            "sensor_id": self.sensor_id,
            "attributes": self.attributes,
            "three_d_bbox": self.three_d_bbox,
            "box": self.two_d_box,
            "two_d_mask": self.two_d_mask,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Creates a DeepenAnnotation instance from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary containing the annotation data.

        Returns:
            DeepenAnnotation: An instance of the DeepenAnnotation class.

        Raises:
            KeyError: If required keys are missing in the input dictionary.
            AssertionError: If validation fails in __post_init__.
        """
        # Required fields
        required_fields = [
            "dataset_id",
            "file_id",
            "label_category_id",
            "label_id",
            "label_type",
            "sensor_id",
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise KeyError(f"Missing required fields in data: {missing_fields}")

        dataset_id = data["dataset_id"]
        file_id = data["file_id"]
        label_category_id = data["label_category_id"]
        label_id = data["label_id"]
        label_type = data["label_type"]
        sensor_id = data["sensor_id"]
        labeller_email = data.get("labeller_email", "default@tier4.jp")
        attributes = data.get("attributes", {})
        three_d_bbox = data.get("three_d_bbox")
        two_d_box = data.get("box")
        two_d_mask = data.get("two_d_mask")

        # Create and return the instance
        return cls(
            dataset_id=dataset_id,
            file_id=file_id,
            label_category_id=label_category_id,
            label_id=label_id,
            label_type=label_type,
            sensor_id=sensor_id,
            labeller_email=labeller_email,
            attributes=attributes,
            three_d_bbox=three_d_bbox,
            two_d_box=two_d_box,
            two_d_mask=two_d_mask,
        )
