from dataclasses import asdict, dataclass, field
from numbers import Number
import re
from typing import Any, Dict, List, Optional, Tuple

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
                self.two_d_box is not None,
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
            if self.two_d_box is None:
                return

            assert isinstance(self.two_d_box, list), "two_d_box must be a list."
            assert len(self.two_d_box) == 4, "two_d_box must be a list of four elements."
            assert all(
                isinstance(value, Number) and value >= 0 for value in self.two_d_box
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
        # Required keys
        required = (
            "dataset_id",
            "file_id",
            "label_category_id",
            "label_id",
            "label_type",
            "sensor_id",
        )

        # Ensures all required keys are included.
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing required fields in data: {missing}")

        return cls(**data)

    def is_ignorable(self, *, ignore_interpolate_label: bool = False) -> bool:
        ignore = True

        # check whether to ignore interpolated label
        ignore &= ignore_interpolate_label and self.labeller_email == "auto_interpolation"

        return ignore

    def _format_attributes(self) -> Dict[str, Any]:
        attribute_names = [f"{name.lower()}.{state}" for name, state in self.attributes.items()]

        if "Occlusion_State" in self.attributes:
            visibility_name = _convert_occlusion_to_visibility(self.attributes["Occlusion_State"])
        elif "occlusion_state" in self.attributes:
            visibility_name = _convert_occlusion_to_visibility(self.attributes["occlusion_state"])
        else:
            visibility_name = "Not available"

        return {"attribute_names": attribute_names, "visibility_name": visibility_name}

    def format_annotation(self, camera_id: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        filename = self.file_id.split(".")[0]
        file_id = int(re.sub(r"\D", "", filename[-6:]))

        output: Dict[str, Any] = {
            "category_name": self.label_category_id,
            "instance_id": self.label_id,
        }

        # format attributes
        output.update(self._format_attributes())

        # format 3D box annotation
        if self.three_d_bbox is not None:
            output.update(
                {
                    "three_d_bbox": {
                        "translation": {
                            "x": self.three_d_bbox["cx"],
                            "y": self.three_d_bbox["cy"],
                            "z": self.three_d_bbox["cz"],
                        },
                        "velocity": None,
                        "acceleration": None,
                        "size": {
                            "width": self.three_d_bbox["w"],
                            "length": self.three_d_bbox["l"],
                            "height": self.three_d_bbox["h"],
                        },
                        "rotation": {
                            "w": self.three_d_bbox["quaternion"]["w"],
                            "x": self.three_d_bbox["quaternion"]["x"],
                            "y": self.three_d_bbox["quaternion"]["y"],
                            "z": self.three_d_bbox["quaternion"]["z"],
                        },
                    },
                    "num_lidar_pts": 0,
                    "num_radar_pts": 0,
                }
            )

        # format 2D box annotation
        if self.two_d_box is not None:
            output.update(
                {
                    "two_d_box": [
                        self.two_d_box[0],
                        self.two_d_box[1],
                        self.two_d_box[2],
                        self.two_d_box[3],
                    ],
                },
            )

        # format 2D segmentation annotation
        if self.two_d_mask is not None:
            output.update({"two_d_segmentation": self.two_d_mask})

        # format sensor ID for camera
        if self.two_d_box is not None or self.two_d_mask is not None:
            # TODO(ktro2828): Do not use `self.sensor_id[-1]`
            sensor_id = self.sensor_id[-1] if camera_id is None else camera_id
            output.update({"sensor_id": sensor_id})

        return file_id, output


def _convert_occlusion_to_visibility(self, name: str) -> str:
    if name == "full":
        return "none"
    elif name == "partial":
        return "most"
    elif name == "most":
        return "partial"
    else:
        return "full"
