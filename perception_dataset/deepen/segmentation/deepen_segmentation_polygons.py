import base64
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

from PIL import Image
from pycocotools import mask as cocomask

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation


class DeepenSegmentationPolygons:
    """
    Class to handle Deepen segmentation polygon data.

    Attributes:
        annotations_data (Dict[str, Any]): The loaded data containing polygon annotations.
        input_base (Path): The base directory containing image data (if needed).
    """

    def __init__(
        self,
        input_anno_file: str,
        input_base: str,
        t4data_name_to_deepen_dataset_id: Dict[str, str],
        camera_name_to_index: Dict[str, int],
    ):
        """
        Initializes the DeepenSegmentationPolygons object and loads the data.

        Args:
            input_anno_file (str): Path to the JSON file containing polygon annotations.
            input_base (str): Path to the base directory containing image data.
        """
        self.annotations_data: Dict[str, Any] = {}
        self.load_data(Path(input_anno_file))
        deepen_dataset_id = self.annotations_data["labels"][0]["dataset_id"]
        t4dataset_name = ""
        for t4dataset_name_candidate in t4data_name_to_deepen_dataset_id.keys():
            if t4data_name_to_deepen_dataset_id[t4dataset_name_candidate] == deepen_dataset_id:
                t4dataset_name = t4dataset_name_candidate
        # !!!!Tentative implementation!!!!
        # self.sensor_id = list(camera_name_to_index.keys())[0]
        self.camera_index_to_name = {index: name for name, index in camera_name_to_index.items()}
        self.width, self.height = self._get_image_dimensions(Path(input_base) / t4dataset_name)

    def load_data(self, json_path: Path):
        """
        Loads annotation data from the JSON file.

        Args:
            json_path (Path): Path to the JSON file containing polygon annotations.


        Raises:
            FileNotFoundError: If the JSON file does not exist.
            JSONDecodeError: If the JSON file is not properly formatted.
        """
        if not json_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")

        with json_path.open("r") as f:
            self.annotations_data = json.load(f)

    def _get_image_dimensions(self, input_base: Path) -> Tuple[int, int]:
        # Get image dimensions from the first image
        labels = self.annotations_data.get("labels", [])
        if labels:
            first_label = labels[0]
            file_id = first_label["file_id"]
            sensor_id = first_label["sensor_id"]
            match = re.match(r"sensor(\d+)", sensor_id)
            sensor_number = int(match.group(1))
            image_path = input_base / "data" / self.camera_index_to_name[sensor_number] / file_id
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with Image.open(image_path) as img:
                width, height = img.size
        else:
            raise ValueError("No labels found in the annotation data.")

        return width, height

    def calculate_bbox_from_polygons(self, polygons: List[List[List[float]]]) -> List[float]:
        """
        Calculates the bounding box (BBox) from a set of polygons.
        polygons: A 3D list [ [ [x1, y1], [x2, y2], ...], ... ]
        return: BBox in the format [corner_x, corner_y, width, height]
        """
        # Extract all x and y coordinates
        all_x = [point[0] for polygon in polygons for point in polygon]
        all_y = [point[1] for polygon in polygons for point in polygon]

        # Get minimum and maximum x, y
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)

        # Calculate the bounding box: corner_x, corner_y, width, height
        corner_x = min_x
        corner_y = min_y
        width = max_x - min_x
        height = max_y - min_y

        return [corner_x, corner_y, width, height]

    def to_deepen_annotations(self) -> List[DeepenAnnotation]:
        """
        Converts the loaded data into a list of DeepenAnnotation instances.

        Returns:
            List[DeepenAnnotation]: A list of DeepenAnnotation instances.
        """
        annotations = []
        labels = self.annotations_data.get("labels", [])
        for label in labels:
            # Extract required fields with defaults where appropriate
            dataset_id = label["dataset_id"]
            file_id = label["file_id"]
            label_category_id = label["label_category_id"]
            label_id = label["label_id"]
            label_type = "2d_segmentation"
            sensor_id = label["sensor_id"]
            labeller_email = label["labeller_email"]
            attributes = label.get("attributes", {})

            # Extract polygons
            polygons = label["polygons"]
            bbox = self.calculate_bbox_from_polygons(polygons)

            flattened_polygons = []
            for polygon in polygons:
                # Create DeepenAnnotation instance

                flattened_polygon: List[List[float]] = [
                    coord for point in polygon for coord in point
                ]
                flattened_polygons.append(flattened_polygon)

            # Create RLE mask from polygons
            rle_objs = cocomask.frPyObjects(flattened_polygons, self.height, self.width)
            rle = cocomask.merge(rle_objs)
            # Encode the 'counts' to base64 string
            rle_counts_encoded = base64.b64encode(rle["counts"]).decode("ascii")
            # Replace the 'counts' with the encoded string
            rle["counts"] = rle_counts_encoded

            annotation = DeepenAnnotation(
                dataset_id=dataset_id,
                file_id=file_id,
                label_category_id=label_category_id,
                label_id=label_id,
                label_type=label_type,
                sensor_id=sensor_id,
                labeller_email=labeller_email,
                attributes=attributes,
                two_d_box=bbox,
                two_d_mask=rle,
            )
            annotations.append(annotation)

        return annotations

    def to_deepen_annotation_dicts(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Converts the loaded data into a list of DeepenAnnotation dicts.

        Returns:
            List[Dict[str, Any]]: A list of DeepenAnnotation dicts.
        """
        deepen_annotations: List[DeepenAnnotation] = self.to_deepen_annotations()
        deepen_annotation_dicts: List[Dict[str, Any]] = []
        for deepen_annotation in deepen_annotations:
            deepen_annotation_dicts.append(deepen_annotation.to_dict())

        return {"labels": deepen_annotation_dicts}
