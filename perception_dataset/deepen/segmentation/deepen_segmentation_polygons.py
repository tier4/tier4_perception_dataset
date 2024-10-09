import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image
import base64
from pycocotools import mask as cocomask
from perception_dataset.deepen.deepen_annotation import DeepenAnnotation

class DeepenSegmentationPolygons:
    """
    Class to handle Deepen segmentation polygon data.

    Attributes:
        annotations_data (Dict[str, Any]): The loaded data containing polygon annotations.
        input_base (Path): The base directory containing image data (if needed).
    """

    def __init__(self, input_anno_file: str, input_base: str):
        """
        Initializes the DeepenSegmentationPolygons object and loads the data.

        Args:
            input_anno_file (str): Path to the JSON file containing polygon annotations.
            input_base (str): Path to the base directory containing image data.
        """
        self.annotations_data: Dict[str, Any] = {}
        self.load_data(Path(input_anno_file))
        self.width, self.height = self._get_image_dimensions(Path(input_base))

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

        with json_path.open('r') as f:
            self.annotations_data = json.load(f)
        
    def _get_image_dimensions(self, input_base: Path) -> Tuple[int, int]:
        # Get image dimensions from the first image
        labels = self.annotations_data.get("labels", [])
        if labels:
            first_label = labels[0]
            file_id = first_label["file_id"]
            sensor_id = first_label["sensor_id"]
            image_path = input_base / 'data' / sensor_id / file_id
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with Image.open(image_path) as img:
                width, height = img.size
        else:
            raise ValueError("No labels found in the annotation data.")
        
        return width, height

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
            dataset_id = label["dataset_name"]
            file_id = label["file_id"]
            label_category_id = label["label_category_id"]
            label_id = label["label_id"]
            label_type = label["label_type"]
            sensor_id = label["sensor_id"]
            labeller_email = label["labeller_email"]
            attributes = label.get("attributes", {})

            # Extract polygons
            polygons = label["polygons"]
            for polygon in polygons:
                # Create DeepenAnnotation instance

                flattened_polygon: List[List[float]] = [[coord for point in polygon for coord in point]]
                # Create RLE mask from polygons
                rle_objs = cocomask.frPyObjects(flattened_polygon, self.height, self.width)
                rle = cocomask.merge(rle_objs)
                # Encode the 'counts' to base64 string
                rle_counts_encoded = base64.b64encode(rle['counts']).decode('ascii')
                # Replace the 'counts' with the encoded string
                rle['counts'] = rle_counts_encoded

                annotation = DeepenAnnotation(
                    dataset_id=dataset_id,
                    file_id=file_id,
                    label_category_id=label_category_id,
                    label_id=label_id,
                    label_type=label_type,
                    sensor_id=sensor_id,
                    labeller_email=labeller_email,
                    attributes=attributes,
                    two_d_mask=rle['counts'],
                )
            annotations.append(annotation)

        return annotations
    
    def to_deepen_annotation_dicts(self) -> List[Dict[str, Any]]:
        """
        Converts the loaded data into a list of DeepenAnnotation dicts.

        Returns:
            List[Dict[str, Any]]: A list of DeepenAnnotation dicts.
        """
        deepen_annotations: List[DeepenAnnotation] = self.to_deepen_annotations()
        deepen_annotation_dicts: List[Dict[str, Any]] = []
        for deepen_annotation in deepen_annotations:
            deepen_annotation_dicts.append(deepen_annotation.to_dict())

        return deepen_annotation_dicts
