import base64
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

from PIL import Image
import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as cocomask
import skimage

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation
from perception_dataset.deepen.segmentation.preprocess_deepen_segmentation_annotations import (
    preprocess_deepen_segmentation_annotation,
)

"""
TODO(Shin-kyoto): !!!Tentative Implementation. Please remove it before merge!!!

Motivation to add this mode:

- When annotating with semantic paints, annotation data is not saved for each instance.Therefore, even if the same object appears in different frames, it will be a different instance. It will not be consistent between frames.
- However, if it is confirmed that there is only one instance per scene at the time of data collection, it is possible to make it consistent between frames by assuming that there is only one instance in the scene. can.
- If you have annotated with semantic paints, but there is only one object, and you want it to be consistent, please use ONE_INSTANCE_PER_SCENE.
- This workaround should be tentative, so don't merge it into main. Please use Polygons from now on.
"""
ONE_INSTANCE_PER_SCENE = True


class DeepenSegmentationPaints:
    # Class to handle Deepen segmentation paints data.

    def __init__(self, input_anno_file: str, input_base: str):
        # Preprocess the input annotation file
        input_anno_file_path = Path(input_anno_file)
        base_dir_path = Path(input_anno_file).parent
        input_base_path = Path(input_base)
        segmentation_dir_path = base_dir_path / input_anno_file_path.stem
        preprocess_deepen_segmentation_annotation(
            input_anno_file_path, base_dir_path, logging.getLogger(__name__)
        )

        self.index_to_category: List[str] = []  # ["category1", "category2"]
        self.dataset_id: str = segmentation_dir_path.stem
        self.segmentation_masks: Dict[Tuple[str, str], NDArray] = {}
        self.load_data(segmentation_dir_path, input_base_path)

    def load_data(self, segmentation_dir: Path, input_base: Path):
        """
        Loads metadata and segmentation masks from the directory.

        Args:
            segmentation_dir (Path): The directory containing segmentation data.
            input_base (Path): The base directory containing image data.
        """

        def _load_metadata(segmentation_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
            # Load metadata.json
            metadata_file = segmentation_dir / "metadata.json"
            if not metadata_file.is_file():
                raise FileNotFoundError(f"metadata.json not found in {segmentation_dir}")

            with open(metadata_file, "r") as f:
                metadata_dict = json.load(f)

            sensor_names: List[str] = list(metadata_dict.keys())
            # Assume the number of images in all sensors is the same
            image_names: List[str] = list(metadata_dict[sensor_names[0]].keys())
            # Assume the categories in all images and sensors is the same
            index_to_category: List[str] = metadata_dict[sensor_names[0]][image_names[0]]

            return metadata_dict, index_to_category

        def _load_segmentation_masks(
            metadata_dict: Dict[str, Any], segmentation_dir: Path, input_base: Path
        ) -> Dict[Tuple[str, str], NDArray]:
            segmentation_masks = {}

            # Assuming all images have the same dimensions
            height, width = _get_image_dimensions(input_base, metadata_dict)

            # Load segmentation masks and create index_to_category mapping
            for sensor_name, images in metadata_dict.items():
                sensor_dir = segmentation_dir / sensor_name
                for image_name in images.keys():
                    # Load segmentation mask
                    npy_file = sensor_dir / image_name.replace(".jpg", "_jpg.npy")
                    if not npy_file.is_file():
                        continue  # Skip if the .npy file doesn't exist

                    data = np.load(npy_file)
                    segmentation_mask: NDArray = data.reshape((height, width))
                    segmentation_masks[(sensor_name, image_name)] = segmentation_mask

            return segmentation_masks

        def _extract_actual_image_name(image_name_in_metadata: str) -> str:
            """
            Extracts the actual image name used in input_base from the image name in metadata.

            Args:
                image_name_in_metadata (str): The image name as found in metadata.

            Returns:
                str: The actual image name used in input_base.
            """
            # Assuming the image names in metadata have a prefix and we need to get the last part
            # For example, from 'data_CAM_TRAFFIC_LIGHT_NEAR_00000.jpg' extract '00000.jpg'

            # Use regular expression to match the numeric part followed by '.jpg' at the end
            match = re.search(r"(\d+\.jpg)$", image_name_in_metadata)
            if match:
                actual_image_name = match.group(1)
                return actual_image_name
            else:
                raise ValueError(f"Cannot extract actual image name from {image_name_in_metadata}")

        def _get_image_dimensions(
            input_base: str, metadata_dict: Dict[str, Any]
        ) -> Tuple[int, int]:
            """
            Retrieves image dimensions from the first image specified in metadata_dict.

            Args:
                input_base (str): The base directory containing image data.
                metadata_dict (Dict[str, Any]): The metadata dictionary containing sensor and image information.

            Returns:
                Tuple[int, int]: The width and height of the images.
            """
            for sensor_name in metadata_dict.keys():
                image_names = list(metadata_dict[sensor_name].keys())
                if image_names:
                    first_image_name = _extract_actual_image_name(image_names[0])
                    image_path = input_base / "data" / sensor_name / first_image_name
                    if image_path.is_file():
                        with Image.open(image_path) as img:
                            width, height = img.size
                        return width, height
                    else:
                        raise FileNotFoundError(f"Image file {image_path} not found.")
            raise ValueError("No images found in metadata_dict to retrieve dimensions.")

        metadata_dict, self.index_to_category = _load_metadata(segmentation_dir)

        self.segmentation_masks = _load_segmentation_masks(
            metadata_dict, segmentation_dir, input_base
        )

    def to_deepen_annotations(self) -> List[DeepenAnnotation]:
        # Converts segmentation data to Deepen annotation format.

        def _dummy_instance_id(index_to_category: List[str]) -> Dict[str, int]:
            instance_id = {}
            for category_name in index_to_category:
                instance_id[category_name] = 0

            return instance_id

        def _mask_to_rle(object_mask: NDArray) -> List[str]:
            labeled_mask = skimage.measure.label(object_mask, connectivity=1)
            num_instances = labeled_mask.max()

            rle_list = []
            for instance_id in range(1, num_instances + 1):
                instance_mask = (labeled_mask == instance_id).astype(np.uint8)
                rle = cocomask.encode(np.asfortranarray(instance_mask))
                rle = base64.b64encode(rle["counts"]).decode("ascii")
                rle_list.append(rle)
            return rle_list

        annotations = []
        dummy_instance_id: Dict[str, int] = _dummy_instance_id(self.index_to_category)
        for (sensor_name, image_name), segmentation_mask in self.segmentation_masks.items():

            for idx, category_name in enumerate(self.index_to_category):
                category_index = idx + 1  # Categories are indexed starting from 1

                # Create a binary mask for the specific category
                object_mask: NDArray = (segmentation_mask == category_index).astype(np.uint8)
                if object_mask.sum() == 0:
                    continue  # Skip if the category is not present

                # Convert from mask to RLE(Run-length encoding)
                rle_list = _mask_to_rle(object_mask)

                for rle in rle_list:
                    # Create a unique label_id
                    label_id = f"{category_name}:{dummy_instance_id[category_name]}"
                    dummy_instance_id[category_name] += 1

                    # TODO(Shin-kyoto): !!!Tentative Implementation. Please remove it before merge!!!
                    if ONE_INSTANCE_PER_SCENE:
                        dummy_instance_id[category_name] -= 1  # ALWAYS INSTANCE ID IS ZERO

                    # Create DeepenAnnotation instance
                    annotation = DeepenAnnotation(
                        dataset_id=self.dataset_id,
                        file_id=image_name,
                        label_category_id=category_name,
                        label_id=label_id,
                        label_type="2d_segmentation",
                        sensor_id=sensor_name,
                        attributes=None,
                        three_d_bbox=None,
                        two_d_box=None,
                        two_d_mask=rle,
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
