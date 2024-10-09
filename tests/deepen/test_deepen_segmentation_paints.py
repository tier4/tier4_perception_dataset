import json
from pathlib import Path
from typing import List
from zipfile import ZipFile

from PIL import Image
import numpy as np
import pytest

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation
from perception_dataset.deepen.segmentation.deepen_segmentation_paints import (
    DeepenSegmentationPaints,
)


@pytest.fixture
def input_anno_file(tmp_path: Path) -> str:
    """
    Pytest fixture to set up a temporary test environment by creating a zip file containing simulated segmentation annotation data.

    This fixture performs the following steps:
    - Creates a temporary directory structure under 'base_dir/tmp/deepenLabels-*'.
    - Generates a 'Semantic Segmentation - metadata.json' file with sample metadata.
    - Creates dummy .npy segmentation mask files with predefined patterns.
    - Compresses the data directory into a zip file.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Returns:
        str: The path to the created zip file containing the simulated segmentation data.
    """
    test_dir = tmp_path
    base_dir = test_dir / "base_dir"
    base_dir.mkdir()

    # Create the directory structure inside base_dir/tmp/deepenLabels-*
    data_dir = base_dir / "tmp" / "deepenLabels-c0gc9xbz-DFE7eUiuaocC2WFQ58FjjA7s-1726015539"
    data_dir.mkdir(parents=True)

    # Create 'Semantic Segmentation - metadata.json'
    metadata = {
        "CAM_TRAFFIC_LIGHT_NEAR": {
            "data_CAM_TRAFFIC_LIGHT_NEAR_00000.jpg": ["category1", "category2"],
            "data_CAM_TRAFFIC_LIGHT_NEAR_00001.jpg": ["category1", "category2"],
        },
        "CAM_TRAFFIC_LIGHT_FAR": {
            "data_CAM_TRAFFIC_LIGHT_FAR_00000.jpg": ["category1", "category2"],
            "data_CAM_TRAFFIC_LIGHT_FAR_00001.jpg": ["category1", "category2"],
        },
    }
    metadata_file = data_dir / "Semantic Segmentation - metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    # Create npy files in data_dir
    # For simplicity, we'll just create empty npy files
    width: int = 640
    height: int = 480
    npy_filenames = [
        "Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy",
        "Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy",
        "Semantic Segmentation - sensor2 - data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy",
        "Semantic Segmentation - sensor2 - data_CAM_TRAFFIC_LIGHT_FAR_00001_jpg.npy",
    ]
    for npy_filename in npy_filenames:
        npy_file = data_dir / npy_filename
        # Create dummy segmentation data
        segmentation_mask = np.zeros((width * height,), dtype=np.uint8)
        segmentation_mask[: int(width * height * 0.5)] = 1  # Half the mask is category1
        segmentation_mask[int(width * height * 0.5) :] = 2  # Half the mask is

        segmentation_mask = np.zeros((width, height), dtype=np.uint8)
        third_height = height // 3
        segmentation_mask[:third_height, :] = 1  # Top third is category 1
        segmentation_mask[third_height : 2 * third_height, :] = 2  # Middle third is category 2
        segmentation_mask[2 * third_height :, :] = 1  # Bottom third is category 1
        # Flatten the segmentation mask to save as 1D array
        segmentation_mask = segmentation_mask.flatten()

        np.save(npy_file, segmentation_mask)

    # Create a zip file of data_dir
    zip_file_path = base_dir / "segmentation_prd_uuid_1970-01-01_00-00-00_00-01-00.zip"
    with ZipFile(zip_file_path, "w") as zipf:
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                # Calculate the relative path for the archive name
                arcname = file_path.relative_to(base_dir)
                zipf.write(file_path, arcname=arcname)

    # Return the path to the zip file
    return str(zip_file_path)


@pytest.fixture
def input_base(tmp_path: Path) -> str:
    """
    Pytest fixture to set up the 'input_base' directory with a simulated data structure containing dummy image files.

    This fixture performs the following steps:
    - Creates an 'input_base/data' directory structure.
    - For each camera ('CAM_TRAFFIC_LIGHT_NEAR', 'CAM_TRAFFIC_LIGHT_FAR'), creates a subdirectory.
    - Generates dummy image files ('00000.jpg', '00001.jpg') in each camera directory with specified dimensions.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Returns:
        str: The path to the 'input_base' directory containing the simulated image data.
    """
    input_base = tmp_path / "input_base"
    data_dir = input_base / "data"
    data_dir.mkdir(parents=True)

    # Create directories and image files
    cameras = ["CAM_TRAFFIC_LIGHT_NEAR", "CAM_TRAFFIC_LIGHT_FAR"]
    images = ["00000.jpg", "00001.jpg"]
    width: int = 640
    height: int = 480

    for camera in cameras:
        camera_dir = data_dir / camera
        camera_dir.mkdir(parents=True)
        for image_name in images:
            image_path = camera_dir / image_name
            # Create dummy image files
            img = Image.new("RGB", (width, height))
            img.save(image_path)

    return str(input_base)


def test_load_data(input_anno_file: str, input_base: str):
    """
    Tests the data loading functionality of the DeepenSegmentationPaints class.

    This test verifies that the DeepenSegmentationPaints class correctly loads the segmentation masks and category indices from the provided annotation zip file and input base directory.

    Args:
        input_anno_file (str): The path to the zip file containing the simulated segmentation annotations.
        input_base (str): The path to the 'input_base' directory containing the simulated image data.

    Asserts:
        - The number of loaded segmentation masks matches the expected count.
        - The number of categories matches the expected count.
    """
    deepen_segmentation_paints = DeepenSegmentationPaints(input_anno_file, input_base)
    assert len(deepen_segmentation_paints.segmentation_masks) == 4  # 2 sensors * 2 images
    assert len(deepen_segmentation_paints.index_to_category) == 2  # category1, category2


def test_to_deepen_annotations(input_anno_file: str, input_base: str):
    """
    Tests the conversion of segmentation data to Deepen annotations.

    This test verifies that the DeepenSegmentationPaints class correctly converts the loaded segmentation masks into DeepenAnnotation instances with the expected properties.

    Args:
        input_anno_file (str): The path to the zip file containing the simulated segmentation annotations.
        input_base (str): The path to the 'input_base' directory containing the simulated image data.

    Asserts:
        - The number of generated annotations matches the expected count.
        - Each annotation has the correct label category.
        - Each annotation has the correct dataset ID and label type.
    """
    deepen_segmentation_paints = DeepenSegmentationPaints(input_anno_file, input_base)
    annotations: List[DeepenAnnotation] = deepen_segmentation_paints.to_deepen_annotations()
    assert len(annotations) == 12  # 2 sensors * 2 images * 3 instances

    for annotation in annotations:
        assert annotation.label_category_id in ["category1", "category2"]
        assert annotation.dataset_id == "segmentation_prd_uuid_1970-01-01_00-00-00_00-01-00"
        assert annotation.label_type == "2d_segmentation"


def test_to_deepen_annotation_dicts(input_anno_file: str, input_base: str):
    """
    Tests the to_deepen_annotation_dicts method of DeepenSegmentationPaints.

    Args:
        input_anno_file (str): The path to the zip file containing the simulated segmentation annotations.
        input_base (str): The path to the 'input_base' directory containing the simulated image data.
    """
    # Instantiate the DeepenSegmentationPaints class
    deepen_segmentation_paints = DeepenSegmentationPaints(input_anno_file, input_base)

    # Call the to_deepen_annotation_dicts method
    annotation_dicts = deepen_segmentation_paints.to_deepen_annotation_dicts()

    # Check that the output is a list of dictionaries
    assert len(annotation_dicts) == 12  # 2 sensors * 2 images * 3 instances

    for annotation_dict in annotation_dicts:
        assert isinstance(annotation_dict, dict)
        # Check that required keys are present
        required_keys = [
            "dataset_id",
            "file_id",
            "label_category_id",
            "label_id",
            "label_type",
            "labeller_email",
            "sensor_id",
            "attributes",
            "three_d_bbox",
            "box",
            "two_d_mask",
        ]
        for key in required_keys:
            assert key in annotation_dict

        # Check values
        assert annotation_dict["label_category_id"] in ["category1", "category2"]
        assert annotation_dict["dataset_id"] == deepen_segmentation_paints.dataset_id
        assert annotation_dict["label_type"] == "2d_segmentation"
        assert annotation_dict["three_d_bbox"] is None
        assert annotation_dict["box"] is None
        assert annotation_dict["two_d_mask"] is not None
        assert isinstance(annotation_dict["two_d_mask"], str)
