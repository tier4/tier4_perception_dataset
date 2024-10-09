import json
from pathlib import Path
from typing import List
import pytest
from PIL import Image
import numpy as np
from pycocotools import mask as cocomask
import base64
from perception_dataset.deepen.deepen_annotation import DeepenAnnotation
from perception_dataset.deepen.segmentation.deepen_segmentation_polygons import DeepenSegmentationPolygons


@pytest.fixture
def input_base(tmp_path: Path) -> str:
    """
    Pytest fixture to set up the 'input_base' directory with a simulated data structure containing a dummy image file.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Returns:
        str: The path to the 'input_base' directory containing the simulated image data.
    """
    input_base = tmp_path / 'input_base'
    data_dir = input_base / 'data' / 'sensor1'
    data_dir.mkdir(parents=True)

    # Create a dummy image file
    file_id = '1.jpg'
    width: int = 640
    height: int = 480
    image_path = data_dir / file_id
    img = Image.new('RGB', (width, height))
    img.save(image_path)

    return str(input_base)


def test_deepen_segmentation_polygons(input_base: str, tmp_path: Path):
    """
    Tests the DeepenSegmentationPolygons class by processing test data with simplified polygons.

    Args:
        input_base (str): The path to the 'input_base' directory containing the simulated image data.
        tmp_path (Path): Temporary directory provided by pytest.
    """
    # Create test annotation data with simplified polygons (4 points)
    test_polygon = [
        [100.0, 100.0],
        [200.0, 100.0],
        [200.0, 200.0],
        [100.0, 200.0]
    ]
    test_data = {
        "labels": [
            {
                "file_id": "1.jpg",
                "label_category_id": "vehicle_car",
                "label_id": "vehicle_car:2",
                "label_type": "polygon",
                "stage_id": "QA",
                "attributes": {},
                "label_set_id": "default",
                "labeller_email": "annotator@example.com",
                "polygons": [test_polygon],
                "sensor_id": "sensor1",
                "dataset_name": "test_dataset"
            }
        ]
    }

    # Write test annotation data to a JSON file
    annotations_file = tmp_path / 'annotations.json'
    with annotations_file.open('w') as f:
        json.dump(test_data, f)

    # Instantiate the DeepenSegmentationPolygons class
    deepen_segmentation_polygons = DeepenSegmentationPolygons(str(annotations_file), input_base)

    # Process the annotations
    deepen_annotations: List[DeepenAnnotation] = deepen_segmentation_polygons.to_deepen_annotations()

    # Verify the output annotations
    assert len(deepen_annotations) == 1
    annotation = deepen_annotations[0]
    assert annotation.dataset_id == "test_dataset"
    assert annotation.file_id == "1.jpg"
    assert annotation.label_category_id == "vehicle_car"
    assert annotation.label_id == "vehicle_car:2"
    assert annotation.label_type == "polygon"
    assert annotation.sensor_id == "sensor1"
    assert annotation.labeller_email == "annotator@example.com"
    assert annotation.attributes == {}
    assert annotation.two_d_mask is not None
    assert annotation.three_d_bbox is None
    assert annotation.two_d_box is None

    # Decode the RLE counts from base64
    rle_counts = base64.b64decode(annotation.two_d_mask)
    rle = {
        'counts': rle_counts,
        'size': [deepen_segmentation_polygons.height, deepen_segmentation_polygons.width]
    }

    # Convert RLE to binary mask
    binary_mask_from_rle = cocomask.decode(rle)

    # Generate binary mask from original polygon using pycocotools
    formatted_polygon = [[coord for point in test_polygon for coord in point]]
    rle_original = cocomask.frPyObjects(formatted_polygon, deepen_segmentation_polygons.height, deepen_segmentation_polygons.width)
    rle_original = cocomask.merge(rle_original)
    binary_mask_from_polygon = cocomask.decode(rle_original)

    # Ensure the masks are the same
    assert np.array_equal(binary_mask_from_rle, binary_mask_from_polygon), "The generated mask does not match the original polygon."

    # Verify that the mask contains the expected number of pixels
    expected_area = np.sum(binary_mask_from_polygon)
    actual_area = np.sum(binary_mask_from_rle)
    assert actual_area == expected_area

def test_to_deepen_annotation_dicts(input_base: str, tmp_path: Path):
    """
    Tests the to_deepen_annotation_dicts method of DeepenSegmentationPolygons.

    Args:
        input_base (str): The path to the 'input_base' directory containing the simulated image data.
        tmp_path (Path): Temporary directory provided by pytest.
    """
    # Create test annotation data with polygons
    test_polygon = [
        [100.0, 100.0],
        [200.0, 100.0],
        [200.0, 200.0],
        [100.0, 200.0]
    ]
    test_data = {
        "labels": [
            {
                "file_id": "1.jpg",
                "label_category_id": "vehicle_car",
                "label_id": "vehicle_car:1",
                "label_type": "polygon",
                "stage_id": "QA",
                "attributes": {},
                "create_time_millis": 1724327319897,
                "label_set_id": "default",
                "labeller_email": "annotator@example.com",
                "polygons": [test_polygon],
                "sensor_id": "sensor1",
                "update_time_millis": 1724327319897,
                "user_id": "tester@example.com",
                "dataset_name": "test_dataset"
            }
        ]
    }

    # Write test annotation data to a JSON file
    annotations_file = tmp_path / 'annotations.json'
    with annotations_file.open('w') as f:
        json.dump(test_data, f)

    # Instantiate the DeepenSegmentationPolygons class
    deepen_segmentation_polygons = DeepenSegmentationPolygons(str(annotations_file), input_base)

    # Call the to_deepen_annotation_dicts method
    annotation_dicts = deepen_segmentation_polygons.to_deepen_annotation_dicts()

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
        assert annotation_dict["dataset_id"] == "test_dataset"
        assert annotation_dict["file_id"] == "1.jpg"
        assert annotation_dict["label_category_id"] == "vehicle_car"
        assert annotation_dict["label_id"] == "vehicle_car:1"
        assert annotation_dict["label_type"] == "polygon"
        assert annotation_dict["sensor_id"] == "sensor1"
        assert annotation_dict["attributes"] == {}
        assert annotation_dict["three_d_bbox"] is None
        assert annotation_dict["box"] is None
        assert annotation_dict["two_d_mask"] is not None
        assert isinstance(annotation_dict["two_d_mask"], str)

        # Optionally, verify the RLE mask
        rle_counts_encoded = annotation_dict["two_d_mask"]
        # Decode the RLE counts from base64
        rle_counts = base64.b64decode(rle_counts_encoded)
        rle = {
            'counts': rle_counts,
            'size': [deepen_segmentation_polygons.height, deepen_segmentation_polygons.width]
        }
        # Convert RLE to binary mask
        binary_mask = cocomask.decode(rle)
        # Verify that the mask has the expected shape
        assert binary_mask.shape == (deepen_segmentation_polygons.height, deepen_segmentation_polygons.width)
        # Verify that the mask contains the expected area
        expected_area = 10000  # Area of the square polygon (100x100 pixels)
        actual_area = np.sum(binary_mask)
        assert actual_area == expected_area
