import pytest

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation


def test_deepen_annotation_valid_three_d_bbox():
    """
    Test that a DeepenAnnotation with a valid three_d_bbox is created successfully.
    """
    three_d_bbox = {
        "cx": 1.0,
        "cy": 2.0,
        "cz": 3.0,
        "h": 1.5,
        "l": 4.0,
        "w": 2.0,
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    annotation = DeepenAnnotation(
        dataset_id="dataset_001",
        file_id="0.pcd",
        label_category_id="car",
        label_id="car:1",
        label_type="3d_bbox",
        sensor_id="lidar",
        three_d_bbox=three_d_bbox,
    )
    assert annotation.three_d_bbox == three_d_bbox
    assert annotation.two_d_box is None
    assert annotation.two_d_mask is None


def test_deepen_annotation_valid_two_d_box():
    """
    Test that a DeepenAnnotation with a valid two_d_box is created successfully.
    """
    two_d_box = [100.0, 200.0, 50.0, 150.0]
    annotation = DeepenAnnotation(
        dataset_id="dataset_002",
        file_id="image_001.jpg",
        label_category_id="pedestrian",
        label_id="pedestrian:1",
        label_type="box",
        sensor_id="camera1",
        two_d_box=two_d_box,
    )
    assert annotation.two_d_box == two_d_box
    assert annotation.three_d_bbox is None
    assert annotation.two_d_mask is None


def test_deepen_annotation_valid_two_d_mask():
    """
    Test that a DeepenAnnotation with a valid two_d_mask is created successfully.
    """
    two_d_mask = "RLE_string_here"
    annotation = DeepenAnnotation(
        dataset_id="dataset_003",
        file_id="image_002.jpg",
        label_category_id="debris",
        label_id="debris:1",
        label_type="2d_segmentation",
        sensor_id="camera2",
        two_d_mask=two_d_mask,
    )
    assert annotation.two_d_mask == two_d_mask
    assert annotation.three_d_bbox is None
    assert annotation.two_d_box is None


def test_deepen_annotation_no_annotation_provided():
    """
    Test that creating a DeepenAnnotation without any annotation data raises an AssertionError.
    """
    with pytest.raises(AssertionError) as excinfo:
        DeepenAnnotation(
            dataset_id="dataset_004",
            file_id="image_003.jpg",
            label_category_id="debris",
            label_id="debris:1",
            label_type="2d_segmentation",
            sensor_id="camera",
            # No annotation data provided
        )
    assert "Exactly one of three_d_bbox, two_d_box, or two_d_mask must be provided." in str(
        excinfo.value
    )


def test_deepen_annotation_multiple_annotations_provided():
    """
    Test that providing multiple annotation types raises an AssertionError.
    """
    with pytest.raises(AssertionError) as excinfo:
        DeepenAnnotation(
            dataset_id="dataset_005",
            file_id="image_004.jpg",
            label_category_id="debris",
            label_id="debris:1",
            label_type="2d_segmentation",
            sensor_id="camera",
            two_d_box=[50.0, 50.0, 200.0, 400.0],
            two_d_mask="Another_RLE_string_here",
        )
    assert "Exactly one of three_d_bbox, two_d_box, or two_d_mask must be provided." in str(
        excinfo.value
    )


def test_deepen_annotation_invalid_three_d_bbox():
    """
    Test that missing required keys in three_d_bbox raises an AssertionError.
    """
    three_d_bbox = {
        "cx": 1.0,
        "cy": 2.0,
        # Missing 'cz' key
        "h": 1.5,
        "l": 4.0,
        "w": 2.0,
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    with pytest.raises(AssertionError) as excinfo:
        DeepenAnnotation(
            dataset_id="dataset_006",
            file_id="0.pcd",
            label_category_id="car",
            label_id="car:2",
            label_type="3d_bbox",
            sensor_id="lidar",
            three_d_bbox=three_d_bbox,
        )
    assert "three_d_bbox is missing keys: ['cz']" in str(excinfo.value)


def test_deepen_annotation_invalid_quaternion():
    """
    Test that missing required keys in quaternion raises an AssertionError.
    """
    three_d_bbox = {
        "cx": 1.0,
        "cy": 2.0,
        "cz": 3.0,
        "h": 1.5,
        "l": 4.0,
        "w": 2.0,
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0},  # Missing 'w' key
    }
    with pytest.raises(AssertionError) as excinfo:
        DeepenAnnotation(
            dataset_id="dataset_007",
            file_id="0.pcd",
            label_category_id="car",
            label_id="car:3",
            label_type="3d_bbox",
            sensor_id="lidar",
            three_d_bbox=three_d_bbox,
        )
    assert "quaternion is missing keys: ['w']" in str(excinfo.value)


def test_deepen_annotation_invalid_two_d_box():
    """
    Test that an invalid two_d_box (incorrect length) raises an AssertionError.
    """
    two_d_box = [100.0, 200.0, 50.0]  # Only three elements
    with pytest.raises(AssertionError) as excinfo:
        DeepenAnnotation(
            dataset_id="dataset_008",
            file_id="image_005.jpg",
            label_category_id="pedestrian",
            label_id="pedestrian:2",
            label_type="box",
            sensor_id="camera1",
            two_d_box=two_d_box,
        )
    assert "two_d_box must be a list of four elements." in str(excinfo.value)


def test_deepen_annotation_invalid_two_d_mask():
    """
    Test that an empty two_d_mask string raises an AssertionError.
    """
    two_d_mask = ""
    with pytest.raises(AssertionError) as excinfo:
        DeepenAnnotation(
            dataset_id="dataset_009",
            file_id="image_006.jpg",
            label_category_id="debris",
            label_id="debris:2",
            label_type="2d_segmentation",
            sensor_id="camera2",
            two_d_mask=two_d_mask,
        )
    assert "two_d_mask must not be an empty string." in str(excinfo.value)


def test_deepen_annotation_valid_attributes():
    """
    Test that attributes are correctly assigned.
    """
    attributes = {"state": "moving", "occlusion": "none", "cycle_state": "with_rider"}
    two_d_box = [150.0, 250.0, 60.0, 180.0]
    annotation = DeepenAnnotation(
        dataset_id="dataset_010",
        file_id="image_007.jpg",
        label_category_id="cyclist",
        label_id="cyclist:1",
        label_type="box",
        sensor_id="camera1",
        two_d_box=two_d_box,
        attributes=attributes,
    )
    assert annotation.attributes == attributes


def test_deepen_annotation_to_dict():
    """
    Test that the to_dict method returns the correct dictionary representation.
    """
    two_d_box = [120.0, 220.0, 70.0, 160.0]
    annotation = DeepenAnnotation(
        dataset_id="dataset_011",
        file_id="image_008.jpg",
        label_category_id="pedestrian",
        label_id="pedestrian:3",
        label_type="box",
        sensor_id="camera1",
        two_d_box=two_d_box,
    )
    annotation_dict = annotation.to_dict()
    expected_dict = {
        "dataset_id": "dataset_011",
        "file_id": "image_008.jpg",
        "label_category_id": "pedestrian",
        "label_id": "pedestrian:3",
        "label_type": "box",
        "labeller_email": "default@tier4.jp",
        "sensor_id": "camera1",
        "attributes": {},
        "three_d_bbox": None,
        "box": [120.0, 220.0, 70.0, 160.0],
        "two_d_mask": None,
    }
    assert annotation_dict == expected_dict


def test_deepen_annotation_from_dict_valid():
    """
    Test that a DeepenAnnotation instance is correctly created from a valid dictionary.
    """
    data = {
        "dataset_id": "dataset_012",
        "file_id": "image_009.jpg",
        "label_category_id": "car",
        "label_id": "car:1",
        "label_type": "box",
        "sensor_id": "camera2",
        "labeller_email": "annotator@example.com",
        "attributes": {"text": "Stop"},
        "box": [50.0, 100.0, 30.0, 60.0],
    }
    annotation = DeepenAnnotation.from_dict(data)
    assert annotation.dataset_id == data["dataset_id"]
    assert annotation.file_id == data["file_id"]
    assert annotation.label_category_id == data["label_category_id"]
    assert annotation.label_id == data["label_id"]
    assert annotation.label_type == data["label_type"]
    assert annotation.sensor_id == data["sensor_id"]
    assert annotation.labeller_email == data["labeller_email"]
    assert annotation.attributes == data["attributes"]
    assert annotation.two_d_box == data["box"]
    assert annotation.three_d_bbox is None
    assert annotation.two_d_mask is None


def test_deepen_annotation_from_dict_missing_required_fields():
    """
    Test that missing required fields in the input dictionary raises a KeyError.
    """
    data = {
        # Missing 'dataset_id' and 'file_id'
        "label_category_id": "pedestrian",
        "label_id": "pedestrian:1",
        "label_type": "box",
        "sensor_id": "camera2",
        "box": [50.0, 100.0, 30.0, 60.0],
    }
    with pytest.raises(KeyError) as excinfo:
        DeepenAnnotation.from_dict(data)
    assert "Missing required fields in data: ['dataset_id', 'file_id']" in str(excinfo.value)


def test_deepen_annotation_from_dict_invalid_annotation_data():
    """
    Test that invalid annotation data in the input dictionary raises an AssertionError.
    """
    data = {
        "dataset_id": "dataset_013",
        "file_id": "image_010.jpg",
        "label_category_id": "bicycle",
        "label_id": "bicycle:2",
        "label_type": "box",
        "sensor_id": "camera2",
        # Invalid 'box' data (only two elements)
        "box": [50.0, 100.0],
    }
    with pytest.raises(AssertionError) as excinfo:
        DeepenAnnotation.from_dict(data)
    assert "two_d_box must be a list of four elements." in str(excinfo.value)
