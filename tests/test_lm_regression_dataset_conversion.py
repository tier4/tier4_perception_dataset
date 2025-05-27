import json
import os
import os.path as osp
import shutil

import pytest
import yaml

from perception_dataset.constants import SENSOR_ENUM
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_t4_loc_converter import Rosbag2ToT4LocConverter
from perception_dataset.utils.rosbag2 import get_topic_count
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR


@pytest.fixture(scope="module")
def t4_dataset_path(request):
    test_rosbag_name = "LM_regression_x1"
    # before test - convert rosbag2 to t4
    with open(TEST_CONFIG_ROOT_DIR / "convert_rosbag2_to_localization_evaluation.yaml") as f:
        config_dict = yaml.safe_load(f)

    input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])

    config_dict["conversion"]["input_base"] = input_base
    config_dict["conversion"]["output_base"] = output_base
    assert osp.exists(input_base), f"input_base doesn't exist: {input_base}"

    param_args = {
        "task": config_dict["task"],
        "scene_description": config_dict["description"]["scene"],
        **config_dict["conversion"],
    }
    converter_params = Rosbag2ConverterParams(**param_args)
    converter = Rosbag2ToT4LocConverter(converter_params)

    converter.convert()

    # provide a path to converted t4_dataset
    yield osp.join(output_base, test_rosbag_name)

    # after test - remove resource
    shutil.rmtree(output_base, ignore_errors=True)


@pytest.fixture
def attribute_list():
    with open(TEST_CONFIG_ROOT_DIR / "label/attribute.yaml") as f:
        attribute_dict = yaml.safe_load(f)
        arrtibute_list = []
        for k, v in attribute_dict.items():
            for key in v.keys():
                arrtibute_list.append(f"{k}.{key}")

        return arrtibute_list


@pytest.fixture
def category_list():
    with open(TEST_CONFIG_ROOT_DIR / "label/object.yaml") as f:
        category_dict = yaml.safe_load(f)
        return list(category_dict.keys())


def load_json(t4_dataset_path, file_name):
    with open(osp.join(t4_dataset_path, "annotation", f"{file_name}.json")) as f:
        return json.load(f)


def get_empty(df, col):
    return (df.iloc[1:-1][col] == "").index.tolist()


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_attribute_json(t4_dataset_path, attribute_list):
    attribute_json = load_json(t4_dataset_path, "attribute")
    assert attribute_json == [], "attribute_json is not empty"


@pytest.mark.parametrize("t4_dataset_path", [False], indirect=True)
def test_calibrated_sensor_json(t4_dataset_path):
    calibrated_sensor = load_json(t4_dataset_path, "calibrated_sensor")
    assert len(calibrated_sensor) == 1


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_category_json(t4_dataset_path, category_list):
    category_json = load_json(t4_dataset_path, "category")
    assert category_json == [], "category_json is not empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_ego_pose_json(t4_dataset_path):
    ego_pose_json = load_json(t4_dataset_path, "ego_pose")
    assert len(ego_pose_json) == 1, f"ego_pose length is {len(ego_pose_json)}, expected 1"
    for ego_pose in ego_pose_json:
        assert ego_pose["translation"], "translation is empty"
        assert ego_pose["rotation"], "rotation is empty"
        assert ego_pose["token"], "token is empty"
        assert ego_pose["timestamp"], "timestamp is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_instance_json(t4_dataset_path):
    instance_json = load_json(t4_dataset_path, "instance")
    assert instance_json == [], "instance is not empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_log_json(t4_dataset_path):
    log_json = load_json(t4_dataset_path, "log")
    assert len(log_json) == 1, f"log length is {len(log_json)}, expected 1"
    for log in log_json:
        assert log["token"], "token is empty"
        assert log["logfile"] == "", "logfile is empty"
        assert log["vehicle"] == "", "vehicle is empty"
        assert log["data_captured"] == "", "data_captured is empty"
        assert log["location"] == "", "location is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_map_json(t4_dataset_path):
    map_json = load_json(t4_dataset_path, "map")
    assert len(map_json) == 1, f"map length is {len(map_json)}, expected 1"
    for map_ in map_json:
        assert map_["token"], "token is empty"
        assert map_["log_tokens"], "log_tokens is empty"
        assert map_["category"] == "", "category is empty"
        assert map_["filename"] == "", "filename is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_object_ann_json(t4_dataset_path):
    object_ann_json = load_json(t4_dataset_path, "object_ann")
    assert object_ann_json == [], "object_ann is not empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_annotation_json(t4_dataset_path):
    sample_annotation = load_json(t4_dataset_path, "sample_annotation")
    assert sample_annotation == [], "sample_annotation is not empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_data_json(t4_dataset_path):
    sample_data_json = load_json(t4_dataset_path, "sample_data")
    assert len(sample_data_json) == 1, f"sample_data length is {len(sample_data_json)}, expected 1"
    sample_data = sample_data_json[0]
    assert sample_data["token"], "token is empty"
    assert sample_data["sample_token"], "sample_token is empty"
    assert sample_data["ego_pose_token"], "instance_token is empty"
    assert sample_data["calibrated_sensor_token"], "calibrated_sensor_token is empty"
    assert (
        sample_data["filename"] == "data/LIDAR_CONCAT/00000.pcd.bin"
    ), "filename is not data/LIDAR_CONCAT/00000.pcd.bin"
    assert sample_data["width"] == 0, "width is not 0"
    assert sample_data["height"] == 0, "height is not 0"
    assert (not sample_data["is_valid"] and not sample_data["is_key_frame"]) or sample_data[
        "is_valid"
    ], f"is_key_frame is {sample_data['is_key_frame']}, is_valid is {sample_data['is_valid']}"
    assert sample_data["next"] == "", "next is not empty"
    assert sample_data["prev"] == "", "prev is not empty"
    assert sample_data["timestamp"], "timestamp is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_json(t4_dataset_path):
    sample_json = load_json(t4_dataset_path, "sample")
    assert len(sample_json) == 1, f"sample length is {len(sample_json)}, expected 1"
    sample = sample_json[0]
    assert sample["token"], "token is empty"
    assert sample["timestamp"], "timestamp is empty"
    assert sample["scene_token"], "scene_token is empty"
    assert "next" in sample.keys(), "next is empty"
    assert "prev" in sample.keys(), "prev is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_scene_json(t4_dataset_path):
    scene_json = load_json(t4_dataset_path, "scene")
    assert len(scene_json) == 1, f"scene length is {len(scene_json)}, expected 1"
    for scene in scene_json:
        assert scene["token"], "token is empty"
        assert scene["name"], "name is empty"
        assert (
            scene["description"] == "localization_evaluation, dummy_pcd"
        ), "description is not localization_evaluation, dummy_pcd"
        assert scene["log_token"], "log_token is empty"
        assert scene["nbr_samples"], "nbr_samples is empty"
        assert scene["first_sample_token"], "first_sample_token is empty"
        assert scene["last_sample_token"], "last_sample_token is empty"
        assert scene["nbr_samples"] == 1, f"nbr_samples is {scene['nbr_samples']}, expected 1"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sensor_json(t4_dataset_path):
    sensor_json = load_json(t4_dataset_path, "sensor")
    assert len(sensor_json) == 1, f"sensor length is {len(sensor_json)}, expected 1"
    for sensor in sensor_json:
        assert sensor["token"], "token is empty"
        assert SENSOR_ENUM.has_channel(sensor["channel"])
        assert sensor["modality"] in [
            "camera",
            "lidar",
        ], f"modality is {sensor['modality']} not in ['camera','lidar']"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_surface_ann_json(t4_dataset_path):
    surface_ann_json = load_json(t4_dataset_path, "surface_ann")
    assert surface_ann_json == [], "surface_ann is not empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_visibility_json(t4_dataset_path):
    visibility_json = load_json(t4_dataset_path, "visibility")
    assert visibility_json == [], "visibility is not empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_directory_structure(t4_dataset_path):
    dir_files = os.listdir(t4_dataset_path)
    assert "annotation" in dir_files, "annotation is not in t4_dataset"
    assert "data" in dir_files, "data is not in t4_dataset"
    assert "input_bag" in dir_files, "input_bag is not in t4_dataset"
    assert "status.json" in dir_files, "status.json is not in t4_dataset"

    input_bag_files = os.listdir(osp.join(t4_dataset_path, "input_bag"))
    assert "metadata.yaml" in input_bag_files, "metadata.yaml is not in input_bag"
    assert (
        "LM_regression_x1_0.db3" in input_bag_files
    ), "LM_regression_x1_0.db3 is not in input_bag"

    topic_count_dict = get_topic_count(osp.join(t4_dataset_path, "input_bag"))

    assert (
        "/localization/util/downsample/pointcloud" in topic_count_dict.keys()
    ), "/localization/util/downsample/pointcloud is not in topic_count_dict"
    assert (
        "/localization/twist_estimator/twist_with_covariance" in topic_count_dict.keys()
    ), "/localization/twist_estimator/twist_with_covariance is not in topic_count_dict"
    assert (
        "/localization/reference_kinematic_state" in topic_count_dict.keys()
    ), "/localization/reference_kinematic_state is not in topic_count_dict"
    assert "/initialpose" in topic_count_dict.keys(), "/initialpose is not in topic_count_dict"
