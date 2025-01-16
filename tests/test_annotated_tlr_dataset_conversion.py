import json
import os.path as osp
import shutil

import pytest
import yaml

from perception_dataset.constants import SENSOR_ENUM
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter import (
    Rosbag2ToAnnotatedT4TlrConverter,
)
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR


@pytest.fixture(scope="module")
def t4_dataset_path():
    test_rosbag_name = "traffic_light_sample_tf"
    # before test - convert rosbag2 to t4
    with open(TEST_CONFIG_ROOT_DIR / "convert_rosbag2_to_annotated_t4_tlr.yaml") as f:
        config_dict = yaml.safe_load(f)

    task = config_dict["task"]
    param_args = {
        "task": task,
        **config_dict["conversion"],
    }
    t4_output_base = osp.join(TEST_ROOT_DIR, param_args["output_base"])

    param_args["input_base"] = osp.join(TEST_ROOT_DIR, param_args["input_base"])
    param_args["output_base"] = t4_output_base
    converter_params = Rosbag2ConverterParams(**param_args, with_camera=False)
    converter = Rosbag2ToAnnotatedT4TlrConverter(converter_params)
    converter.convert()

    yield osp.join(t4_output_base, test_rosbag_name)

    # after test - remove resource
    shutil.rmtree(t4_output_base, ignore_errors=True)


@pytest.fixture
def attribute_list():
    with open(TEST_CONFIG_ROOT_DIR / "label/attribute.yaml") as f:
        arrtibute_dict = yaml.safe_load(f)
        arrtibute_list = []
        for k, v in arrtibute_dict.items():
            for key in v.keys():
                arrtibute_list.append(f"{k}.{key}")

        return arrtibute_list


@pytest.fixture
def category_list():
    with open(TEST_CONFIG_ROOT_DIR / "label/traffic_light.yaml") as f:
        category_dict = yaml.safe_load(f)
        return list(category_dict.keys())


def load_json(t4_dataset_path, file_name):
    with open(osp.join(t4_dataset_path, "annotation", f"{file_name}.json")) as f:
        return json.load(f)


def get_empty(df, col):
    return (df.iloc[1:-1][col] == "").index.tolist()


def test_attribute_json(t4_dataset_path, attribute_list):
    attribute_json = load_json(t4_dataset_path, "attribute")
    assert len(attribute_json) == 0, f"attribute_json length is {len(attribute_json)}, expected 0"


def test_t4_calibrated_sensor_json(t4_dataset_path):
    calibrated_sensor = load_json(t4_dataset_path, "calibrated_sensor")
    assert (
        len(calibrated_sensor) == 1
    ), f"calibrated_sensor length is {len(calibrated_sensor)}, expected 1"


def test_category_json(t4_dataset_path, category_list):
    category_json = load_json(t4_dataset_path, "category")
    assert len(category_json) == 2, f"category length is {len(category_json)}, expected 2"
    for category in category_json:
        assert category["name"], "name is empty"
        assert category["token"], "token is empty"
        assert category["name"] in category_list, f"{category['name']} is not in {category_list}"


def test_ego_pose_json(t4_dataset_path):
    ego_pose_json = load_json(t4_dataset_path, "ego_pose")
    assert len(ego_pose_json) == 15, f"ego_pose length is {len(ego_pose_json)}, expected 15"
    for ego_pose in ego_pose_json:
        assert ego_pose["translation"], "translation is empty"
        assert ego_pose["rotation"], "rotation is empty"
        assert ego_pose["token"], "token is empty"
        assert ego_pose["timestamp"], "timestamp is empty"


def test_instance_json(t4_dataset_path):
    instance_json = load_json(t4_dataset_path, "instance")
    assert len(instance_json) == 2, f"instance length is {len(instance_json)}, expected 2"
    for instance in instance_json:
        assert instance["token"], "token is empty"
        assert instance["category_token"], "category_token is empty"
        assert instance["instance_name"], "instance_name is empty"
        assert instance["nbr_annotations"] >= 0, "nbr_annotations is empty"
        if instance["nbr_annotations"] > 0:
            assert instance["first_annotation_token"], "first_annotation_token is empty"
            assert instance["last_annotation_token"], "last_annotation_token is empty"


def test_log_json(t4_dataset_path):
    log_json = load_json(t4_dataset_path, "log")
    assert len(log_json) == 1, f"log length is {len(log_json)}, expected 1"
    for log in log_json:
        assert log["token"], "token is empty"
        assert log["logfile"] == "", "logfile is empty"
        assert log["vehicle"] == "", "vehicle is empty"
        assert log["data_captured"] == "", "data_captured is empty"
        assert log["location"] == "", "location is empty"


def test_map_json(t4_dataset_path):
    map_json = load_json(t4_dataset_path, "map")
    assert len(map_json) == 1, f"map length is {len(map_json)}, expected 1"
    for map_ in map_json:
        assert map_["token"], "token is empty"
        assert map_["log_tokens"], "log_tokens is empty"
        assert map_["category"] == "", "category is empty"
        assert map_["filename"] == "", "filename is empty"


def test_object_ann_json(t4_dataset_path):
    object_ann_json = load_json(t4_dataset_path, "object_ann")
    assert len(object_ann_json) == 10, f"object_ann length is {len(object_ann_json)}, expected 10"
    for object_ann in object_ann_json:
        assert object_ann["token"], "token is empty"
        assert object_ann["sample_data_token"], "sample_data_token is empty"
        assert object_ann["instance_token"], "instance_token is empty"
        assert object_ann["category_token"], "category_token is empty"
        assert object_ann["bbox"], "bbox is empty"
        assert object_ann["mask"], "mask is empty"


def test_sample_data_json(t4_dataset_path):
    sample_data_json = load_json(t4_dataset_path, "sample_data")
    assert len(sample_data_json) == 5, f"sample_data length is {len(sample_data_json)}, expected 5"
    for sample_data in sample_data_json:
        assert sample_data["token"], "token is empty"
        assert sample_data["sample_token"], "sample_token is empty"
        assert sample_data["ego_pose_token"], "instance_token is empty"
        assert sample_data["calibrated_sensor_token"], "calibrated_sensor_token is empty"
        assert sample_data["filename"], "filename is empty"
        assert sample_data["width"] >= 0, "width is empty"
        assert sample_data["height"] >= 0, "height is empty"
        assert (not sample_data["is_valid"] and not sample_data["is_key_frame"]) or sample_data[
            "is_valid"
        ], f"is_key_frame is {sample_data['is_key_frame']}, is_valid is {sample_data['is_valid']}"
        assert "next" in sample_data.keys(), "next is empty"
        assert "prev" in sample_data.keys(), "prev is empty"
        if sample_data["filename"] == "data/CAM_TRAFFIC_LIGHT_NEAR/00000.jpg":
            assert (
                sample_data["timestamp"] == 1694567424298557
            ), "the first CAM_TRAFFIC_LIGHT_NEAR timestamp is not 1694567424.298557"


def test_sample_json(t4_dataset_path):
    sample_json = load_json(t4_dataset_path, "sample")
    assert len(sample_json) == 5, f"sample length is {len(sample_json)}, expected 5"
    for sample in sample_json:
        assert sample["token"], "token is empty"
        assert sample["timestamp"], "`timesta`mp is empty"
        assert sample["scene_token"], "scene_token is empty"
        assert "next" in sample.keys(), "next is empty"
        assert "prev" in sample.keys(), "prev is empty"


def test_scene_json(t4_dataset_path):
    scene_json = load_json(t4_dataset_path, "scene")
    assert len(scene_json) == 1, f"scene length is {len(scene_json)}, expected 1"
    for scene in scene_json:
        assert scene["token"], "token is empty"
        assert scene["name"], "name is empty"
        assert scene["log_token"], "log_token is empty"
        assert scene["nbr_samples"], "nbr_samples is empty"
        assert scene["first_sample_token"], "first_sample_token is empty"
        assert scene["last_sample_token"], "last_sample_token is empty"


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


def test_surface_ann_json(t4_dataset_path):
    surface_ann_json = load_json(t4_dataset_path, "surface_ann")
    assert surface_ann_json == [], "surface_ann is not empty"


def test_visibility_json(t4_dataset_path):
    visibility_json = load_json(t4_dataset_path, "visibility")
    assert len(visibility_json) <= 4, f"visibility length is {len(visibility_json)}, expected 4"
    for visibility in visibility_json:
        assert visibility["token"], "token is empty"
        assert visibility["level"] in [
            "full",
            "most",
            "partial",
            "none",
        ]
        assert visibility["description"], "description is empty"
