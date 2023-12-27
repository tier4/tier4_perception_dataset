import json
import os
import os.path as osp
import shutil

from nuscenes.nuscenes import NuScenes
import pandas as pd
import pytest
import yaml

from perception_dataset.constants import SENSOR_ENUM, T4_FORMAT_DIRECTORY_NAME
from perception_dataset.rosbag2.converter_params import DataType, Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_t4_tracking_converter import (
    Rosbag2ToT4TrackingConverter,
)
from perception_dataset.t4_dataset.data_validator import validate_data_hz
from perception_dataset.t4_dataset.format_validator import (
    validate_directory_structure,
    validate_format,
)
from perception_dataset.utils.rosbag2 import get_topic_count
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR


@pytest.fixture(scope="module")
def t4_dataset_path(request):
    test_rosbag_name = "tracking_sim_sample_data"
    # before test - convert rosbag2 to t4
    with open(TEST_CONFIG_ROOT_DIR / "convert_tracking_debugger_to_tracking_eval.yaml") as f:
        config_dict = yaml.safe_load(f)

    input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])

    config_dict["conversion"]["input_base"] = input_base
    config_dict["conversion"]["output_base"] = output_base
    assert osp.exists(input_base), f"input_base doesn't exist: {input_base}"

    param_args = {
        "task": config_dict["task"],
        "data_type": DataType.SYNTHETIC,
        "scene_description": config_dict["description"]["scene"],
        "overwrite_mode": True,
        **config_dict["conversion"],
    }
    converter_params = Rosbag2ConverterParams(**param_args)
    converter = Rosbag2ToT4TrackingConverter(converter_params)

    converter.convert()

    # provide a path to converted t4_dataset
    yield osp.join(output_base, test_rosbag_name, "t4_dataset")

    # after test - remove resource
    shutil.rmtree(output_base, ignore_errors=True)


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
    with open(TEST_CONFIG_ROOT_DIR / "label/object.yaml") as f:
        category_dict = yaml.safe_load(f)
        return list(category_dict.keys())


def load_json(t4_dataset_path, file_name):
    with open(osp.join(t4_dataset_path, "annotation", f"{file_name}.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("t4_dataset_path", [True, False], indirect=True)
def test_t4_dataset_format(t4_dataset_path):
    validate_directory_structure(t4_dataset_path)

    nusc = NuScenes(
        version=T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value,
        dataroot=t4_dataset_path,
        verbose=False,
    )

    validate_format(nusc, t4_dataset_path)
    validate_data_hz(nusc)


def get_empty(df, col):
    return (df.iloc[1:-1][col] == "").index.tolist()


@pytest.mark.parametrize("t4_dataset_path", [True, False], indirect=True)
def test_rosbag2_converter_dataset_consistency(t4_dataset_path):
    sample_annotation = load_json(t4_dataset_path, "sample_annotation")
    # First frame doesn't have prev frame
    grouped = pd.DataFrame(sample_annotation).groupby("instance_token")
    for _, annotations in grouped:
        if len(annotations) == 1:
            assert annotations.iloc[0]["prev"] == ""
            assert annotations.iloc[0]["next"] == ""
            continue

        # First frame doesn't have prev frame
        assert annotations.iloc[0]["prev"] == ""
        assert annotations.iloc[0]["next"]

        # Last frame doesn't have next frame
        assert annotations.iloc[-1]["prev"]
        assert annotations.iloc[-1]["next"] == ""

        if len(annotations) <= 2:
            continue

        # All other frames should have both prev and next
        assert (
            annotations.iloc[1:-1]["next"] != ""
        ).all(), f'next is empty at indexes {get_empty("next")}'
        assert (
            annotations.iloc[1:-1]["prev"] != ""
        ).all(), f'prev is empty at indexes {get_empty("prev")}'

        # All other frames should have both prev and next that are not equal
        assert (
            annotations.iloc[1:-1]["prev"] != annotations.iloc[1:-1]["next"]
        ).all() == True  # noqa E712


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_attribute_json(t4_dataset_path, attribute_list):
    attribute_json = load_json(t4_dataset_path, "attribute")
    assert len(attribute_json) == 0, f"attribute_json length is {len(attribute_json)}, expected 0"
    assert len(attribute_json) <= len(
        attribute_list
    ), f"attribute_json length more than {len(attribute_list)}, expected {len(attribute_list), {attribute_list}}"
    for attribute in attribute_json:
        assert attribute["name"], "name is empty"
        assert attribute["token"], "token is empty"
        assert (
            attribute["name"] in attribute_list
        ), f"{attribute['name']} is not in {attribute_list}"


@pytest.mark.parametrize("t4_dataset_path", [False], indirect=True)
def test_calibrated_sensor_json(t4_dataset_path):
    calibrated_sensor = load_json(t4_dataset_path, "calibrated_sensor")
    assert len(calibrated_sensor) == 1


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_category_json(t4_dataset_path, category_list):
    category_json = load_json(t4_dataset_path, "category")
    assert len(category_json) == 1, f"category length is {len(category_json)}, expected 1"
    for category in category_json:
        assert category["name"], "name is empty"
        assert category["token"], "token is empty"
        assert category["name"] in category_list, f"{category['name']} is not in {category_list}"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_ego_pose_json(t4_dataset_path):
    ego_pose_json = load_json(t4_dataset_path, "ego_pose")
    assert len(ego_pose_json) == 88, f"ego_pose length is {len(ego_pose_json)}, expected 88"
    assert (
        ego_pose_json[0]["timestamp"] == 1699630502235902
    ), "the first timestamp of ego_pose is not 1699630502.235902"
    assert (
        ego_pose_json[-1]["timestamp"] == 1699630510935894
    ), "the last timestamp of ego_pose is not 1699630510.935894"
    for ego_pose in ego_pose_json:
        assert ego_pose["translation"], "translation is empty"
        assert ego_pose["rotation"], "rotation is empty"
        assert ego_pose["token"], "token is empty"
        assert ego_pose["timestamp"], "timestamp is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_instance_json(t4_dataset_path):
    instance_json = load_json(t4_dataset_path, "instance")
    assert len(instance_json) == 1, f"instance length is {len(instance_json)}, expected 1"
    for instance in instance_json:
        assert instance["token"], "token is empty"
        assert instance["category_token"], "category_token is empty"
        assert instance["instance_name"], "instance_name is empty"
        assert instance["nbr_annotations"] >= 0, "nbr_annotations is empty"
        if instance["nbr_annotations"] > 0:
            assert instance["first_annotation_token"], "first_annotation_token is empty"
            assert instance["last_annotation_token"], "last_annotation_token is empty"


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
    assert len(object_ann_json) == 0, f"object_ann length is {len(object_ann_json)}, expected 0"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_annotation_json(t4_dataset_path):
    sample_annotation = load_json(t4_dataset_path, "sample_annotation")
    assert (
        len(sample_annotation) == 49
    ), f"sample_annotation length is {len(sample_annotation)}, expected 49"
    for sample_anno in sample_annotation:
        sample_anno: dict
        assert sample_anno["token"], "token is empty"
        assert sample_anno["sample_token"], "sample_token is empty"
        assert sample_anno["instance_token"], "instance_token is empty"

        assert sample_anno["visibility_token"], "visibility_token is empty"
        assert sample_anno["translation"], "translation is empty"
        assert "velocity" in sample_anno.keys(), "sample_annotation must have velocity key"
        assert "acceleration" in sample_anno.keys(), "sample_annotation must have acceleration key"
        assert sample_anno["size"], "size is empty"
        assert sample_anno["rotation"], "rotation is empty"
        assert sample_anno["num_lidar_pts"] >= 0, "num_lidar_pts is empty"
        assert sample_anno["num_radar_pts"] >= 0, "num_radar_pts is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_data_json(t4_dataset_path):
    sample_data_json = load_json(t4_dataset_path, "sample_data")
    assert (
        len(sample_data_json) == 88
    ), f"sample_data length is {len(sample_data_json)}, expected 88"
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
        if sample_data["filename"] == "data/CAM_BACK/00000.jpg":
            assert (
                sample_data["timestamp"] == 1660889208947739
            ), "the first back-camera timestamp is not 1660889208.947739"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_json(t4_dataset_path):
    sample_json = load_json(t4_dataset_path, "sample")
    assert len(sample_json) == 88, f"sample length is {len(sample_json)}, expected 88"
    for sample in sample_json:
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
            scene["description"] == "tracking_regression, synthetic"
        ), "description is not tracking_regression, synthetic"
        assert scene["log_token"], "log_token is empty"
        assert scene["nbr_samples"], "nbr_samples is empty"
        assert scene["first_sample_token"], "first_sample_token is empty"
        assert scene["last_sample_token"], "last_sample_token is empty"

        assert scene["nbr_samples"] == 88, f"nbr_samples is {scene['nbr_samples']}, expected 88"


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
    assert len(visibility_json) == 1, f"visibility length is {len(visibility_json)}, expected 1"
    for visibility in visibility_json:
        assert visibility["token"], "token is empty"
        assert visibility["level"] == "none"
        assert visibility["description"], "description is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_directory_structure(t4_dataset_path):
    dir_files = os.listdir(t4_dataset_path)
    assert "annotation" in dir_files, "annotation is not in t4_dataset"
    assert "data" in dir_files, "data is not in t4_dataset"
    assert "input_bag" in dir_files, "input_bag is not in t4_dataset"
    assert "status.json" in dir_files, "status.json is not in t4_dataset"

    intput_bag_files = os.listdir(osp.join(t4_dataset_path, "input_bag"))
    assert "metadata.yaml" in intput_bag_files, "metadata.yaml is not in input_bag"
    assert (
        "tracking_sim_sample_data_0.db3" in intput_bag_files
    ), "tracking_sim_sample_data_0.db3 is not in input_bag"

    topic_count_dict = get_topic_count(osp.join(t4_dataset_path, "input_bag"))
    assert (
        "/localization/kinematic_state" in topic_count_dict.keys()
    ), "kinematic_state is not in input_bag"
    assert (
        "/perception/object_recognition/detection/objects" in topic_count_dict.keys()
    ), "object_recognition/detection/objects is not in input_bag"
    assert "/tf" in topic_count_dict.keys(), "tf is not in input_bag"
    assert "/tf_static" in topic_count_dict.keys(), "tf_static is not in input_bag"
