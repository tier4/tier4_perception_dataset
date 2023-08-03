import json
import os
import os.path as osp
import shutil

from nuscenes.nuscenes import NuScenes
import pandas as pd
import pytest
import yaml

from perception_dataset.constants import SENSOR_ENUM, T4_FORMAT_DIRECTORY_NAME
from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter
from perception_dataset.deepen.non_annotated_t4_to_deepen_converter import (
    NonAnnotatedT4ToDeepenConverter,
)
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import (
    Rosbag2ToNonAnnotatedT4Converter,
)
from perception_dataset.t4_dataset.data_validator import validate_data_hz
from perception_dataset.t4_dataset.format_validator import (
    validate_directory_structure,
    validate_format,
)
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR


@pytest.fixture(scope="module")
def t4_dataset_path(request):
    test_rosbag_name = "sample_bag"
    # before test - convert rosbag2 to t4
    with open(TEST_CONFIG_ROOT_DIR / "convert_rosbag2_to_non_annotated_t4.yaml") as f:
        param_args = yaml.safe_load(f)

    input_rosbag_base = osp.join(TEST_ROOT_DIR, param_args["conversion"]["input_base"])
    r2t4_output_base = osp.join(TEST_ROOT_DIR, param_args["conversion"]["output_base"])

    param_args["conversion"]["input_base"] = input_rosbag_base
    param_args["conversion"]["output_base"] = r2t4_output_base
    assert osp.exists(input_rosbag_base), f"input_base doesn't exist: {input_rosbag_base}"

    converter_params = Rosbag2ConverterParams(
        task=param_args["task"],
        overwrite_mode=True,
        without_compress=True,
        **param_args["conversion"],
    )
    converter = Rosbag2ToNonAnnotatedT4Converter(converter_params)
    converter.convert()

    # before test - convert t4 to deepen
    with open(TEST_CONFIG_ROOT_DIR / "convert_t4_to_deepen.yaml") as f:
        config_dict = yaml.safe_load(f)
    t42d_input_base = config_dict["conversion"]["input_base"]
    t42d_output_base = config_dict["conversion"]["output_base"]
    camera_sensors = config_dict["conversion"]["camera_sensors"]
    annotation_hz = config_dict["conversion"]["annotation_hz"]
    workers_number = config_dict["conversion"]["workers_number"]

    converter = NonAnnotatedT4ToDeepenConverter(
        input_base=t42d_input_base,
        output_base=t42d_output_base,
        camera_sensors=camera_sensors,
        annotation_hz=annotation_hz,
        workers_number=workers_number,
    )

    converter.convert()

    # before test - convert deepen to t4
    with open(TEST_CONFIG_ROOT_DIR / "convert_deepen_to_t4.yaml") as f:
        config_dict = yaml.safe_load(f)

    d2t4_input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    input_anno_file = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_anno_file"])
    d2t4_output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])
    dataset_corresponding = config_dict["conversion"]["dataset_corresponding"]
    description = config_dict["description"]
    input_bag_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_bag_base"])
    topic_list_yaml_path = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["topic_list"])
    ignore_interpolate_label = request.param
    if ignore_interpolate_label:
        d2t4_output_base = d2t4_output_base + "_wo_interpolate_label"
    with open(topic_list_yaml_path) as f:
        topic_list_yaml = yaml.safe_load(f)

    converter = DeepenToT4Converter(
        input_base=d2t4_input_base,
        output_base=d2t4_output_base,
        input_anno_file=input_anno_file,
        dataset_corresponding=dataset_corresponding,
        overwrite_mode=True,
        description=description,
        input_bag_base=input_bag_base,
        topic_list=topic_list_yaml,
        ignore_interpolate_label=ignore_interpolate_label,
    )
    converter.convert()

    # provide a path to converted t4_dataset
    yield osp.join(d2t4_output_base, test_rosbag_name, "t4_dataset")

    # after test - remove resource
    shutil.rmtree(r2t4_output_base, ignore_errors=True)
    shutil.rmtree(osp.join(t42d_output_base, test_rosbag_name), ignore_errors=True)
    shutil.rmtree(d2t4_output_base, ignore_errors=True)


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
    assert len(attribute_json) == 9, f"attribute_json length is {len(attribute_json)}, expected 9"
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
    assert len(calibrated_sensor) == 7


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_category_json(t4_dataset_path, category_list):
    category_json = load_json(t4_dataset_path, "category")
    assert len(category_json) == 3, f"category length is {len(category_json)}, expected 3"
    for category in category_json:
        assert category["name"], "name is empty"
        assert category["token"], "token is empty"
        assert category["name"] in category_list, f"{category['name']} is not in {category_list}"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_ego_pose_json(t4_dataset_path):
    ego_pose_json = load_json(t4_dataset_path, "ego_pose")
    assert len(ego_pose_json) == 197, f"ego_pose length is {len(ego_pose_json)}, expected 197"
    for ego_pose in ego_pose_json:
        assert ego_pose["translation"], "translation is empty"
        assert ego_pose["rotation"], "rotation is empty"
        assert ego_pose["token"], "token is empty"
        assert ego_pose["timestamp"], "timestamp is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_instance_json(t4_dataset_path):
    instance_json = load_json(t4_dataset_path, "instance")
    assert len(instance_json) == 3, f"instance length is {len(instance_json)}, expected 3"
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
    assert len(object_ann_json) == 25, f"object_ann length is {len(object_ann_json)}, expected 25"
    for object_ann in object_ann_json:
        assert object_ann["token"], "token is empty"
        assert object_ann["sample_data_token"], "sample_data_token is empty"
        assert object_ann["instance_token"], "instance_token is empty"
        assert object_ann["category_token"], "category_token is empty"
        assert object_ann["attribute_tokens"], "attribute_tokens is empty"
        assert object_ann["bbox"], "bbox is empty"
        assert object_ann["mask"], "mask is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_annotation_json(t4_dataset_path):
    sample_annotation = load_json(t4_dataset_path, "sample_annotation")
    assert (
        len(sample_annotation) == 30
    ), f"sample_annotation length is {len(sample_annotation)}, expected 30"
    for sample_anno in sample_annotation:
        assert sample_anno["token"], "token is empty"
        assert sample_anno["sample_token"], "sample_token is empty"
        assert sample_anno["instance_token"], "instance_token is empty"

        assert sample_anno["attribute_tokens"], "attribute_tokens is empty"
        assert sample_anno["visibility_token"], "visibility_token is empty"
        assert sample_anno["translation"], "translation is empty"
        assert sample_anno["size"], "size is empty"
        assert sample_anno["rotation"], "rotation is empty"
        assert sample_anno["num_lidar_pts"] >= 0, "num_lidar_pts is empty"
        assert sample_anno["num_radar_pts"] >= 0, "num_radar_pts is empty"


@pytest.mark.parametrize("t4_dataset_path", [False], indirect=True)
def test_sample_annotation_json_with_interpolate_label(t4_dataset_path):
    sample_annotation = load_json(t4_dataset_path, "sample_annotation")
    assert len(sample_annotation) == 56
    for sample_anno in sample_annotation:
        assert sample_anno["token"], "token is empty"
        assert sample_anno["sample_token"], "sample_token is empty"
        assert sample_anno["instance_token"], "instance_token is empty"

        assert sample_anno["attribute_tokens"], "attribute_tokens is empty"
        assert sample_anno["visibility_token"], "visibility_token is empty"
        assert sample_anno["translation"], "translation is empty"
        assert sample_anno["size"], "size is empty"
        assert sample_anno["rotation"], "rotation is empty"
        assert sample_anno["num_lidar_pts"] >= 0, "num_lidar_pts is empty"
        assert sample_anno["num_radar_pts"] >= 0, "num_radar_pts is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_data_json(t4_dataset_path):
    sample_data_json = load_json(t4_dataset_path, "sample_data")
    assert (
        len(sample_data_json) == 196
    ), f"sample_data length is {len(sample_data_json)}, expected 196"
    for sample_data in sample_data_json:
        assert sample_data["token"], "token is empty"
        assert sample_data["sample_token"], "sample_token is empty"
        assert sample_data["ego_pose_token"], "instance_token is empty"
        assert sample_data["calibrated_sensor_token"], "calibrated_sensor_token is empty"
        assert sample_data["filename"], "filename is empty"
        assert sample_data["width"] >= 0, "width is empty"
        assert sample_data["height"] >= 0, "height is empty"
        assert (
            sample_data["is_key_frame"] == sample_data["is_valid"]
        ), f"is_key_frame is {sample_data['is_key_frame']}, is_valid is {sample_data['is_valid']}"
        assert "next" in sample_data.keys(), "next is empty"
        assert "prev" in sample_data.keys(), "prev is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sample_json(t4_dataset_path):
    sample_json = load_json(t4_dataset_path, "sample")
    assert len(sample_json) == 28, f"sample length is {len(sample_json)}, expected 28"
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
        assert "description" in scene.keys(), "next is empty"
        assert scene["log_token"], "log_token is empty"
        assert scene["nbr_samples"], "nbr_samples is empty"
        assert scene["first_sample_token"], "first_sample_token is empty"
        assert scene["last_sample_token"], "last_sample_token is empty"


@pytest.mark.parametrize("t4_dataset_path", [True], indirect=True)
def test_sensor_json(t4_dataset_path):
    sensor_json = load_json(t4_dataset_path, "sensor")
    assert len(sensor_json) == 7, f"sensor length is {len(sensor_json)}, expected 7"
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
    assert len(visibility_json) == 4, f"visibility length is {len(visibility_json)}, expected 4"
    for visibility in visibility_json:
        assert visibility["token"], "token is empty"
        assert visibility["level"] in [
            "full",
            "most",
            "partial",
            "none",
        ]
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
    assert "sample_bag_0.db3" in intput_bag_files, ".db3 is not in input_bag"
