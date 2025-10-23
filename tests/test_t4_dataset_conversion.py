import json
import os.path as osp
from pathlib import Path
import shutil

from nuscenes.nuscenes import NuScenes
import pytest
import yaml

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME
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
from tests.utils.check_equality import diff_check_folder, diff_check_t4_dataset

# Test constants
TEST_ROSBAG_NAME = "sample_bag"


@pytest.fixture(scope="module")
def non_annotated_t4_dataset_path():
    """Convert rosbag2 to non-annotated T4 dataset."""
    # Load configuration
    with open(TEST_CONFIG_ROOT_DIR / "convert_rosbag2_to_non_annotated_t4.yaml") as f:
        param_args = yaml.safe_load(f)

    input_rosbag_base = osp.join(TEST_ROOT_DIR, param_args["conversion"]["input_base"])
    r2t4_output_base = osp.join(TEST_ROOT_DIR, param_args["conversion"]["output_base"])

    param_args["conversion"]["input_base"] = input_rosbag_base
    param_args["conversion"]["output_base"] = r2t4_output_base
    assert osp.exists(input_rosbag_base), f"input_base doesn't exist: {input_rosbag_base}"

    # Convert rosbag2 to non-annotated T4
    converter_params = Rosbag2ConverterParams(
        task=param_args["task"],
        scene_description=param_args["description"]["scene"],
        overwrite_mode=True,
        without_compress=True,
        **param_args["conversion"],
    )
    converter = Rosbag2ToNonAnnotatedT4Converter(converter_params)
    converter.convert()

    # Return the output path
    yield osp.join(r2t4_output_base, TEST_ROSBAG_NAME)

    # Cleanup
    shutil.rmtree(r2t4_output_base, ignore_errors=True)


@pytest.fixture(scope="module")
def deepen_path(non_annotated_t4_dataset_path):
    """Convert non-annotated T4 to Deepen format."""
    # Load configuration
    with open(TEST_CONFIG_ROOT_DIR / "convert_non_annotated_t4_to_deepen.yaml") as f:
        config_dict = yaml.safe_load(f)

    t42d_input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    t42d_output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])
    camera_sensors = config_dict["conversion"]["camera_sensors"]
    annotation_hz = config_dict["conversion"]["annotation_hz"]
    workers_number = config_dict["conversion"]["workers_number"]

    # Convert non-annotated T4 to Deepen
    converter = NonAnnotatedT4ToDeepenConverter(
        input_base=t42d_input_base,
        output_base=t42d_output_base,
        camera_sensors=camera_sensors,
        annotation_hz=annotation_hz,
        workers_number=workers_number,
    )
    converter.convert()

    # Return the output path
    yield osp.join(t42d_output_base, TEST_ROSBAG_NAME)

    # Cleanup
    shutil.rmtree(osp.join(t42d_output_base), ignore_errors=True)


@pytest.mark.parametrize("t4_dataset_path", [True, False], indirect=True)
def test_t4_dataset_validation(t4_dataset_path):
    validate_directory_structure(t4_dataset_path)

    nusc = NuScenes(
        version=T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value,
        dataroot=t4_dataset_path,
        verbose=False,
    )

    validate_format(nusc, t4_dataset_path)
    validate_data_hz(nusc)


@pytest.fixture(scope="module")
def t4_dataset_path(request, deepen_path):
    """Convert Deepen format to T4 dataset."""
    # Load configuration
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

    # Convert Deepen to T4
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

    # Return the T4 dataset path
    yield osp.join(d2t4_output_base, TEST_ROSBAG_NAME, "t4_dataset")

    # Cleanup
    shutil.rmtree(d2t4_output_base, ignore_errors=True)


def load_json(t4_dataset_path, file_name):
    with open(osp.join(t4_dataset_path, "annotation", f"{file_name}.json")) as f:
        return json.load(f)


def test_non_annotated_t4_dataset_diff(non_annotated_t4_dataset_path):
    """Test that generated non-annotated T4 dataset matches expected output."""
    generated_path = Path(non_annotated_t4_dataset_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))

    diff_check_t4_dataset(generated_path, expected_path)


def test_deepen_dataset_diff(deepen_path):
    """Test that generated Deepen dataset matches expected output."""
    generated_path = Path(deepen_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))

    diff_check_folder(generated_path, expected_path)


@pytest.mark.parametrize("t4_dataset_path", [True, False], indirect=True)
def test_t4_dataset_diff(t4_dataset_path):
    """Test that generated T4 dataset matches expected output."""
    generated_path = Path(t4_dataset_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))

    diff_check_t4_dataset(generated_path, expected_path)
