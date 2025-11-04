import json
import os.path as osp
from pathlib import Path
import shutil

import pytest
from t4_devkit import Tier4
import yaml

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.deepen.annotated_t4_to_deepen_converter import (
    AnnotatedT4ToDeepenConverter,
)
from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter
from perception_dataset.deepen.non_annotated_t4_to_deepen_converter import (
    NonAnnotatedT4ToDeepenConverter,
)
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import (
    Rosbag2ToNonAnnotatedT4Converter,
)
from perception_dataset.t4_dataset.data_interpolator import DataInterpolator
from perception_dataset.t4_dataset.data_validator import validate_data_hz
from perception_dataset.t4_dataset.format_validator import (
    validate_directory_structure,
    validate_format,
)
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR
from tests.utils.check_equality import (
    diff_check_folder,
    diff_check_json_files,
    diff_check_t4_dataset,
)

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
def deepen_dataset_from_annotated_t4_path(t4_dataset_path):
    # before test - convert annotated t4 to deepen label files
    with open(TEST_CONFIG_ROOT_DIR / "convert_annotated_t4_to_deepen_sample.yaml") as f:
        config_dict = yaml.safe_load(f)

    t4_to_deepen_input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    t4_to_deepen_output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])

    converter = AnnotatedT4ToDeepenConverter(
        input_base=t4_to_deepen_input_base,
        output_base=t4_to_deepen_output_base,
        camera_position=config_dict["conversion"]["camera_position"],
    )
    converter.convert()
    # provide a path to converted deepen format
    yield osp.join(t4_to_deepen_output_base, TEST_ROSBAG_NAME + EXTENSION_ENUM.JSON.value)

    # after test - remove resource
    shutil.rmtree(t4_to_deepen_output_base, ignore_errors=True)


@pytest.fixture(scope="module")
def t4_dataset_post_interpolated_path(t4_dataset_path):
    """Interpolate T4 dataset and return the path to the interpolated dataset."""
    # before test - convert t4 to interpolated t4
    with open(TEST_CONFIG_ROOT_DIR / "interpolate_t4.yaml") as f:
        config_dict = yaml.safe_load(f)

    t4_input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    t4_interpolated_output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])

    interpolator = DataInterpolator(
        input_base=t4_input_base,
        output_base=t4_interpolated_output_base,
        copy_excludes=config_dict["conversion"]["copy_excludes"],
    )
    interpolator.convert()

    # Return the path to the interpolated T4 dataset
    yield osp.join(t4_interpolated_output_base, "t4_dataset")

    # Cleanup
    shutil.rmtree(t4_interpolated_output_base, ignore_errors=True)


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

    t4_dataset = Tier4(
        data_root=t4_dataset_path,
        verbose=False,
    )

    validate_format(t4_dataset, t4_dataset_path)
    validate_data_hz(t4_dataset)


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
    expected_path = Path(non_annotated_t4_dataset_path.replace("_generated", ""))

    diff_check_t4_dataset(generated_path, expected_path)


def test_deepen_dataset_diff(deepen_path):
    """Test that generated Deepen dataset matches expected output."""
    generated_path = Path(deepen_path)
    expected_path = Path(deepen_path.replace("_generated", ""))

    diff_check_folder(generated_path, expected_path)


@pytest.mark.parametrize("t4_dataset_path", [True, False], indirect=True)
def test_t4_dataset_diff(t4_dataset_path):
    """Test that generated T4 dataset matches expected output."""
    generated_path = Path(t4_dataset_path)
    expected_path = Path(t4_dataset_path.replace("_generated", ""))

    diff_check_t4_dataset(generated_path, expected_path)


@pytest.mark.parametrize("t4_dataset_path", [False], indirect=True)
def test_t4_dataset_interpolator(t4_dataset_post_interpolated_path):
    """Test that generated interpolated T4 dataset matches expected output."""
    generated_path = Path(t4_dataset_post_interpolated_path)
    expected_path = Path(t4_dataset_post_interpolated_path.replace("_generated", ""))
    diff_check_t4_dataset(generated_path, expected_path)


@pytest.mark.parametrize("t4_dataset_path", [False], indirect=True)
def test_deepen_from_annotated_dataset_diff(deepen_dataset_from_annotated_t4_path):
    """Test that Deepen labels generated from annotated T4 dataset matches expected output."""
    generated_path = Path(deepen_dataset_from_annotated_t4_path)
    expected_path = Path(deepen_dataset_from_annotated_t4_path.replace("_generated", ""))
    diff_check_json_files(generated_path, expected_path)
