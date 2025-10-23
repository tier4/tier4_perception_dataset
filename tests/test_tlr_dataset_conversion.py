import json
import os
import os.path as osp
from pathlib import Path
import shutil

import pytest
import yaml

from perception_dataset.constants import SENSOR_ENUM
from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter
from perception_dataset.deepen.non_annotated_t4_tlr_to_deepen_converter import (
    NonAnnotatedT4TlrToDeepenConverter,
)
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import (
    Rosbag2ToNonAnnotatedT4Converter,
)
from perception_dataset.utils.rosbag2 import get_topic_count
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR
from tests.utils.check_equality import diff_check_folder, diff_check_t4_dataset

# Downloaded rosbag name
TEST_ROSBAG_NAME = "sample_bag"


@pytest.fixture(scope="module")
def non_annotated_t4_dataset_path():
    # before test - convert rosbag2 to t4
    with open(TEST_CONFIG_ROOT_DIR / "convert_rosbag2_to_non_annotated_t4_tlr_test.yaml") as f:
        param_args = yaml.safe_load(f)

    input_rosbag_base = osp.join(TEST_ROOT_DIR, param_args["conversion"]["input_base"])
    r2t4_output_base = osp.join(TEST_ROOT_DIR, param_args["conversion"]["output_base"])

    param_args["conversion"]["input_base"] = input_rosbag_base
    param_args["conversion"]["output_base"] = r2t4_output_base
    assert osp.exists(input_rosbag_base), f"input_base doesn't exist: {input_rosbag_base}"

    converter_params = Rosbag2ConverterParams(
        task=param_args["task"],
        scene_description=param_args["description"]["scene"],
        overwrite_mode=True,
        without_compress=True,
        **param_args["conversion"],
    )
    converter = Rosbag2ToNonAnnotatedT4Converter(converter_params)
    converter.convert()

    # provide a path to converted non-annotated t4 dataset
    yield osp.join(r2t4_output_base, TEST_ROSBAG_NAME)

    # after test - remove resource
    shutil.rmtree(r2t4_output_base, ignore_errors=True)


@pytest.fixture(scope="module")
def t4_dataset_path(non_annotated_t4_dataset_path):
    # before test - convert deepen to t4
    with open(TEST_CONFIG_ROOT_DIR / "convert_deepen_to_t4_tlr_test.yaml") as f:
        config_dict = yaml.safe_load(f)

    d2t4_input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    input_anno_file = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_anno_file"])
    d2t4_output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])
    dataset_corresponding = config_dict["conversion"]["dataset_corresponding"]
    description = config_dict["description"]
    input_bag_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_bag_base"])
    topic_list_yaml_path = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["topic_list"])
    ignore_interpolate_label = True
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
    yield osp.join(d2t4_output_base, TEST_ROSBAG_NAME, "t4_dataset")

    # after test - remove resource
    shutil.rmtree(d2t4_output_base, ignore_errors=True)


@pytest.fixture(scope="module")
def deepen_dataset_path(non_annotated_t4_dataset_path):
    # before test - convert deepen to t4
    with open(TEST_CONFIG_ROOT_DIR / "convert_non_annotated_t4_tlr_to_deepen.yaml") as f:
        config_dict = yaml.safe_load(f)

    t4_to_deepen_input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    t4_to_deepen_output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])

    converter = NonAnnotatedT4TlrToDeepenConverter(
        input_base=t4_to_deepen_input_base,
        output_base=t4_to_deepen_output_base,
    )
    converter.convert()

    # provide a path to converted t4_dataset
    yield osp.join(t4_to_deepen_output_base, TEST_ROSBAG_NAME)

    # after test - remove resource
    shutil.rmtree(t4_to_deepen_output_base, ignore_errors=True)


def test_non_annotated_t4_tlr_dataset_diff(non_annotated_t4_dataset_path):
    """Test that generated non-annotated T4 TLR dataset matches expected output."""
    generated_path = Path(non_annotated_t4_dataset_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))

    diff_check_t4_dataset(generated_path, expected_path)


def test_t4_tlr_dataset_diff(t4_dataset_path):
    """Test that generated T4 TLR dataset matches expected output."""
    generated_path = Path(t4_dataset_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))

    diff_check_t4_dataset(generated_path, expected_path)


def test_deepen_tlr_dataset_diff(deepen_dataset_path):
    """Test that generated Deepen TLR dataset matches expected output."""
    generated_path = Path(deepen_dataset_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))

    diff_check_folder(generated_path, expected_path)
