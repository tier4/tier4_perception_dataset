import os.path as osp
from pathlib import Path
import shutil

import pytest
import yaml

from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter import (
    Rosbag2ToAnnotatedT4TlrConverter,
)
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR
from tests.utils.check_equality import diff_check_t4_dataset


@pytest.fixture(scope="module")
def t4_dataset_path():
    test_rosbag_name = "sample_bag"
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


def test_annotated_tlr_dataset_diff(t4_dataset_path):
    """Test that generated annotated T4 TLR dataset matches expected output."""
    generated_path = Path(t4_dataset_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))

    diff_check_t4_dataset(generated_path, expected_path)
