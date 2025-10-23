import os.path as osp
from pathlib import Path
import shutil

import pandas as pd
import pytest
import yaml

from perception_dataset.rosbag2.converter_params import DataType, Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_t4_tracking_converter import (
    Rosbag2ToT4TrackingConverter,
)
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR
from tests.utils.check_equality import diff_check_t4_dataset


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


def test_tracking_sim_dataset_diff(t4_dataset_path):
    """Test that generated tracking sim dataset matches expected output."""
    generated_path = Path(t4_dataset_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))

    diff_check_t4_dataset(generated_path, expected_path)
