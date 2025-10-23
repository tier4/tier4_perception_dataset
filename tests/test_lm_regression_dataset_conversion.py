import os.path as osp
from pathlib import Path
import shutil

import pytest
import yaml

from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_t4_loc_converter import Rosbag2ToT4LocConverter
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR
from tests.utils.check_equality import diff_check_t4_dataset


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


def test_lm_regression_dataset_diff(t4_dataset_path):
    """Test that generated LM regression dataset matches expected output."""
    generated_path = Path(t4_dataset_path)
    expected_path = Path(str(generated_path).replace("_generated", ""))
    
    diff_check_t4_dataset(generated_path, expected_path)

