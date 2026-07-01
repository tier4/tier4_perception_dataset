import os.path as osp
from pathlib import Path
import shutil

import pytest
import yaml

from perception_dataset.kognic.t4_to_kognic_converter import T4ToKognicConverter
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR
from tests.utils.check_equality import diff_check_folder

# Converted scene name (matches the sequence root under the input fixture).
TEST_SCENE_NAME = "sample_bag"


@pytest.fixture(scope="module")
def kognic_dataset_path():
    # before test - convert non-annotated T4 to the Kognic staging layout
    with open(TEST_CONFIG_ROOT_DIR / "convert_non_annotated_t4_to_kognic_test.yaml") as f:
        config_dict = yaml.safe_load(f)

    input_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["input_base"])
    output_base = osp.join(TEST_ROOT_DIR, config_dict["conversion"]["output_base"])
    assert osp.exists(input_base), f"input_base doesn't exist: {input_base}"

    converter = T4ToKognicConverter(
        input_base=input_base,
        output_base=output_base,
        camera_sensors=config_dict["conversion"]["camera_sensors"],
        workers_number=config_dict["conversion"]["workers_number"],
        drop_camera_token_not_found=config_dict["conversion"]["drop_camera_token_not_found"],
    )
    converter.convert()

    # provide a path to the converted Kognic dataset scene
    yield osp.join(output_base, TEST_SCENE_NAME)

    # after test - remove generated resource
    shutil.rmtree(output_base, ignore_errors=True)


def test_kognic_dataset_diff(kognic_dataset_path):
    """Test that generated Kognic dataset matches expected output."""
    generated_path = Path(kognic_dataset_path)
    expected_path = Path(kognic_dataset_path.replace("_generated", ""))

    diff_check_folder(generated_path, expected_path)
