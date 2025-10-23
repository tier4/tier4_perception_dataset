import json
import os.path as osp
from pathlib import Path
import shutil

from nuscenes.nuscenes import NuScenes
import pytest
import yaml

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_t4_converter import Rosbag2ToT4Converter
from perception_dataset.t4_dataset.data_validator import validate_data_hz
from perception_dataset.t4_dataset.format_validator import (
    validate_directory_structure,
    validate_format,
)
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_ROOT_DIR


@pytest.fixture
def t4_dataset_path():
    """
    Test fixture to provide path to converted t4_dataset from rosbag2 mcap file.
    For rest of .db3 files check ../test_t4_dataset_conversion.py.
    """
    # before test - convert rosbag2 to t4
    test_rosbag_name = "sample_bag"

    with open(TEST_CONFIG_ROOT_DIR / "convert_synthetic_data.yaml") as f:
        param_args = yaml.safe_load(f)

    input_rosbag_base = osp.join(TEST_ROOT_DIR, param_args["conversion"]["input_base"])
    r2t4_output_base = osp.join(TEST_ROOT_DIR, param_args["conversion"]["output_base"])

    param_args["conversion"]["input_base"] = input_rosbag_base
    param_args["conversion"]["output_base"] = r2t4_output_base

    converter_params = Rosbag2ConverterParams(
        task=param_args["task"], overwrite_mode=True, **param_args["conversion"]
    )
    converter = Rosbag2ToT4Converter(converter_params)
    converter.convert()

    # provide a path to converted t4_dataset
    yield Path(r2t4_output_base) / test_rosbag_name

    # after test - remove resource
    shutil.rmtree(r2t4_output_base)


@pytest.fixture
def sample_annotation(t4_dataset_path):
    with (t4_dataset_path / "annotation/sample_annotation.json").open() as f:
        sample_annotation = json.load(f)
    return sample_annotation


def test_t4_dataset_format(t4_dataset_path):
    validate_directory_structure(t4_dataset_path)

    nusc = NuScenes(
        version=T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value,
        dataroot=t4_dataset_path,
        verbose=False,
    )

    validate_format(nusc, t4_dataset_path)
    validate_data_hz(nusc)
