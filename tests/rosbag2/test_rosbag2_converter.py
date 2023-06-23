import json
import shutil

from nuscenes.nuscenes import NuScenes
import pandas as pd
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
from tests.constants import TEST_CONFIG_ROOT_DIR, TEST_DATA_ROOT_DIR


@pytest.fixture
def t4_dataset_path(request):
    """
    Provide data with selected storage id with `request.param` parameter, convert it
    using a config file, yield the path to tests and clean up at the end.

    To find more info about how request param works with pytest.fixtures please refet to:
    https://docs.pytest.org/en/latest/example/parametrize.html#indirect-parametrization
    """
    # before test - convert rosbag2 to t4
    rosbag_dir = "awsim_rosbag" if request.param == "db3" else "awsim_rosbag_mcap"
    input_base = TEST_DATA_ROOT_DIR / rosbag_dir
    output_base = TEST_DATA_ROOT_DIR / "t4_dataset"
    test_rosbag_name = "x2_nishi_shinjuku_dynamic_object_msg"

    assert input_base.exists()

    with open(TEST_CONFIG_ROOT_DIR / "convert_synthetic_data.yaml") as f:
        param_args = yaml.safe_load(f)

    param_args["conversion"]["input_base"] = str(input_base)
    param_args["conversion"]["output_base"] = str(output_base)

    converter_params = Rosbag2ConverterParams(
        task=param_args["task"], overwrite_mode=True, **param_args["conversion"]
    )
    converter = Rosbag2ToT4Converter(converter_params)
    converter.convert()

    # provide a path to converted t4_dataset
    yield output_base / test_rosbag_name

    # after test - remove resource
    shutil.rmtree(output_base)


@pytest.fixture
def sample_annotation(t4_dataset_path):
    with (t4_dataset_path / "annotation/sample_annotation.json").open() as f:
        sample_annotation = json.load(f)
    return sample_annotation


@pytest.mark.parametrize("t4_dataset_path", ["db3", "mcap"], indirect=True)
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


@pytest.mark.parametrize("t4_dataset_path", ["db3", "mcap"], indirect=True)
def test_rosbag2_converter_dataset_consistency(sample_annotation):
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


expected_num_lidar_pts = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    139,
    12,
    13,
    14,
    11,
    16,
    17,
    15,
    19,
    18,
    21,
    22,
    23,
    152,
    25,
    26,
    27,
    28,
    29,
    24,
    31,
    32,
    33,
    34,
    154,
    36,
    37,
    549,
    38,
    40,
    41,
    42,
    550,
    44,
    45,
    39,
    57,
    570,
    59,
    62,
    65,
    69,
    70,
    71,
    75,
    78,
    80,
    81,
    84,
    86,
    88,
    89,
    219,
    92,
    97,
    227,
    231,
    4456,
    236,
    539,
}


@pytest.mark.parametrize("t4_dataset_path", ["db3", "mcap"], indirect=True)
def test_rosbag2_converter_num_lidar_pts(sample_annotation):
    num_lidar_pts_list = [r["num_lidar_pts"] for r in sample_annotation]
    assert expected_num_lidar_pts.difference(set(num_lidar_pts_list)) == set()
