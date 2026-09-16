import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from perception_dataset.constants import LIDAR_CONCAT_CHANNEL
from perception_dataset.kognic.t4_to_kognic_converter import T4ToKognicConverter
from perception_dataset.utils.misc import get_annotation_step, validate_annotation_hz
from perception_dataset.utils.pointcloud import save_pointcloud_csv


@pytest.mark.parametrize("value", [0, -1, 11, 1.0, "1", None, True])
def test_annotation_hz_rejects_unrepresentable_values(value):
    """Test that unsupported annotation frequencies are rejected.

    The cases include values outside 1-10 Hz, non-integers, missing input, and
    ``True`` (which Python otherwise treats as integer 1). Rejecting them keeps
    keyframe selection from dividing by zero or using a misleading frame rate.

    Args:
        value: Invalid frequency supplied by the parametrized test.
    """
    with pytest.raises(ValueError, match="annotation_hz must be an integer"):
        validate_annotation_hz(value)


@pytest.mark.parametrize("value", [1, 2, 5, 10])
def test_annotation_hz_accepts_supported_integer_values(value: int):
    """Test that supported whole-number annotation frequencies are accepted.

    These values fit within the 10 Hz T4 sample rate and can be represented by
    the converter's frame-selection logic.

    Args:
        value (int): Supported frequency supplied by the parametrized test.
    """
    assert validate_annotation_hz(value) == value


def test_annotation_step_rounds_non_divisor_frequency():
    """Use a rounded stride consistently for Kognic and Deepen sampling."""
    assert get_annotation_step(6) == 2


def test_scene_conversion_removes_stale_sensor_directories_before_generation(tmp_path: Path):
    """Test that conversion removes stale camera and LiDAR output first.

    The conversion is deliberately stopped immediately after cleanup. This
    proves files left by an earlier run are deleted before new files are
    generated, preventing sorted file positions from shifting frame pairing.

    Args:
        tmp_path (Path): Pytest directory used for the temporary source and
            output scene.
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    stale_camera = output_dir / "cameras/CAM_FRONT/stale.jpg"
    stale_lidar = output_dir / "lidar/LIDAR_FRONT/stale.csv"
    stale_camera.parent.mkdir(parents=True)
    stale_lidar.parent.mkdir(parents=True)
    stale_camera.touch()
    stale_lidar.touch()
    converter = T4ToKognicConverter(
        input_base=str(input_dir),
        output_base=str(output_dir),
        camera_sensors=[],
    )
    converter._build_lookup_maps = Mock(side_effect=RuntimeError("stop after cleanup"))

    with pytest.raises(RuntimeError, match="stop after cleanup"):
        converter._convert_one_scene(input_dir, output_dir)

    assert not (output_dir / "cameras").exists()
    assert not (output_dir / "lidar").exists()


@pytest.mark.parametrize(
    ("info_filename", "create_info_directory", "expected"),
    [
        ("metadata/concat.json", False, True),
        (None, True, False),
    ],
)
def test_lookup_maps_detect_lidar_concat_info_from_sample_data(
    tmp_path: Path,
    info_filename: str | None,
    create_info_directory: bool,
    expected: bool,
):
    """Test that concat metadata detection follows the loaded sample-data record."""
    if create_info_directory:
        (tmp_path / "data/LIDAR_CONCAT_INFO").mkdir(parents=True)

    tables = {
        "sensor": [SimpleNamespace(token="lidar-token", channel=LIDAR_CONCAT_CHANNEL)],
        "calibrated_sensor": [],
        "sample": [],
        "sample_data": [
            SimpleNamespace(
                channel=LIDAR_CONCAT_CHANNEL,
                filename="data/LIDAR_CONCAT/0.pcd.bin",
                info_filename=info_filename,
                timestamp=0,
            )
        ],
        "ego_pose": [],
    }
    t4 = Mock()
    t4.get_table.side_effect = tables.__getitem__
    converter = T4ToKognicConverter(
        input_base=str(tmp_path),
        output_base=str(tmp_path / "output"),
        camera_sensors=[],
        annotated=False,
    )

    with patch(
        "perception_dataset.kognic.t4_to_kognic_converter.Tier4",
        return_value=t4,
    ):
        converter._build_lookup_maps(tmp_path)

    assert converter._has_lidar_concat_info is expected


def test_duplicate_camera_timestamps_raise_error(tmp_path: Path):
    """Test that repeated camera timestamps fail before overwriting a destination.

    Three frame records share the same source timestamp, as can happen around
    dropped camera frames. Timestamp-based Kognic filenames cannot represent
    those frames uniquely, so conversion must fail explicitly.

    Args:
        tmp_path (Path): Pytest directory used for the temporary image and
            staging output.
    """
    source = tmp_path / "input/data/CAM_FRONT/image.jpg"
    source.parent.mkdir(parents=True)
    source.touch()
    output = tmp_path / "output"
    records = [
        SimpleNamespace(timestamp=123, filename="data/CAM_FRONT/image.jpg") for _ in range(3)
    ]
    converter = T4ToKognicConverter(
        input_base=str(tmp_path / "input"),
        output_base=str(output),
        camera_sensors=[{"channel": "CAM_FRONT"}],
    )
    converter._channel_to_token = {"CAM_FRONT": "camera-token"}
    converter._sample_data_by_channel = {"CAM_FRONT": records}
    converter._frame_records = [{"CAM_FRONT": record} for record in records]

    with pytest.raises(ValueError, match="CAM_FRONT.*duplicate timestamp 123000 ns"):
        converter._collect_image_copies(tmp_path / "input", output, "CAM_FRONT")


def test_annotated_keyframes_follow_samples_with_annotations(tmp_path: Path):
    """Test that annotated conversion selects only samples containing objects."""
    converter = object.__new__(T4ToKognicConverter)
    converter._annotated = True
    converter._anchor_channel = "LIDAR_TOP"
    converter._annotated_sample_tokens = {"annotated-sample"}
    converter._frame_records = [
        {"LIDAR_TOP": SimpleNamespace(sample_token="annotated-sample")},
        {"LIDAR_TOP": SimpleNamespace(sample_token="empty-sample")},
    ]

    converter._write_keyframes(tmp_path)
    payload = json.loads((tmp_path / "keyframes.json").read_text())

    assert payload["keyframe_indices"] == [0]


def test_save_pointcloud_csv_writes_timestamp_and_first_four_features(tmp_path: Path):
    """Test point-cloud CSV formatting and omission of extra point features."""
    csv_path = tmp_path / "points.csv"
    points = np.array([[1.25, -2.0, 3.125, 4.5, 99.0]], dtype=np.float32)

    save_pointcloud_csv(csv_path, 123456789, points)

    assert csv_path.read_text() == (
        "ts_gps,x,y,z,intensity\n"
        "123456789,1.250000,-2.000000,3.125000,4.500000\n"
    )


def test_save_pointcloud_csv_writes_header_for_empty_cloud(tmp_path: Path):
    """Test that an empty point cloud still produces a valid CSV header."""
    csv_path = tmp_path / "points.csv"

    save_pointcloud_csv(csv_path, 123456789, np.empty((0, 4), dtype=np.float32))

    assert csv_path.read_text() == "ts_gps,x,y,z,intensity\n"
