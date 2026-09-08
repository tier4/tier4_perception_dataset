from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from perception_dataset.kognic.t4_to_kognic_converter import T4ToKognicConverter
from perception_dataset.utils.misc import validate_annotation_hz


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


def test_duplicate_camera_timestamps_keep_one_file_per_frame(tmp_path: Path):
    """Test that repeated camera timestamps still create distinct destinations.

    Three frame records share the same source timestamp, as can happen around
    dropped camera frames. The converter must add one-nanosecond offsets so the
    files remain ordered and no frame overwrites another on disk.

    Args:
        tmp_path (Path): Pytest directory used for the temporary image and
            staging output.
    """
    source = tmp_path / "input/data/CAM_FRONT/image.jpg"
    source.parent.mkdir(parents=True)
    source.touch()
    output = tmp_path / "output"
    records = [
        SimpleNamespace(timestamp=123, filename="data/CAM_FRONT/image.jpg")
        for _ in range(3)
    ]
    converter = T4ToKognicConverter(
        input_base=str(tmp_path / "input"),
        output_base=str(output),
        camera_sensors=[{"channel": "CAM_FRONT"}],
    )
    converter._channel_to_token = {"CAM_FRONT": "camera-token"}
    converter._sample_data_by_channel = {"CAM_FRONT": records}
    converter._frame_records = [{"CAM_FRONT": record} for record in records]

    copies = converter._collect_image_copies(
        tmp_path / "input", output, "CAM_FRONT"
    )

    assert [destination.name for _, destination in copies] == [
        "123000.jpg",
        "123001.jpg",
        "123002.jpg",
    ]
