import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CompressedImage

from perception_dataset.rosbag2 import rosbag2_to_non_annotated_t4_converter as converter_module
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import (
    _Rosbag2ToNonAnnotatedT4Converter,
)

_MISSING = object()


def _import_t4_converter():
    module_name = "perception_dataset.rosbag2.rosbag2_to_t4_converter"
    message_module_name = "perception_dataset.rosbag2.autoware_msgs"
    fake_module_names = (
        module_name,
        message_module_name,
        "autoware_auto_perception_msgs",
        "autoware_auto_perception_msgs.msg",
    )
    original_modules = {name: sys.modules.get(name, _MISSING) for name in fake_module_names}
    rosbag2_package = importlib.import_module("perception_dataset.rosbag2")
    original_package_attributes = {
        name: getattr(rosbag2_package, name, _MISSING)
        for name in ("rosbag2_to_t4_converter", "autoware_msgs")
    }

    try:
        return importlib.import_module(module_name)._Rosbag2ToT4Converter
    except ModuleNotFoundError as error:
        if error.name != "autoware_auto_perception_msgs":
            raise

    try:
        # The checked ROS installation no longer provides this legacy message package. The
        # inherited _save_config method does not use its object-message types, so provide only
        # those import-time symbols needed to load the class under test.
        message_module = ModuleType("autoware_auto_perception_msgs.msg")
        for name in (
            "DetectedObject",
            "DetectedObjects",
            "ObjectClassification",
            "TrackedObject",
            "TrackedObjects",
        ):
            setattr(message_module, name, type(name, (), {}))
        package_module = ModuleType("autoware_auto_perception_msgs")
        package_module.msg = message_module
        sys.modules[package_module.__name__] = package_module
        sys.modules[message_module.__name__] = message_module

        return importlib.import_module(module_name)._Rosbag2ToT4Converter
    finally:
        for name, original_module in original_modules.items():
            if original_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
        for name, original_attribute in original_package_attributes.items():
            if original_attribute is _MISSING:
                rosbag2_package.__dict__.pop(name, None)
            else:
                setattr(rosbag2_package, name, original_attribute)


_Rosbag2ToT4Converter = _import_t4_converter()


class _SampleDataTable:
    def __init__(self, filename):
        self._record = SimpleNamespace(filename=filename)

    def insert_into_table(self, **_kwargs):
        return "sample-data-token"

    def get_record_from_token(self, token):
        assert token == "sample-data-token"
        return self._record


@pytest.mark.parametrize(
    ("jpeg_quality", "jpeg_optimize", "legacy_no_params", "expected_params"),
    [
        (95, False, True, None),
        (95, False, False, [int(cv2.IMWRITE_JPEG_QUALITY), 95]),
        (85, False, True, [int(cv2.IMWRITE_JPEG_QUALITY), 85]),
        (
            95,
            True,
            True,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                95,
                int(cv2.IMWRITE_JPEG_OPTIMIZE),
                1,
            ],
        ),
    ],
    ids=(
        "legacy-default-has-no-params",
        "explicit-default-quality",
        "configured-quality",
        "configured-optimization",
    ),
)
def test_write_jpeg_preserves_or_configures_opencv_parameters(
    mocker, jpeg_quality, jpeg_optimize, legacy_no_params, expected_params
):
    converter = object.__new__(_Rosbag2ToNonAnnotatedT4Converter)
    converter._jpeg_quality = jpeg_quality
    converter._jpeg_optimize = jpeg_optimize
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    output_path = "image.jpg"
    imwrite = mocker.patch.object(cv2, "imwrite")

    converter._write_jpeg(output_path, image, legacy_no_params=legacy_no_params)

    args = imwrite.call_args.args
    assert args[0] == output_path
    assert args[1] is image
    if expected_params is None:
        assert len(args) == 2
    else:
        assert len(args) == 3
        assert args[2] == expected_params
    assert imwrite.call_args.kwargs == {}


def _random_image(seed: int = 0):
    return (np.random.default_rng(seed).random((64, 96, 3)) * 255).astype(np.uint8)


def _write_jpeg_with(tmp_path, name, jpeg_quality, jpeg_optimize, *, legacy_no_params):
    converter = object.__new__(_Rosbag2ToNonAnnotatedT4Converter)
    converter._jpeg_quality = jpeg_quality
    converter._jpeg_optimize = jpeg_optimize
    output_path = tmp_path / name
    converter._write_jpeg(str(output_path), _random_image(), legacy_no_params=legacy_no_params)
    return output_path


def test_write_jpeg_settings_reach_the_encoder(tmp_path):
    """Guard against the settings being accepted but never applied to real output."""
    low = _write_jpeg_with(tmp_path, "low.jpg", 1, False, legacy_no_params=False)
    high = _write_jpeg_with(tmp_path, "high.jpg", 95, False, legacy_no_params=False)
    optimized = _write_jpeg_with(tmp_path, "optimized.jpg", 95, True, legacy_no_params=False)

    assert low.stat().st_size < high.stat().st_size
    assert optimized.read_bytes() != high.read_bytes()
    for path in (low, high, optimized):
        assert cv2.imread(str(path)) is not None, f"{path.name} is not a decodable JPEG"


def test_write_jpeg_defaults_stay_byte_identical_to_a_bare_imwrite(tmp_path):
    """Existing datasets must not change: defaults reproduce the pre-change call exactly."""
    legacy_path = tmp_path / "legacy.jpg"
    cv2.imwrite(str(legacy_path), _random_image())

    for name, legacy_no_params in (("new_legacy.jpg", True), ("new_explicit.jpg", False)):
        written = _write_jpeg_with(tmp_path, name, 95, False, legacy_no_params=legacy_no_params)
        assert (
            written.read_bytes() == legacy_path.read_bytes()
        ), f"default output diverged from a bare cv2.imwrite (legacy_no_params={legacy_no_params})"


def _converter_params(**overrides):
    fields = {
        "task": "task",
        "input_base": "input_base",
        "output_base": "output_base",
        "object_msg_type": "DetectedObjects",
        "num_load_frames": 1,
        "skip_timestamp": 1.0,
        "camera_sensors": [{"topic": "/camera"}],
    }
    fields.update(overrides)
    return Rosbag2ConverterParams(**fields)


@pytest.mark.parametrize(
    ("jpeg_quality", "jpeg_optimize"),
    [(95, False), (85, True)],
    ids=("defaults", "configured"),
)
def test_init_wires_jpeg_settings_from_params(mocker, tmp_path, jpeg_quality, jpeg_optimize):
    """Catch the settings being dropped or swapped between params and the converter."""
    for method in (
        "_set_sensors",
        "_make_directories",
        "_calc_actual_num_load_frames",
    ):
        mocker.patch.object(_Rosbag2ToNonAnnotatedT4Converter, method)
    mocker.patch.object(
        _Rosbag2ToNonAnnotatedT4Converter, "_make_file_index_func", return_value=lambda *_a: 0
    )
    mocker.patch.object(
        _Rosbag2ToNonAnnotatedT4Converter, "_make_optional_ins_handler", return_value=None
    )
    mocker.patch.object(converter_module, "Rosbag2Reader")

    converter = _Rosbag2ToNonAnnotatedT4Converter(
        _converter_params(
            input_bag_path=str(tmp_path / "bag"),
            output_base=str(tmp_path / "out"),
            jpeg_quality=jpeg_quality,
            jpeg_optimize=jpeg_optimize,
        )
    )

    assert converter._jpeg_quality == jpeg_quality
    assert converter._jpeg_optimize is jpeg_optimize


@pytest.fixture
def bare_image_converter(tmp_path):
    converter = object.__new__(_Rosbag2ToNonAnnotatedT4Converter)
    converter._generate_ego_pose = lambda _stamp: "ego-pose-token"
    converter._sample_data_table = _SampleDataTable("image.jpg")
    converter._output_scene_dir = str(tmp_path)
    converter._undistort_image = False
    converter.undistort_map_x = np.zeros((2, 3), dtype=np.float32)
    converter.undistort_map_y = np.zeros((2, 3), dtype=np.float32)
    return converter


def _generate_image(converter, image, camera_info=None):
    return converter._generate_image_data(
        image_arr=image,
        image_unix_timestamp=123.456,
        sample_token="sample-token",
        calibrated_sensor_token="calibrated-sensor-token",
        sensor_channel="camera",
        frame_index=0,
        camera_info=camera_info,
    )


def test_generate_image_data_routes_ndarray_to_jpeg_writer(bare_image_converter, mocker):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    write_jpeg = mocker.patch.object(bare_image_converter, "_write_jpeg")
    imwrite = mocker.patch.object(converter_module.cv2, "imwrite")

    sample_data_token = _generate_image(bare_image_converter, image)

    assert sample_data_token == "sample-data-token"
    write_jpeg.assert_called_once()
    args = write_jpeg.call_args.args
    assert args[0] == str(Path(bare_image_converter._output_scene_dir, "image.jpg"))
    assert args[1] is image
    assert write_jpeg.call_args.kwargs == {"legacy_no_params": False}
    imwrite.assert_not_called()


def test_generate_image_data_routes_undistorted_compressed_image_to_jpeg_writer(
    bare_image_converter, mocker
):
    bare_image_converter._undistort_image = True
    compressed_image = CompressedImage()
    decoded_image = np.zeros((2, 3, 3), dtype=np.uint8)
    remapped_image = np.ones((2, 3, 3), dtype=np.uint8)
    decoded = mocker.patch.object(
        converter_module.rosbag2_utils,
        "compressed_msg_to_numpy",
        return_value=decoded_image,
    )
    remap = mocker.patch.object(converter_module.cv2, "remap", return_value=remapped_image)
    write_jpeg = mocker.patch.object(bare_image_converter, "_write_jpeg")
    imwrite = mocker.patch.object(converter_module.cv2, "imwrite")

    _generate_image(bare_image_converter, compressed_image, camera_info=object())

    decoded.assert_called_once_with(compressed_image)
    remap.assert_called_once_with(
        decoded_image,
        bare_image_converter.undistort_map_x,
        bare_image_converter.undistort_map_y,
        cv2.INTER_LINEAR,
    )
    write_jpeg.assert_called_once()
    args = write_jpeg.call_args.args
    assert args[0] == str(Path(bare_image_converter._output_scene_dir, "image.jpg"))
    assert args[1] is remapped_image
    assert write_jpeg.call_args.kwargs == {"legacy_no_params": True}
    imwrite.assert_not_called()


def test_generate_image_data_preserves_compressed_image_bytes(bare_image_converter, mocker):
    compressed_image = CompressedImage()
    compressed_image.data = b"original-compressed-image-bytes"
    write_jpeg = mocker.patch.object(bare_image_converter, "_write_jpeg")
    decoded = mocker.patch.object(converter_module.rosbag2_utils, "compressed_msg_to_numpy")
    remap = mocker.patch.object(converter_module.cv2, "remap")
    imwrite = mocker.patch.object(converter_module.cv2, "imwrite")

    _generate_image(bare_image_converter, compressed_image)

    assert (
        Path(bare_image_converter._output_scene_dir, "image.jpg").read_bytes()
        == b"original-compressed-image-bytes"
    )
    write_jpeg.assert_not_called()
    decoded.assert_not_called()
    remap.assert_not_called()
    imwrite.assert_not_called()


def _bare_config_converter(converter_cls, output_scene_dir, jpeg_quality, jpeg_optimize):
    converter = object.__new__(converter_cls)
    converter._output_scene_dir = str(output_scene_dir)
    converter._skip_timestamp = 0.5
    converter._undistort_image = True
    converter._jpeg_quality = jpeg_quality
    converter._jpeg_optimize = jpeg_optimize
    return converter


@pytest.mark.parametrize(
    "converter_cls",
    [_Rosbag2ToNonAnnotatedT4Converter, _Rosbag2ToT4Converter],
    ids=("base-converter", "inherited-converter"),
)
def test_save_config_omits_default_jpeg_settings_and_preserves_legacy_bytes(
    tmp_path, converter_cls
):
    converter = _bare_config_converter(converter_cls, tmp_path, 95, False)

    converter._save_config()

    assert (tmp_path / "status.json").read_bytes() == (
        b"{\n"
        b'    "rosbag2_to_non_annotated_t4_converter": {\n'
        b'        "_skip_timestamp": 0.5,\n'
        b'        "_undistort_image": true\n'
        b"    }\n"
        b"}"
    )


@pytest.mark.parametrize(
    "converter_cls",
    [_Rosbag2ToNonAnnotatedT4Converter, _Rosbag2ToT4Converter],
    ids=("base-converter", "inherited-converter"),
)
@pytest.mark.parametrize(
    ("jpeg_quality", "jpeg_optimize"),
    [(85, False), (95, True)],
    ids=("quality-85", "optimize"),
)
def test_save_config_records_complete_non_default_jpeg_settings(
    tmp_path, converter_cls, jpeg_quality, jpeg_optimize
):
    converter = _bare_config_converter(converter_cls, tmp_path, jpeg_quality, jpeg_optimize)

    converter._save_config()

    assert json.loads((tmp_path / "status.json").read_text()) == {
        "rosbag2_to_non_annotated_t4_converter": {
            "_skip_timestamp": 0.5,
            "_undistort_image": True,
            "_jpeg_quality": jpeg_quality,
            "_jpeg_optimize": jpeg_optimize,
        }
    }
