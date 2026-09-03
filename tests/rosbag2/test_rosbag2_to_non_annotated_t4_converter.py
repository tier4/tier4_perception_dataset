import importlib
from types import SimpleNamespace

import cv2
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
import numpy as np
import pytest


class _DummySampleDataTable:
    def __init__(self):
        self.last_insert = None

    def insert_into_table(self, **kwargs):
        self.last_insert = kwargs
        return "sample_data_token"

    def get_record_from_token(self, token):
        assert token == "sample_data_token"
        return SimpleNamespace(filename=self.last_insert["filename"])


@pytest.fixture
def converter_module():
    pytest.importorskip("t4_devkit")

    return importlib.import_module(
        "perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter"
    )


@pytest.fixture
def ffmpeg_packet_message():
    ffmpeg_msg = FFMPEGPacket()
    ffmpeg_msg.data = b"ffmpeg-packet"
    return ffmpeg_msg


@pytest.fixture
def decoded_ffmpeg_image():
    return np.full((4, 6, 3), 127, dtype=np.uint8)


@pytest.fixture
def converter_stub(converter_module, tmp_path):
    converter_class = converter_module._Rosbag2ToNonAnnotatedT4Converter

    converter = converter_class.__new__(converter_class)
    converter._generate_ego_pose = lambda stamp: "ego_pose_token"
    converter._sample_data_table = _DummySampleDataTable()
    converter._output_scene_dir = str(tmp_path)
    converter._video_decompressor = object()
    converter._undistort_image = False

    (tmp_path / "data" / "CAM_FRONT").mkdir(parents=True)

    return converter


def test_generate_image_data_decodes_ffmpeg_packet(
    converter_stub,
    converter_module,
    ffmpeg_packet_message,
    decoded_ffmpeg_image,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter.rosbag2_utils.decode_image_msg",
        lambda image_msg, *, video_decompressor=None: converter_module.rosbag2_utils.DecodedImage(
            array=decoded_ffmpeg_image,
            fileformat="png",
        ),
    )

    converter_stub._generate_image_data(
        ffmpeg_packet_message,
        image_unix_timestamp=1.0,
        sample_token="sample_token",
        calibrated_sensor_token="calibrated_sensor_token",
        sensor_channel="CAM_FRONT",
        frame_index=0,
    )

    insert_args = converter_stub._sample_data_table.last_insert
    output_path = tmp_path / insert_args["filename"]

    assert insert_args["fileformat"] == "png"
    assert insert_args["height"] == 4
    assert insert_args["width"] == 6
    assert output_path.suffix == ".png"
    assert output_path.exists()
    assert cv2.imread(str(output_path)) is not None


def test_generate_image_data_undistorts_decoded_ffmpeg_packet(
    converter_stub,
    converter_module,
    ffmpeg_packet_message,
    decoded_ffmpeg_image,
    monkeypatch,
    tmp_path,
):
    converter_stub._undistort_image = True
    converter_stub.undistort_map_x = np.zeros(decoded_ffmpeg_image.shape[:2], dtype=np.float32)
    converter_stub.undistort_map_y = np.zeros(decoded_ffmpeg_image.shape[:2], dtype=np.float32)

    remapped_image = np.full_like(decoded_ffmpeg_image, 64)

    monkeypatch.setattr(
        "perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter.rosbag2_utils.decode_image_msg",
        lambda image_msg, *, video_decompressor=None: converter_module.rosbag2_utils.DecodedImage(
            array=decoded_ffmpeg_image,
            fileformat="png",
        ),
    )
    monkeypatch.setattr(
        "perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter.cv2.remap",
        lambda image, map_x, map_y, interpolation: remapped_image,
    )

    converter_stub._generate_image_data(
        ffmpeg_packet_message,
        image_unix_timestamp=1.0,
        sample_token="sample_token",
        calibrated_sensor_token="calibrated_sensor_token",
        sensor_channel="CAM_FRONT",
        frame_index=1,
        camera_info=object(),
    )

    insert_args = converter_stub._sample_data_table.last_insert
    output_path = tmp_path / insert_args["filename"]
    saved_image = cv2.imread(str(output_path))

    assert output_path.exists()
    assert saved_image is not None
    assert np.array_equal(saved_image, remapped_image)
