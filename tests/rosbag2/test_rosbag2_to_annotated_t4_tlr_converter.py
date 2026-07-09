import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from ffmpeg_image_transport_msgs.msg import FFMPEGPacket


class _DummySampleTable:
    def __init__(self, records=None):
        self._records = records or []

    def to_records(self):
        return self._records

    def insert_into_table(self, **kwargs):
        return "sample_token"


class _DummyEgoPoseTable:
    @staticmethod
    def get_record_from_token(token):
        assert token == "ego_pose_token"
        return SimpleNamespace(translation=[1.0, 0.0, 0.0])


@pytest.fixture
def annotated_tlr_module():
    pytest.importorskip("t4_devkit")

    tier4_perception_msgs = types.ModuleType("tier4_perception_msgs")
    tier4_perception_msgs_msg = types.ModuleType("tier4_perception_msgs.msg")

    class _TrafficLightArray:
        pass

    class _TrafficLightRoiArray:
        pass

    tier4_perception_msgs_msg.TrafficLightArray = _TrafficLightArray
    tier4_perception_msgs_msg.TrafficLightRoiArray = _TrafficLightRoiArray
    tier4_perception_msgs.msg = tier4_perception_msgs_msg

    autoware_msgs = types.ModuleType("perception_dataset.rosbag2.autoware_msgs")
    autoware_msgs.parse_traffic_lights = lambda *args, **kwargs: []
    autoware_msgs.parse_perception_objects = lambda *args, **kwargs: []

    sys.modules.setdefault("tier4_perception_msgs", tier4_perception_msgs)
    sys.modules.setdefault("tier4_perception_msgs.msg", tier4_perception_msgs_msg)
    sys.modules.setdefault("perception_dataset.rosbag2.autoware_msgs", autoware_msgs)

    return importlib.import_module(
        "perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter"
    )


@pytest.fixture
def ffmpeg_packet_message():
    ffmpeg_msg = FFMPEGPacket()
    ffmpeg_msg.data = b"ffmpeg-packet"
    ffmpeg_msg.width = 6
    ffmpeg_msg.height = 4
    ffmpeg_msg.header.stamp.sec = 1
    ffmpeg_msg.header.stamp.nanosec = 0
    return ffmpeg_msg


def test_convert_image_camera_only_passes_ffmpeg_packet_through(
    annotated_tlr_module, ffmpeg_packet_message
):
    converter_class = annotated_tlr_module._Rosbag2ToAnnotatedT4TlrConverter
    sensor_mode = annotated_tlr_module.SensorMode

    converter = converter_class.__new__(converter_class)
    converter._sensor_mode = sensor_mode.NO_LIDAR
    converter._sample_table = _DummySampleTable()
    converter._ego_pose_table = _DummyEgoPoseTable()
    converter._num_load_frames = 1
    converter._generate_frame_every = 1
    converter._generate_frame_every_meter = 0.0
    converter._end_timestamp = 0.0
    converter._generate_ego_pose = lambda stamp: "ego_pose_token"
    converter._generate_calibrated_sensor = lambda sensor_channel, stamp, topic: (
        "calibrated_sensor_token",
        None,
    )
    converter._is_traffic_light_label_available = lambda timestamp: True

    captured = {}

    def _capture_generate_image_data(*args, **kwargs):
        captured["image_msg"] = args[0]
        captured["frame_index"] = args[5]
        return "sample_data_token"

    converter._generate_image_data = _capture_generate_image_data
    converter._bag_reader = SimpleNamespace(
        read_messages=lambda **kwargs: iter([ffmpeg_packet_message])
    )

    sample_data_tokens = converter._convert_image(
        start_timestamp=0.0,
        sensor_channel="CAM_FRONT",
        topic="/camera/ffmpeg",
        delay_msec=0.0,
        scene_token="scene_token",
    )

    assert sample_data_tokens == ["sample_data_token"]
    assert captured["image_msg"] is ffmpeg_packet_message
    assert captured["frame_index"] == 0


def test_convert_image_lidar_mode_uses_generic_image_decode(
    annotated_tlr_module, ffmpeg_packet_message, monkeypatch
):
    converter_class = annotated_tlr_module._Rosbag2ToAnnotatedT4TlrConverter
    sensor_mode = annotated_tlr_module.SensorMode

    converter = converter_class.__new__(converter_class)
    converter._sensor_mode = sensor_mode.DEFAULT
    converter._sample_table = _DummySampleTable(
        [SimpleNamespace(timestamp=1_000_000, token="lidar_sample_token")]
    )
    converter._lidar_latency = 0.0
    converter._system_scan_period_sec = 0.1
    converter._max_camera_jitter_sec = 0.01
    converter._num_load_frames = 1
    converter._msg_display_interval = 100
    converter._end_timestamp = 0.0
    converter._video_decompressor = object()
    converter._generate_calibrated_sensor = lambda sensor_channel, stamp, topic: (
        "calibrated_sensor_token",
        None,
    )
    converter._is_traffic_light_label_available = lambda timestamp: True

    monkeypatch.setattr(
        "perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter.misc_utils.get_lidar_camera_synced_frame_info",
        lambda **kwargs: [(0, 0, None)],
    )

    decoded_image = np.zeros((4, 6, 3), dtype=np.uint8)
    decode_calls = []

    def _decode_image_msg(image_msg, *, video_decompressor=None):
        decode_calls.append((image_msg, video_decompressor))
        return annotated_tlr_module.rosbag2_utils.DecodedImage(
            array=decoded_image,
            fileformat="png",
        )

    monkeypatch.setattr(
        "perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter.rosbag2_utils.decode_image_msg",
        _decode_image_msg,
    )

    captured = {}

    def _capture_generate_image_data(*args, **kwargs):
        captured["image_msg"] = args[0]
        captured["image_shape"] = args[6]
        return "sample_data_token"

    converter._generate_image_data = _capture_generate_image_data
    converter._bag_reader = SimpleNamespace(
        read_messages=lambda **kwargs: iter([ffmpeg_packet_message])
    )

    sample_data_tokens = converter._convert_image(
        start_timestamp=0.0,
        sensor_channel="CAM_FRONT",
        topic="/camera/ffmpeg",
        delay_msec=0.0,
        scene_token="scene_token",
    )

    assert sample_data_tokens == ["sample_data_token"]
    assert decode_calls == [(ffmpeg_packet_message, converter._video_decompressor)]
    assert captured["image_msg"] is ffmpeg_packet_message
    assert captured["image_shape"] == decoded_image.shape
