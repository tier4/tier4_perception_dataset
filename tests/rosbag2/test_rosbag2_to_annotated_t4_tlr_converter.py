import importlib
import sys
import types
from types import SimpleNamespace

import builtin_interfaces.msg
import numpy as np
import pytest

from perception_dataset.utils import rosbag2 as rosbag2_utils


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
    """Duck-typed FFMPEGPacket; the converter never needs the real message class."""
    return SimpleNamespace(
        data=b"ffmpeg-packet",
        pts=0,
        flags=1,
        encoding="h264",
        width=6,
        height=4,
        header=SimpleNamespace(stamp=builtin_interfaces.msg.Time(sec=1, nanosec=0)),
    )


@pytest.fixture
def decoded_video_frame(ffmpeg_packet_message):
    return rosbag2_utils.VideoFrame(
        array=np.zeros((4, 6, 3), dtype=np.uint8),
        stamp=ffmpeg_packet_message.header.stamp,
        width=6,
        height=4,
    )


def _make_bag_reader(message, topic_type):
    return SimpleNamespace(
        read_messages=lambda **kwargs: iter([message]),
        get_topic_type=lambda topic_name: topic_type,
    )


def test_convert_image_camera_only_decodes_ffmpeg_topic(
    annotated_tlr_module, ffmpeg_packet_message, decoded_video_frame, monkeypatch
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
    converter._bag_reader = _make_bag_reader(
        ffmpeg_packet_message, "ffmpeg_image_transport_msgs/msg/FFMPEGPacket"
    )

    decode_calls = []

    def _decode_ffmpeg_frames(messages, *, start_time=None):
        decode_calls.append((list(messages), start_time))
        yield decoded_video_frame

    monkeypatch.setattr(
        "perception_dataset.utils.rosbag2.decode_ffmpeg_frames",
        _decode_ffmpeg_frames,
    )

    captured = {}

    def _capture_generate_image_data(*args, **kwargs):
        captured["image_msg"] = args[0]
        captured["frame_index"] = args[5]
        return "sample_data_token"

    converter._generate_image_data = _capture_generate_image_data

    sample_data_tokens = converter._convert_image(
        start_timestamp=0.0,
        sensor_channel="CAM_FRONT",
        topic="/camera/ffmpeg",
        delay_msec=0.0,
        scene_token="scene_token",
    )

    assert sample_data_tokens == ["sample_data_token"]
    # the whole packet stream reaches the decoder, the decoded frame reaches the writer
    assert decode_calls[0][0] == [ffmpeg_packet_message]
    assert captured["image_msg"] is decoded_video_frame
    assert captured["frame_index"] == 0


def test_convert_image_lidar_mode_probes_shape_without_decoding(
    annotated_tlr_module, ffmpeg_packet_message, decoded_video_frame, monkeypatch
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
    converter._generate_calibrated_sensor = lambda sensor_channel, stamp, topic: (
        "calibrated_sensor_token",
        None,
    )
    converter._is_traffic_light_label_available = lambda timestamp: True
    converter._bag_reader = _make_bag_reader(
        ffmpeg_packet_message, "ffmpeg_image_transport_msgs/msg/FFMPEGPacket"
    )

    monkeypatch.setattr(
        "perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter.misc_utils.get_lidar_camera_synced_frame_info",
        lambda **kwargs: [(0, 0, None)],
    )
    monkeypatch.setattr(
        "perception_dataset.utils.rosbag2.decode_ffmpeg_frames",
        lambda messages, *, start_time=None: iter([decoded_video_frame]),
    )

    decode_image_msg_calls = []
    monkeypatch.setattr(
        "perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter.rosbag2_utils.decode_image_msg",
        lambda image_msg: decode_image_msg_calls.append(image_msg),
    )

    captured = {}

    def _capture_generate_image_data(*args, **kwargs):
        captured["image_msg"] = args[0]
        captured["image_shape"] = args[6]
        return "sample_data_token"

    converter._generate_image_data = _capture_generate_image_data

    sample_data_tokens = converter._convert_image(
        start_timestamp=0.0,
        sensor_channel="CAM_FRONT",
        topic="/camera/ffmpeg",
        delay_msec=0.0,
        scene_token="scene_token",
    )

    assert sample_data_tokens == ["sample_data_token"]
    # the shape comes from the packet metadata; nothing is decoded out of band
    assert decode_image_msg_calls == []
    assert captured["image_shape"] == (4, 6, 3)
    assert captured["image_msg"] is decoded_video_frame


def test_convert_image_camera_only_re_encodes_compressed_image(annotated_tlr_module, monkeypatch):
    """CompressedImage frames are decoded at the call site so the output stays
    byte-identical with previous converter versions (re-encode, no passthrough)."""
    from sensor_msgs.msg import CompressedImage

    compressed_image = CompressedImage()
    compressed_image.header.stamp.sec = 1
    compressed_image.format = "jpeg"

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
    converter._bag_reader = _make_bag_reader(compressed_image, "sensor_msgs/msg/CompressedImage")
    decoded_array = np.zeros((4, 6, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "perception_dataset.rosbag2.rosbag2_to_annotated_t4_tlr_converter.rosbag2_utils.decode_image_msg",
        lambda image_msg: rosbag2_utils.DecodedImage(array=decoded_array, fileformat="jpg"),
    )

    captured = {}

    def _capture_generate_image_data(*args, **kwargs):
        captured["image_msg"] = args[0]
        return "sample_data_token"

    converter._generate_image_data = _capture_generate_image_data

    sample_data_tokens = converter._convert_image(
        start_timestamp=0.0,
        sensor_channel="CAM_FRONT",
        topic="/camera/compressed",
        delay_msec=0.0,
        scene_token="scene_token",
    )

    assert sample_data_tokens == ["sample_data_token"]
    assert captured["image_msg"] is decoded_array
