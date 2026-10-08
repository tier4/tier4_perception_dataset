from fractions import Fraction
from types import SimpleNamespace
import warnings

import av
import builtin_interfaces.msg
import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CompressedImage

from perception_dataset.utils.image import (
    compressed_image_fileformat,
    decode_ffmpeg_frames,
    decode_image_msg,
    video_frame_to_bgr,
)


@pytest.mark.parametrize(
    ("format_str", "expected"),
    [
        ("jpeg", "jpg"),
        ("rgb8; jpeg compressed bgr8", "jpg"),
        ("png", "png"),
        ("rgb8; png compressed ", "png"),
    ],
)
def test_compressed_image_fileformat(format_str, expected):
    msg = CompressedImage()
    msg.format = format_str
    assert compressed_image_fileformat(msg) == expected


def test_decode_compressed_image_msg():
    image = np.full((4, 6, 3), 127, dtype=np.uint8)
    msg = CompressedImage()
    msg.format = "rgb8; jpeg compressed bgr8"
    msg.data = cv2.imencode(".jpg", image)[1].tobytes()

    decoded = decode_image_msg(msg)

    assert decoded.fileformat == "jpg"
    assert decoded.array.shape == (4, 6, 3)
    np.testing.assert_allclose(decoded.array, image, atol=3)


def _encode_video_packets(images, *, zerolatency=True):
    """Encode BGR images with libx264 and return the packets in encoder output order."""
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = images[0].shape[1]
    encoder.height = images[0].shape[0]
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    if zerolatency:
        encoder.options = {"tune": "zerolatency"}
    packets = []
    for pts, image in enumerate(images):
        frame = av.VideoFrame.from_ndarray(image, format="bgr24").reformat(format="yuv420p")
        frame.pts = pts
        packets.extend(encoder.encode(frame))
    packets.extend(encoder.encode(None))
    return packets


def _packets_to_messages(packets, width, height):
    """Wrap av packets into duck-typed FFMPEGPacket-like messages (stamp = pts seconds)."""
    return [
        SimpleNamespace(
            data=bytes(packet),
            pts=int(packet.pts),
            flags=1 if packet.is_keyframe else 0,
            encoding="h264",
            width=width,
            height=height,
            header=SimpleNamespace(
                stamp=builtin_interfaces.msg.Time(sec=1000 + int(packet.pts), nanosec=0)
            ),
        )
        for packet in packets
    ]


@pytest.fixture
def video_images():
    return [np.full((48, 64, 3), i * 30, dtype=np.uint8) for i in range(8)]


@pytest.fixture
def video_messages(video_images):
    return _packets_to_messages(_encode_video_packets(video_images), width=64, height=48)


def test_decode_ffmpeg_frames_yields_one_frame_per_packet(video_messages, video_images):
    frames = list(decode_ffmpeg_frames(video_messages))

    assert len(frames) == len(video_messages)
    assert [frame.stamp.sec for frame in frames] == [1000 + i for i in range(len(frames))]
    for i, frame in enumerate(frames):
        assert frame.array is not None
        assert frame.array.shape == (48, 64, 3)
        assert frame.width == 64
        assert frame.height == 48
        # the decoded frame must carry the content of its own source packet
        assert abs(float(frame.array.mean()) - float(video_images[i].mean())) < 6.0


def test_decode_ffmpeg_frames_mimics_message_interface(video_messages):
    frame = next(decode_ffmpeg_frames(video_messages))
    assert frame.header.stamp == frame.stamp


def test_decode_ffmpeg_frames_filters_by_start_time_but_decodes_all(video_messages):
    start_time = builtin_interfaces.msg.Time(sec=1003, nanosec=0)

    frames = list(decode_ffmpeg_frames(video_messages, start_time=start_time))

    # earlier packets are decoded (the stream depends on them) but not yielded
    assert [frame.stamp.sec for frame in frames] == [1003, 1004, 1005, 1006, 1007]
    assert all(frame.array is not None for frame in frames)


def test_decode_ffmpeg_frames_yields_none_for_undecodable_packet(video_messages):
    video_messages[4].data = b"\x00\x01garbage"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = list(decode_ffmpeg_frames(video_messages))

    assert len(frames) == len(video_messages)
    assert frames[4].array is None
    for i, frame in enumerate(frames):
        assert frame.stamp.sec == 1000 + i
        if i != 4:
            assert frame.array is not None


def test_decode_ffmpeg_frames_never_mis_stamps_reordered_streams(video_images):
    """B-frame streams are unsupported: frames may be dropped (-> dummy), never mis-stamped."""
    packets = _encode_video_packets(video_images, zerolatency=False)
    assert [int(p.pts) for p in packets] != sorted(
        int(p.pts) for p in packets
    ), "encoder did not reorder; test setup is broken"
    messages = _packets_to_messages(packets, width=64, height=48)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = list(decode_ffmpeg_frames(messages))

    for frame in frames:
        if frame.array is not None:
            source_index = frame.stamp.sec - 1000
            assert (
                abs(float(frame.array.mean()) - float(video_images[source_index].mean())) < 6.0
            ), f"frame yielded with the wrong stamp: {frame.stamp.sec}"


def _yuv_frame(y, u=128, v=128, *, fmt="yuv420p", width=64, height=48):
    frame = av.VideoFrame(width, height, fmt)
    frame.planes[0].update(np.full((height, width), y, np.uint8).tobytes())
    frame.planes[1].update(np.full((height // 2, width // 2), u, np.uint8).tobytes())
    frame.planes[2].update(np.full((height // 2, width // 2), v, np.uint8).tobytes())
    return frame


@pytest.mark.parametrize(
    ("tagged_range", "override", "expected_black_level"),
    [
        (1, None, 0),  # tagged limited: Y=16 is pure black
        (2, None, 16),  # tagged full: Y=16 stays dark gray
        (0, None, 0),  # untagged: swscale default (limited)
        (1, "full", 16),  # wrong "limited" tag overridden (the real-bag case)
        (2, "limited", 0),  # wrong "full" tag overridden
    ],
)
def test_video_frame_to_bgr_color_range(tagged_range, override, expected_black_level):
    frame = _yuv_frame(16)
    frame.color_range = tagged_range

    bgr = video_frame_to_bgr(frame, color_range=override)

    assert int(bgr[0, 0, 0]) == expected_black_level
    # white point check: Y=235 clips to 255 in limited, stays 235 in full
    white = _yuv_frame(235)
    white.color_range = tagged_range
    bgr_white = video_frame_to_bgr(white, color_range=override)
    assert int(bgr_white[0, 0, 0]) == (255 if expected_black_level == 0 else 235)


def test_video_frame_to_bgr_colorspace_matrix():
    colored = _yuv_frame(120, 90, 170)

    bt601 = video_frame_to_bgr(colored, colorspace="bt601")
    default = video_frame_to_bgr(colored)
    bt709 = video_frame_to_bgr(colored, colorspace="bt709")

    assert np.array_equal(bt601, default), "bt601 must be the default matrix"
    assert not np.array_equal(bt709, bt601), "bt709 must use a different matrix"


def test_video_frame_to_bgr_full_range_with_bt709():
    frame = _yuv_frame(16)

    bgr = video_frame_to_bgr(frame, color_range="full", colorspace="bt709")

    # the explicit colorspace must not silently reset the range to limited
    assert int(bgr[0, 0, 0]) == 16


def test_decode_ffmpeg_frames_passes_color_overrides(video_messages):
    limited = next(decode_ffmpeg_frames(video_messages)).array
    full = next(decode_ffmpeg_frames(video_messages, color_range="full")).array

    assert not np.array_equal(limited, full)


def test_decode_ffmpeg_frames_rejects_unknown_color_settings(video_messages):
    with pytest.raises(ValueError, match="color_range"):
        next(decode_ffmpeg_frames(video_messages, color_range="fullish"))
    with pytest.raises(ValueError, match="colorspace"):
        next(decode_ffmpeg_frames(video_messages, colorspace="bt2020"))
