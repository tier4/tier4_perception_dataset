"""Utilities to decode camera image messages (CompressedImage and FFMPEGPacket streams)."""

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Optional, Tuple
import warnings

import av
import builtin_interfaces.msg
import cv2
from nptyping import NDArray
import numpy as np
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage

from perception_dataset.constants import EXTENSION_ENUM


@dataclass(frozen=True)
class DecodedImage:
    """Decoded image data and the file format to use when saving it.

    Attributes:
        array (NDArray | None): Decoded image array. This is `None` when decoding or decompression
            fails.
        fileformat (str): Output file format without the leading dot, such as `jpg` or `png`.
    """

    array: NDArray | None
    fileformat: str


@dataclass(frozen=True)
class VideoFrame:
    """A frame decoded from an FFMPEGPacket video stream.

    Attributes:
        array (NDArray | None): Decoded BGR image. This is `None` when the decoder
            produced no frame for the source packet (e.g. decoder delay at the start
            of the stream or a corrupt packet).
        stamp (builtin_interfaces.msg.Time): Header stamp of the source packet.
        width (int): Frame width from the packet metadata.
        height (int): Frame height from the packet metadata.
    """

    array: NDArray | None
    stamp: builtin_interfaces.msg.Time
    width: int
    height: int

    @property
    def header(self) -> SimpleNamespace:
        """Mimic the ROS message interface (`msg.header.stamp`)."""
        return SimpleNamespace(stamp=self.stamp)


def compressed_image_fileformat(compressed_image_msg: CompressedImage) -> str:
    """Infer the output file format of a CompressedImage from its format field.

    The format field is a free-form string such as `jpeg` or `rgb8; jpeg compressed bgr8`.
    """
    return (
        EXTENSION_ENUM.JPG.value[1:]
        if "jpeg" in compressed_image_msg.format
        else EXTENSION_ENUM.PNG.value[1:]
    )


def decode_image_msg(image_msg: CompressedImage) -> DecodedImage:
    """Decode a CompressedImage message into image data and file metadata.

    Returns:
        DecodedImage: Decoded pixel array and the file format to use when saving it.
        `array` can be `None` when decoding fails.
    """
    if hasattr(image_msg, "_encoding"):
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            image = np.reshape(np_arr, (image_msg.height, image_msg.width, 3))
        except Exception as e:
            print(e)
            image = None
    else:
        image_buf = np.ndarray(
            shape=(1, len(image_msg.data)),
            dtype=np.uint8,
            buffer=image_msg.data,
        )
        image = cv2.imdecode(image_buf, cv2.IMREAD_ANYCOLOR)

    return DecodedImage(array=image, fileformat=compressed_image_fileformat(image_msg))


def decode_ffmpeg_frames(
    messages: Iterable[Any],
    *,
    start_time: Optional[builtin_interfaces.msg.Time] = None,
) -> Iterator[VideoFrame]:
    """Decode a single FFMPEGPacket stream into one VideoFrame per packet, in packet order.

    Every packet is fed to a dedicated software decoder in arrival order, so the caller
    must pass all packets of the stream from the beginning of the bag (video frames
    depend on the preceding packets). Only frames whose stamp is at or after
    `start_time` are yielded. Decoded frames are paired with their source packet by
    pts, so a yielded frame has `array=None` when the decoder produced no frame for
    its packet (e.g. initial decoder delay before the first keyframe or a corrupt
    packet); frames still buffered in the decoder are drained when the stream ends.

    Args:
        messages: FFMPEGPacket messages of one topic, in arrival order.
        start_time: Yield only frames with a header stamp at or after this time.

    Yields:
        VideoFrame: One frame per input packet with stamp >= `start_time`.
    """
    codec_ctx: Optional[av.CodecContext] = None
    start = Time.from_msg(start_time) if start_time is not None else None
    # packets sent to the decoder whose frames have not come out yet
    pending: deque[Tuple[int, builtin_interfaces.msg.Time, int, int]] = deque()

    for msg in messages:
        if codec_ctx is None:
            codec_ctx = av.CodecContext.create(_ffmpeg_encoding_to_codec_name(msg.encoding), "r")
        packet = av.Packet(bytes(msg.data))
        packet.pts = msg.pts
        pending.append((msg.pts, msg.header.stamp, msg.width, msg.height))
        try:
            frames = codec_ctx.decode(packet)
        except av.FFmpegError as e:
            warnings.warn(f"failed to decode video packet: {e}")
            frames = []
        for frame in frames:
            yield from _emit_decoded_frame(frame, pending, start)

    if codec_ctx is not None:
        # drain the frames still buffered in the decoder
        try:
            frames = codec_ctx.decode(None)
        except av.FFmpegError as e:
            warnings.warn(f"failed to flush the video decoder: {e}")
            frames = []
        for frame in frames:
            yield from _emit_decoded_frame(frame, pending, start)

    # packets that never produced a frame
    while pending:
        _, stamp, width, height = pending.popleft()
        warnings.warn("no frame decoded for a trailing video packet")
        if start is None or Time.from_msg(stamp) >= start:
            yield VideoFrame(array=None, stamp=stamp, width=width, height=height)


def _emit_decoded_frame(
    frame: "av.VideoFrame",
    pending: deque,
    start: Optional[Time],
) -> Iterator[VideoFrame]:
    """Pair a decoded frame with its source packet (by pts) and yield VideoFrames.

    Pending packets older than the decoded frame produced no output; they are yielded
    with `array=None` so the caller can substitute a blank image. Streams with frame
    reordering (B-frames) are not supported: the whole conversion pipeline assumes
    monotonic frame stamps, so such frames are dropped with a warning here.
    """
    if pending and frame.pts is not None and frame.pts < pending[0][0]:
        warnings.warn(
            "decoded video frame is out of order (B-frames are not supported), dropping it"
        )
        return
    while pending:
        pts, stamp, width, height = pending.popleft()
        if frame.pts is None or pts == frame.pts:
            array = frame.to_ndarray(format="bgr24")
            if start is None or Time.from_msg(stamp) >= start:
                yield VideoFrame(array=array, stamp=stamp, width=width, height=height)
            return
        warnings.warn(f"no frame decoded for video packet with pts {pts}")
        if start is None or Time.from_msg(stamp) >= start:
            yield VideoFrame(array=None, stamp=stamp, width=width, height=height)
    # a frame without a matching pending packet: should not happen, drop it
    warnings.warn("decoded video frame does not match any pending packet")


# software decoder names per codec, in order of preference. FFmpeg's decoder
# named "av1" only supports hardware acceleration, so it must not be selected.
_SOFTWARE_DECODER_NAMES = {
    "h264": ("h264",),
    "h265": ("hevc",),
    "hevc": ("hevc",),
    "av1": ("libdav1d", "libaom-av1"),
}


def _ffmpeg_encoding_to_codec_name(encoding: str) -> str:
    """Map the FFMPEGPacket encoding field to an available FFmpeg software decoder name."""
    codec_candidates = _split_string_by_comma_and_semicolon(encoding)
    for codec in codec_candidates:
        decoder_names = _SOFTWARE_DECODER_NAMES.get(codec)
        if decoder_names is None:
            continue
        for decoder_name in decoder_names:
            if decoder_name in av.codecs_available:
                return decoder_name
        raise ValueError(f"No software decoder available for codec: {codec}")
    raise ValueError(f"Unsupported codec: {encoding}")


def _split_string_by_comma_and_semicolon(encoding: str) -> list[str]:
    def _split_by_delimiter(strings: list[str], delimiter: str) -> list[str]:
        result: list[str] = []
        for s in strings:
            result.extend(s.split(delimiter))
        return result

    return _split_by_delimiter(_split_by_delimiter([encoding], ","), ";")
