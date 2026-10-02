"""some implementations are from https://github.com/tier4/ros2bag_extensions/blob/main/ros2bag_extensions/ros2bag_extensions/verb/__init__.py"""

from collections import deque
from dataclasses import dataclass
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import warnings

import av
import builtin_interfaces.msg
import cv2
from nptyping import NDArray
import numpy as np
from pypcd4 import PointCloud
from radar_msgs.msg import RadarTrack, RadarTracks
from rclpy.time import Time
from rosbag2_py import (
    ConverterOptions,
    Reindexer,
    SequentialReader,
    SequentialWriter,
    StorageOptions,
)
from sensor_msgs.msg import CompressedImage, PointCloud2
import yaml

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.utils.misc import unix_timestamp_to_nusc_timestamp


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


def get_options(
    bag_dir: str,
    storage_options: Optional[StorageOptions] = None,
    converter_options: Optional[ConverterOptions] = None,
) -> Tuple[StorageOptions, ConverterOptions]:
    storage_options = storage_options if storage_options else get_default_storage_options(bag_dir)
    converter_options = converter_options if converter_options else get_default_converter_options()
    return storage_options, converter_options


def create_reader(
    bag_dir: str,
    storage_options: Optional[StorageOptions] = None,
    converter_options: Optional[ConverterOptions] = None,
) -> SequentialReader:
    storage_options, converter_options = get_options(bag_dir, storage_options, converter_options)
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    return reader


def create_writer(bag_dir: str, storage_id: Optional[str] = None) -> SequentialWriter:
    """Create and open a rosbag2 SequentialWriter.

    Parameters
    ----------
    bag_dir : str
        Target directory URI where the bag will be written.
    storage_id : Optional[str], optional
        Identifier of the storage backend to use. This value is passed to
        :class:`rosbag2_py.StorageOptions` as ``storage_id``. Common values
        include ``"sqlite3"`` and ``"mcap"``, depending on which storage
        plugins are available in the environment. If ``None``, the writer
        defaults to ``"sqlite3"`` for backward compatibility.

    Returns
    -------
    SequentialWriter
        A writer instance opened with the specified storage and converter options.
    """
    if storage_id is None:
        storage_id = "sqlite3"  # default to sqlite3 for backward compatibility
    storage_options = StorageOptions(uri=bag_dir, storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    writer = SequentialWriter()
    writer.open(storage_options, converter_options)

    return writer


def reindex(bag_dir: str):
    storage_options = get_default_storage_options(bag_dir)
    Reindexer().reindex(storage_options)


def get_topic_type_dict(bag_dir: str) -> Dict[str, str]:
    reader = create_reader(bag_dir)

    topic_name_to_topic_type: Dict[str, str] = {}
    for topic in reader.get_all_topics_and_types():
        topic_name_to_topic_type[topic.name] = topic.type

    return topic_name_to_topic_type


def get_topic_count(bag_dir: str) -> Dict[str, int]:
    with open(osp.join(bag_dir, "metadata.yaml")) as f:
        bagfile_metadata = yaml.safe_load(f)["rosbag2_bagfile_information"]
    topic_name_to_topic_count: Dict[str, int] = {}
    for topic in bagfile_metadata["topics_with_message_count"]:
        topic_name_to_topic_count[topic["topic_metadata"]["name"]] = topic["message_count"]
    return topic_name_to_topic_count


def get_default_converter_options() -> ConverterOptions:
    return ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )


def infer_storage_id(bag_dir: str, storage_ids={".db3": "sqlite3", ".mcap": "mcap"}) -> str:
    bag_dir_path = Path(bag_dir)
    data_file = next(p for p in bag_dir_path.glob("*") if p.suffix in storage_ids)
    if data_file.suffix not in storage_ids:
        raise ValueError(f"Unsupported storage id: {data_file.suffix}")
    return storage_ids[data_file.suffix]


def get_default_storage_options(bag_dir: str) -> StorageOptions:
    storage_id = infer_storage_id(bag_dir)
    return StorageOptions(uri=bag_dir, storage_id=storage_id)


def _get_field(
    pointcloud: PointCloud, field_names: Tuple[str, ...], required: bool = False
) -> NDArray:
    available_fields = pointcloud.fields
    num_points = pointcloud.metadata.points
    for field_name in field_names:
        if field_name in available_fields:
            return pointcloud.numpy((field_name,)).reshape(-1).astype(np.float32)
    if required:
        raise ValueError(
            f"PointCloud2 must contain {field_names}. Available fields: {available_fields}"
        )
    return np.full((num_points,), -1, dtype=np.float32)


def pointcloud_msg_to_numpy(pointcloud_msg: PointCloud2, num_lidar_feats: int = 5) -> NDArray:
    """Convert ROS PointCloud2 message to a float32 numpy array."""
    if num_lidar_feats not in (5, 7):
        raise ValueError(f"num_lidar_feats must be 5 or 7, got {num_lidar_feats}")

    if not isinstance(pointcloud_msg, PointCloud2):
        return np.zeros((0, num_lidar_feats), dtype=np.float32)

    pointcloud = PointCloud.from_msg(pointcloud_msg)

    columns = [
        _get_field(pointcloud, ("x",), required=True),
        _get_field(pointcloud, ("y",), required=True),
        _get_field(pointcloud, ("z",), required=True),
        _get_field(pointcloud, ("intensity", "i")),
        _get_field(pointcloud, ("ring", "channel")),
    ]

    if num_lidar_feats == 7:
        columns.append(_get_field(pointcloud, ("return_type",)))
        columns.append(_get_field(pointcloud, ("time_stamp", "timestamp")))

    points_arr = np.column_stack(columns).astype(np.float32, copy=False)
    return points_arr


def radar_tracks_msg_to_list(radar_tracks_msg: RadarTracks) -> List[Dict[str, Any]]:
    """Convert `RadarTracks` into list.
    Each element of list is dict as shown below.

    translation (Tuple[float, float, float]): x, y, z coordinates of the centroid of the object.
    velocity (Tuple[float, float, float]): The velocity of the object in each spatial dimension.
    acceleration (Tuple[float, float, float]): The acceleration of the object in each spatial dimension.
    size (Tuple[float, float, float]): The object size in the sensor frame.
    uuid (str): A unique ID of the object generated by the radar.
    classification (int): Object classification. NO_CLASSIFICATION=0, STATIC=1, DYNAMIC=2.
    """
    radar_tracks: List[Dict[str, Any]] = []
    for track in radar_tracks_msg.tracks:
        track: RadarTrack
        translation: Tuple[float, float, float] = (
            track.position.x,
            track.position.y,
            track.position.z,
        )
        velocity: Tuple[float, float, float] = (
            track.velocity.x,
            track.velocity.y,
            track.velocity.z,
        )
        acceleration: Tuple[float, float, float] = (
            track.acceleration.x,
            track.acceleration.y,
            track.acceleration.z,
        )
        size: Tuple[float, float, float] = (track.size.x, track.size.y, track.size.z)
        obj_uuid: str = str(uuid.UUID(bytes=track.uuid.uuid.tobytes()))

        radar_tracks.append(
            {
                "translation": translation,
                "velocity": velocity,
                "acceleration": acceleration,
                "size": size,
                "uuid": obj_uuid,
                "classification": track.classification,
            }
        )
    return radar_tracks


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


def stamp_to_unix_timestamp(stamp: builtin_interfaces.msg.Time) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def unix_timestamp_to_stamp(timestamp: float) -> builtin_interfaces.msg.Time:
    sec_int = int(timestamp)
    nano_sec_int = (timestamp - sec_int) * 1e9
    return Time(seconds=sec_int, nanoseconds=nano_sec_int).to_msg()


def stamp_to_nusc_timestamp(stamp: builtin_interfaces.msg.Time) -> int:
    return unix_timestamp_to_nusc_timestamp(stamp_to_unix_timestamp(stamp))
