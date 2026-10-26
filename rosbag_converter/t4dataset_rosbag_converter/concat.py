from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoware_sensing_msgs.msg import ConcatenatedPointCloudInfo
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
import yaml

from .models import LidarSource
from .pointcloud import normalize_pointcloud_layout, stamp_from_seconds, stamp_to_seconds


@dataclass(frozen=True)
class ConcatenatedFrame:
    cloud: PointCloud2
    info: ConcatenatedPointCloudInfo
    source_count: int
    source_span_sec: float = 0.0
    status: str = ""
    dropped_frames_missing_transform: tuple[str, ...] = ()


class PointCloudConcatenator:
    def __init__(
        self,
        *,
        sources: list[LidarSource],
        output_frame: str,
        allow_drop: bool,
        options: dict[str, Any] | None = None,
    ) -> None:
        from autoware_pointcloud_preprocessor.concatenate_pointclouds import Concatenator

        self._sources_by_packet_topic = {source.packet_topic: source for source in sources}
        self._allow_drop = allow_drop
        self._input_topic_by_packet_topic = {
            source.packet_topic: source.info_topic for source in sources
        }
        self._input_topics = [source.info_topic for source in sources]
        self._packet_topics = [source.packet_topic for source in sources]
        self._options = options or {}
        self._param_options = _load_param_options(
            self._options.get("param_file"), self._input_topics
        )
        self._input_timestamp_offsets = self._topic_offset_map("input_timestamp_offsets")

        strategy = str(self._option("matching_strategy", "advanced"))
        offsets = self._optional_float_list("lidar_timestamp_offsets")
        noise = self._optional_float_list("lidar_timestamp_noise_window")
        timeout_sec = float(self._option("timeout_sec", 0.2))
        is_motion_compensated = bool(self._option("is_motion_compensated", True))
        self._output_timestamp_offset_sec = float(
            self._options.get("output_timestamp_offset_sec", 0.0)
        )

        tf_static = {source.frame_id: source.base_to_sensor.to_msg() for source in sources}
        self._concatenator = Concatenator(
            input_topics=self._input_topics,
            output_frame=output_frame,
            tf_static=tf_static,
            timeout_sec=timeout_sec,
            is_motion_compensated=is_motion_compensated,
            matching_strategy=strategy,
            lidar_timestamp_offsets=offsets,
            lidar_timestamp_noise_window=noise,
        )

    def add_cloud(
        self, packet_topic: str, cloud: PointCloud2, *, arrival_time: float
    ) -> list[ConcatenatedFrame]:
        input_topic = self._input_topic_by_packet_topic.get(packet_topic)
        if input_topic is None:
            return []
        offset_sec = self._input_timestamp_offsets.get(input_topic, 0.0)
        if offset_sec:
            _shift_header_stamp(cloud, offset_sec)
        frames = self._concatenator.process_cloud(input_topic, cloud, arrival_time=arrival_time)
        return self._adapt_frames(frames)

    def process_twist(self, msg: TwistWithCovarianceStamped) -> None:
        self._concatenator.process_twist(msg)

    def finish(self) -> list[ConcatenatedFrame]:
        return self._adapt_frames(self._concatenator.flush())

    def _adapt_frames(self, frames) -> list[ConcatenatedFrame]:
        return [frame for emitted in frames if (frame := self._adapt_frame(emitted)) is not None]

    def _adapt_frame(self, emitted) -> ConcatenatedFrame | None:
        from autoware_pointcloud_preprocessor.concatenate_pointclouds import CollectorStatus

        if emitted.status == CollectorStatus.TIMEOUT and not self._allow_drop:
            return None
        result = emitted.result
        cloud = result.concatenated_cloud
        info = result.concatenation_info
        if cloud is None or info is None:
            return None
        if self._output_timestamp_offset_sec:
            _shift_header_stamp(cloud, self._output_timestamp_offset_sec)
            _shift_header_stamp(info, self._output_timestamp_offset_sec)
        source_stamps = [
            stamp_to_seconds(source.header.stamp)
            for source in info.source_info
            if stamp_to_seconds(source.header.stamp) > 0.0
        ]
        source_count = len(source_stamps)
        source_span_sec = max(source_stamps) - min(source_stamps) if source_stamps else 0.0
        return ConcatenatedFrame(
            cloud=normalize_pointcloud_layout(cloud),
            info=info,
            source_count=source_count,
            source_span_sec=source_span_sec,
            status=emitted.status,
            dropped_frames_missing_transform=tuple(result.dropped_frames_missing_transform),
        )

    def _optional_float_list(self, key: str) -> list[float] | None:
        value = self._option(key)
        if value is None:
            return None
        return [float(item) for item in value]

    def _option(self, key: str, default=None):
        if key in self._options:
            return self._options[key]
        return self._param_options.get(key, default)

    def _topic_offset_map(self, key: str) -> dict[str, float]:
        value = self._options.get(key)
        if value is None:
            return {}
        if isinstance(value, dict):
            offsets = {}
            for packet_topic, input_topic in self._input_topic_by_packet_topic.items():
                if input_topic in value:
                    offsets[input_topic] = float(value[input_topic])
                elif packet_topic in value:
                    offsets[input_topic] = float(value[packet_topic])
            missing = [topic for topic in self._input_topics if topic not in offsets]
            if missing:
                raise ValueError(f"{key} is missing entries for input topics: {missing}")
            return offsets
        values = list(value)
        if len(values) != len(self._input_topics):
            raise ValueError(
                f"{key} has {len(values)} values but there are {len(self._input_topics)} lidar topics"
            )
        return {topic: float(offset) for topic, offset in zip(self._input_topics, values)}


def _shift_header_stamp(msg, offset_sec: float) -> None:
    msg.header.stamp = stamp_from_seconds(stamp_to_seconds(msg.header.stamp) + offset_sec)


def _load_param_options(param_file: Any, input_topics: list[str]) -> dict[str, Any]:
    if not param_file:
        return {}
    path = Path(str(param_file))
    with path.open("r") as fp:
        raw = yaml.safe_load(fp) or {}
    params = raw.get("/**", {}).get("ros__parameters", raw.get("ros__parameters", raw))
    param_topics = list(params.get("input_topics", []) or [])
    options: dict[str, Any] = {}
    if "timeout_sec" in params:
        options["timeout_sec"] = params["timeout_sec"]
    if "is_motion_compensated" in params:
        options["is_motion_compensated"] = params["is_motion_compensated"]
    matching = params.get("matching_strategy", {}) or {}
    if isinstance(matching, dict):
        options["matching_strategy"] = matching.get("type", "naive")
        for key in ("lidar_timestamp_offsets", "lidar_timestamp_noise_window"):
            values = matching.get(key)
            if values is not None:
                options[key] = _reorder_topic_values(
                    values,
                    from_topics=param_topics,
                    to_topics=input_topics,
                    key=key,
                    path=path,
                )
    elif matching:
        options["matching_strategy"] = matching
    return options


def _reorder_topic_values(
    values: Any,
    *,
    from_topics: list[str],
    to_topics: list[str],
    key: str,
    path: Path,
) -> list[float]:
    values = list(values)
    if len(values) != len(from_topics):
        raise ValueError(
            f"{path}: {key} has {len(values)} values but input_topics has {len(from_topics)}"
        )
    by_topic = {topic: float(value) for topic, value in zip(from_topics, values)}
    missing = [topic for topic in to_topics if topic not in by_topic]
    if missing:
        raise ValueError(f"{path}: missing {key} entries for input topics: {missing}")
    return [by_topic[topic] for topic in to_topics]
