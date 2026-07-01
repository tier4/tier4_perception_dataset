from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message
from rclpy.serialization import serialize_message
from rosbag2_py import ConverterOptions
from rosbag2_py import SequentialReader
from rosbag2_py import SequentialWriter
from rosbag2_py import StorageFilter
from rosbag2_py import StorageOptions
from rosbag2_py import TopicMetadata
from rosidl_runtime_py.utilities import get_message

from .geometry import RigidTransform
from .geometry import rigid_from_matrix


@dataclass(frozen=True)
class BagEvent:
    topic: str
    serialized: bytes
    storage_time_ns: int
    msg_type: str

    def deserialize(self):
        return deserialize_message(self.serialized, get_message(self.msg_type))


class SequentialBagReader:
    def __init__(self, bag_paths: Iterable[str | Path]) -> None:
        self.bag_paths = [Path(path) for path in bag_paths]

    def iter_events(self, topics: set[str] | None = None):
        for bag_path in self.bag_paths:
            reader = SequentialReader()
            storage_id = _infer_storage_id(bag_path)
            reader.open(
                StorageOptions(uri=str(bag_path), storage_id=storage_id),
                ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
            )
            if topics is not None:
                reader.set_filter(StorageFilter(topics=sorted(topics)))
            topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                if topics is not None and topic not in topics:
                    continue
                yield BagEvent(
                    topic=topic,
                    serialized=bytes(data),
                    storage_time_ns=int(timestamp),
                    msg_type=topic_types[topic],
                )

    def topic_metadata(self) -> dict[str, TopicMetadata]:
        metadata = {}
        for bag_path in self.bag_paths:
            reader = SequentialReader()
            reader.open(
                StorageOptions(uri=str(bag_path), storage_id=infer_storage_id(bag_path)),
                ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
            )
            for topic in reader.get_all_topics_and_types():
                metadata.setdefault(topic.name, topic)
        return metadata


class SequentialBagWriter:
    def __init__(self, bag_path: str | Path, *, storage_id: str) -> None:
        self.bag_path = Path(bag_path)
        self.storage_id = storage_id
        self._writer = SequentialWriter()
        self._created_topics: set[str] = set()

    def open(self) -> None:
        self.bag_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer.open(
            StorageOptions(uri=str(self.bag_path), storage_id=self.storage_id),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )

    def create_topic(self, name: str, msg_type: str, *, offered_qos_profiles: str = "") -> None:
        if name in self._created_topics:
            return
        self._writer.create_topic(
            TopicMetadata(
                name=name,
                type=msg_type,
                serialization_format="cdr",
                offered_qos_profiles=offered_qos_profiles,
            )
        )
        self._created_topics.add(name)

    def create_existing_topic(self, metadata: TopicMetadata) -> None:
        self.create_topic(
            metadata.name,
            metadata.type,
            offered_qos_profiles=getattr(metadata, "offered_qos_profiles", ""),
        )

    def write_serialized(self, topic: str, serialized: bytes, timestamp_ns: int) -> None:
        self._writer.write(topic, serialized, int(timestamp_ns))

    def write_message(self, topic: str, msg, timestamp_ns: int) -> None:
        self.write_serialized(topic, bytes(serialize_message(msg)), timestamp_ns)

    def close(self) -> None:
        close = getattr(self._writer, "close", None)
        if close is not None:
            close()


def transform_stamped_to_rigid(msg: TransformStamped) -> RigidTransform:
    matrix = _matrix_from_transform_stamped(msg)
    return rigid_from_matrix(msg.header.frame_id, msg.child_frame_id, matrix)


def _matrix_from_transform_stamped(msg: TransformStamped):
    from .geometry import RigidTransform

    return RigidTransform(
        parent=msg.header.frame_id,
        child=msg.child_frame_id,
        translation=(
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z,
        ),
        rotation_xyzw=(
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w,
        ),
    ).matrix()


def infer_storage_id(bag_path: str | Path) -> str:
    return _infer_storage_id(Path(bag_path))


def _infer_storage_id(bag_path: Path) -> str:
    suffixes = {".db3": "sqlite3", ".mcap": "mcap"}
    for child in bag_path.iterdir():
        if child.suffix in suffixes:
            return suffixes[child.suffix]
    raise FileNotFoundError(f"No supported rosbag storage file found in {bag_path}")
