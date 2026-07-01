#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
from pathlib import Path
import shutil

from rclpy.serialization import deserialize_message
from rosbag2_py import (
    ConverterOptions,
    SequentialReader,
    SequentialWriter,
    StorageFilter,
    StorageOptions,
    TopicMetadata,
)
from rosidl_runtime_py.utilities import get_message

DEFAULT_REMAPS = {
    "/sensing/lidar/concatenated/pointcloud": "/old/sensing/lidar/concatenated/pointcloud",
    "/sensing/lidar/concatenated/pointcloud_info": "/old/sensing/lidar/concatenated/pointcloud_info",
}


@dataclass
class Event:
    topic: str
    data: bytes
    timestamp_ns: int


class BagStream:
    def __init__(
        self,
        bag: Path,
        topics: set[str] | None,
        remaps: dict[str, str],
        *,
        use_header_time: bool = False,
    ) -> None:
        self.reader = SequentialReader()
        self.reader.open(
            StorageOptions(uri=str(bag), storage_id=infer_storage_id(bag)),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        if topics is not None:
            self.reader.set_filter(StorageFilter(topics=sorted(topics)))
        self.remaps = remaps
        self.topics = topics
        self.use_header_time = use_header_time
        self.topic_types = {
            topic.name: topic.type for topic in self.reader.get_all_topics_and_types()
        }

    def next(self) -> Event | None:
        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            if self.topics is not None and topic not in self.topics:
                continue
            if self.use_header_time:
                timestamp = header_timestamp_ns(bytes(data), self.topic_types[topic]) or timestamp
            return Event(
                topic=self.remaps.get(topic, topic),
                data=bytes(data),
                timestamp_ns=int(timestamp),
            )
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a new converter rosbag and inject old concatenated pointcloud topics under "
            "remapped names for single-clock RViz comparison."
        )
    )
    parser.add_argument(
        "--new-bag", required=True, type=Path, help="New converter input_bag path."
    )
    parser.add_argument("--old-bag", required=True, type=Path, help="Old injected rosbag path.")
    parser.add_argument(
        "--output-bag", required=True, type=Path, help="Merged output rosbag path."
    )
    parser.add_argument(
        "--old-topic",
        action="append",
        default=[],
        metavar="FROM:=TO",
        help=(
            "Old topic remap. Can be repeated. Defaults to remapping concatenated pointcloud and "
            "pointcloud_info under /old/..."
        ),
    )
    parser.add_argument(
        "--storage-id",
        default=None,
        help="Output storage id. Defaults to the new bag storage id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output bag directory if it already exists.",
    )
    parser.add_argument(
        "--preserve-qos",
        action="store_true",
        help="Preserve offered_qos_profiles metadata. Default clears QoS for Humble/Jazzy playback compatibility.",
    )
    parser.add_argument(
        "--old-write-time",
        choices=("header", "storage"),
        default="header",
        help=(
            "Timestamp to use when writing injected old topics. Default uses message header stamps, "
            "which is usually required for RViz/TF comparison."
        ),
    )
    args = parser.parse_args()

    old_remaps = dict(DEFAULT_REMAPS)
    for remap in args.old_topic:
        if ":=" not in remap:
            raise ValueError(f"Invalid --old-topic remap, expected FROM:=TO: {remap}")
        src, dst = remap.split(":=", 1)
        old_remaps[src] = dst

    if args.output_bag.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output bag already exists: {args.output_bag}")
        shutil.rmtree(args.output_bag)

    new_metadata = topic_metadata(args.new_bag)
    old_metadata = topic_metadata(args.old_bag)
    missing = sorted(topic for topic in old_remaps if topic not in old_metadata)
    if missing:
        raise KeyError(f"Old bag is missing requested topic(s): {missing}")

    storage_id = args.storage_id or infer_storage_id(args.new_bag)
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=str(args.output_bag), storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    for metadata in sorted(new_metadata.values(), key=lambda item: item.name):
        create_topic(writer, metadata, metadata.name, preserve_qos=args.preserve_qos)
    for src, dst in sorted(old_remaps.items(), key=lambda item: item[1]):
        create_topic(writer, old_metadata[src], dst, preserve_qos=args.preserve_qos)

    streams = [
        BagStream(args.new_bag, topics=None, remaps={}),
        BagStream(
            args.old_bag,
            topics=set(old_remaps),
            remaps=old_remaps,
            use_header_time=args.old_write_time == "header",
        ),
    ]
    heap: list[tuple[int, int, int, Event]] = []
    sequence = 0
    for stream_index, stream in enumerate(streams):
        event = stream.next()
        if event is not None:
            heapq.heappush(heap, (event.timestamp_ns, sequence, stream_index, event))
            sequence += 1

    counts: dict[str, int] = {}
    total = 0
    while heap:
        _, _, stream_index, event = heapq.heappop(heap)
        writer.write(event.topic, event.data, event.timestamp_ns)
        counts[event.topic] = counts.get(event.topic, 0) + 1
        total += 1

        next_event = streams[stream_index].next()
        if next_event is not None:
            heapq.heappush(heap, (next_event.timestamp_ns, sequence, stream_index, next_event))
            sequence += 1

    close = getattr(writer, "close", None)
    if close is not None:
        close()

    print(f"Wrote merged bag: {args.output_bag}")
    print(f"Total messages: {total}")
    for topic in sorted(old_remaps.values()):
        print(f"Injected {topic}: {counts.get(topic, 0)}")
    return 0


def topic_metadata(bag: Path) -> dict[str, TopicMetadata]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag), storage_id=infer_storage_id(bag)),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return {topic.name: topic for topic in reader.get_all_topics_and_types()}


def create_topic(
    writer: SequentialWriter,
    metadata: TopicMetadata,
    name: str,
    *,
    preserve_qos: bool,
) -> None:
    writer.create_topic(
        TopicMetadata(
            name=name,
            type=metadata.type,
            serialization_format=metadata.serialization_format,
            offered_qos_profiles=(
                getattr(metadata, "offered_qos_profiles", "") if preserve_qos else ""
            ),
        )
    )


def header_timestamp_ns(serialized: bytes, msg_type: str) -> int | None:
    try:
        msg = deserialize_message(serialized, get_message(msg_type))
    except Exception:
        return None
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return None
    return sec * 1_000_000_000 + nanosec


def infer_storage_id(bag: Path) -> str:
    suffixes = {".db3": "sqlite3", ".mcap": "mcap"}
    for child in bag.iterdir():
        if child.suffix in suffixes:
            return suffixes[child.suffix]
    raise FileNotFoundError(f"No supported rosbag storage file found in {bag}")


if __name__ == "__main__":
    raise SystemExit(main())
