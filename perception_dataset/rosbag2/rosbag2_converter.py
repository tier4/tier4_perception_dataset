import os.path as osp
import re
import shutil
import sys
from typing import Dict, List, Text

from rosbag2_py import StorageFilter

from perception_dataset.utils.rosbag2 import create_reader, create_writer, get_topic_count, reindex


class Rosbag2Converter:
    MANDATORY_TOPICS = [
        "pointcloud",
        "/tf",
        "/tf_static",
    ]

    def __init__(
        self,
        input_bag_dir: Text,
        output_bag_dir: Text,
        topic_list: List = [],
        start_time_sec: float = 0,
        end_time_sec: float = sys.float_info.max,
        mandatory_topics: List = MANDATORY_TOPICS,
    ):
        self._input_bag_dir: str = input_bag_dir
        self._output_bag_dir: str = output_bag_dir
        self._topic_list: str = topic_list
        self._start_time_sec = start_time_sec
        self._end_time_sec = end_time_sec
        self._mandatory_topics = mandatory_topics
        self._check_topic_count()

    def _check_topic_count(self):
        topic_count: Dict[str, int] = get_topic_count(self._input_bag_dir)

        for topic in self._mandatory_topics:
            if topic not in topic_count.keys():
                for key in topic_count.keys():
                    if re.search(topic, key):
                        topic_count[topic] = topic_count[key]
                        break
            try:
                if topic_count[topic] == 0:
                    raise ValueError(
                        f"{topic} topic count is 0. The input rosbag must contain {topic}."
                    )
            except KeyError:
                raise ValueError(
                    f"There is no {topic} in the rosbag. The input rosbag must contain {topic}."
                )

    def convert(self):
        writer = create_writer(self._output_bag_dir)
        reader = create_reader(self._input_bag_dir)
        if len(self._topic_list) != 0:
            reader.set_filter(StorageFilter(topics=self._topic_list))

        for topic in reader.get_all_topics_and_types():
            writer.create_topic(topic)

        write_topic_count = 0
        while reader.has_next():
            topic_name, data, timestamp = reader.read_next()
            message_time = timestamp * 1e-9
            if topic_name == "/tf_static":
                writer.write(topic_name, data, timestamp)
            elif message_time <= self._start_time_sec:
                continue
            elif message_time <= self._end_time_sec:
                writer.write(topic_name, data, timestamp)
                write_topic_count += 1
            else:
                break
        del writer
        # Reindex for cleanup metadata
        reindex(self._output_bag_dir)

        if write_topic_count == 0:
            raise ValueError(
                "Total topic count in rosbag is 0. The input rosbag timestamp might not match the timestamp in dataset."
            )

    def make_input_bag(self):
        output_bag_dir_temp: str = osp.join(
            self._output_bag_dir, osp.basename(self._input_bag_dir)
        )
        output_bag_dir: str = osp.join(self._output_bag_dir, "input_bag")
        self._output_bag_dir = output_bag_dir_temp
        self.convert()
        shutil.move(output_bag_dir_temp, output_bag_dir)
