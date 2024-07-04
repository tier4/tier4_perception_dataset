import logging
import os.path as osp
import re
import shutil
import sys
from typing import Dict, List, Text, Union

from rclpy.serialization import deserialize_message
from rosbag2_py import StorageFilter
from rosidl_runtime_py.utilities import get_message

from perception_dataset.utils.rosbag2 import (
    create_reader,
    create_writer,
    get_topic_count,
    get_topic_type_dict,
    reindex,
)


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
        topic_list: Union[Dict[str, List[str]], List[str]] = [],
        start_time_sec: float = 0,
        end_time_sec: float = sys.float_info.max,
        mandatory_topics: List = MANDATORY_TOPICS,
    ):
        self._input_bag_dir: str = input_bag_dir
        self._output_bag_dir: str = output_bag_dir

        if "topic_list" in topic_list:
            # if topic_list is Dict, topic_list["topic_list"] is used as allow_topics
            allow_topics = topic_list["topic_list"]
        elif isinstance(topic_list, list):
            # if topic_list is List, topic_list is used as allow_topics
            allow_topics = topic_list
        else:
            # if topic_list is not specified, allow_topics is empty
            allow_topics = []

        if "mandatory_topic_list" in topic_list:
            # if topic_list is Dict, topic_list["mandatory_topic_list"] is used as mandatory_topics
            mandatory_topics_list = topic_list["mandatory_topic_list"]
        elif mandatory_topics is None:
            # if mandatory_topics is None, mandatory_topics_list is empty
            mandatory_topics_list = []
        else:
            # if mandatory_topics is not specified, mandatory_topics_list is used as mandatory_topics
            mandatory_topics_list = mandatory_topics

        self._topic_list: List[str] = allow_topics
        self._mandatory_topics: List[str] = mandatory_topics_list

        self._start_time_sec = start_time_sec
        self._end_time_sec = end_time_sec
        self._check_topic_count()
        self._topic_name_to_topic_type = get_topic_type_dict(self._input_bag_dir)

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

            topic_type = self._topic_name_to_topic_type[topic_name]
            try:
                msg_type = get_message(topic_type)
                msg = deserialize_message(data, msg_type)
                if hasattr(msg, "header"):
                    message_time: float = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                elif hasattr(msg, "transforms"):
                    message_time: float = (
                        msg.transforms[0].header.stamp.sec
                        + msg.transforms[0].header.stamp.nanosec * 1e-9
                    )
            except ModuleNotFoundError:
                logging.warning(f"ModuleNotFoundError: {topic_type}")
            except AttributeError:
                logging.error(
                    f"AttributeError: {topic_type}. Sourced message type is differ from the one in rosbag. {topic_name} is ignored."
                )
                continue

            if topic_name == "/tf_static":
                start_time_in_ns = int(self._start_time_sec*1e9)
                timestamp = max(timestamp, start_time_in_ns)
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
