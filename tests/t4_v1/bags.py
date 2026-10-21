from rclpy.serialization import serialize_message
from rosbag2_py import TopicMetadata

from perception_dataset.utils.rosbag2 import create_writer


def write_bag(path, messages):
    """Write (topic, type, message or raw CDR bytes, sec) to a sqlite3 bag."""
    writer = create_writer(str(path))
    for topic, type_name in {(topic, type_name) for topic, type_name, _, _ in messages}:
        writer.create_topic(TopicMetadata(name=topic, type=type_name, serialization_format="cdr"))
    for topic, _, message, sec in messages:
        data = message if isinstance(message, bytes) else serialize_message(message)
        writer.write(topic, data, sec * 10**9)
    del writer
