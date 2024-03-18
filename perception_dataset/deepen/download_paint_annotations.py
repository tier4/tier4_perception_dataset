import argparse
from datetime import date
import json
import os
from typing import List
import zlib
import yaml
import sqlite3
import requests
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from rclpy.serialization import serialize_message
import rosbag2_py

from perception_dataset.utils import misc

CLIENT_ID = os.environ["DEEPEN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DEEPEN_ACCESS_TOKEN"]
today = str(date.today()).replace("-", "")

point_fields = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="entity_id", offset=16, datatype=PointField.UINT32, count=1),
]

def get_datasets(dataset_ids: List[str], dataset_dir: str, output_name: str, input_bag_file: str, input_base_dir: str):
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    DATASET_URL = f"https://tools.deepen.ai/api/v2/datasets/{dataset_ids[0]}/label_types/3d_point/paint_labels?stageId=QA"

    response = requests.get(DATASET_URL, headers=headers)
    decompress_data = bytearray(zlib.decompress(bytearray(response.content)))
    # print(decompress_data)
    # print(list(response.headers.values()))
    header_list = list(response.headers.values())
    label_info = json.loads(header_list[3])
    frame_size = list(label_info['frame_sizes'])
    # skip_timestamp = 1.0 # TODO これをconvertのconfig.yamlからロードしたい

    sample_data_file = os.path.join(input_base_dir, "annotation", "sample_data.json")
    sample_data = json.load(open(sample_data_file, 'r'))
    sample_data = list(filter(lambda d : d["filename"].split(".")[-2] == "pcd", sample_data))
    
    target_topic_name = '/sensing/lidar/concatenated/pointcloud'
    bag_connection = sqlite3.connect(input_bag_file)
    cursor = bag_connection.cursor()
    # reader = rosbag2_py

    cursor.execute('PRAGMA table_info(messages)')
    table_info = cursor.fetchall()
    print(table_info)

    # return

    cursor.execute('SELECT * from(topics)')
    topic_names = cursor.fetchall()

    target_topic_row = None
    msg_types = []
    for topic in topic_names:
        msg_types.append(topic[2])
        if topic[1] == target_topic_name:
            target_topic_row = topic
            print(target_topic_row)
    type_map = {topic_names[i][1]:msg_types[i] for i in range(len(msg_types))}
    
    if target_topic_row is None:
        print('target topic does not appeared')
        return

    time_stamps = []
    messages = []
    cursor.execute('SELECT * from(messages)')
    msgs = cursor.fetchall()
    for msg in msgs:
        # print(msg)
        if msg[1] == target_topic_row[0]: # id check
            # print(f'ts : {msg[2]} , msg : {msg[3]}')
            time_stamps.append(msg[2])
            messages.append(msg[3])
    
    msg_type = get_message(type_map[target_topic_name])

    def header_stamp_to_nusc_ts(msg: PointCloud2):
        unix_ts = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        nusc_ts = misc.unix_timestamp_to_nusc_timestamp(unix_ts)
        return nusc_ts
    
    def out_of_range(l: list, i)->bool:
        return len(l) <= i
    
    msg_cnt = 0
    print(len(sample_data))
    for i in range(len(sample_data)):
        msg: PointCloud2 = deserialize_message(messages[msg_cnt], msg_type)
        while sample_data[i]["timestamp"] != header_stamp_to_nusc_ts(msg):
            msg_cnt += 1
            msg: PointCloud2 = deserialize_message(messages[msg_cnt], msg_type)
        
        print(f'Label pointcloud ts: {msg.header.stamp}')
        np_cloud = point_cloud2.read_points(msg, ['x', 'y', 'z', 'intensity'])
        cloud_list = np_cloud.tolist()
        labelled_cloud_list = [list(cloud_list[j]) + [int(decompress_data[i + j])] for j in range(len(cloud_list))]
        labelled_cloud = point_cloud2.create_cloud(msg.header,point_fields, labelled_cloud_list)
        serialized_labelled_cloud = serialize_message(labelled_cloud)
        print(sample_data[i]["timestamp"])
        replace = "REPLACE INTO messages(topic_id,timestamp,data) VALUES(?,?,?)"
        cursor.execute(replace, (target_topic_row[0], sample_data[i]["timestamp"], serialized_labelled_cloud,))
        bag_connection.commit()
        return
        msg_cnt += 1
        if out_of_range(messages, msg_cnt):
            break

    # print(deserialized_msgs)
    # print(len(messages))
    # bag_connection.commit()
    bag_connection.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/convert_deepen_to_t4.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="the directory where the annotation file is saved.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    assert (
        config["task"] == "convert_deepen_to_t4"
    ), f"use config file of convert_deepen_to_t4 task: {config['task']}"
    dataset_ids = list(config["conversion"]["dataset_corresponding"].values())
    output_name = config["conversion"]["input_anno_file"]
    input_bag_file = config["conversion"]["input_bag_file"]
    input_base_dir = config["conversion"]["input_base"] + "/" + (config["conversion"]["input_base"]).split("/")[-1].split("_")[0]

    get_datasets(dataset_ids, args.output_dir, output_name, input_bag_file, input_base_dir)
