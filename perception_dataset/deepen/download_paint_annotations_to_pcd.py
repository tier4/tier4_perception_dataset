import argparse
from datetime import date
import json
import os
from typing import List
import zlib
import yaml
import sqlite3
import pathlib
import requests
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from rclpy.serialization import serialize_message
import rosbag2_py
from rclpy.time import Time
from perception_dataset.utils import misc

import numpy as np

import tqdm

CLIENT_ID = os.environ["DEEPEN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DEEPEN_ACCESS_TOKEN"]
today = str(date.today()).replace("-", "")

NUM_DIMENSIONS = 5

point_fields = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="entity_id", offset=16, datatype=PointField.UINT32, count=1),
]

test_point_fields = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
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
    print(list(response.headers.values()))
    header_list = list(response.headers.values())
    label_info = json.loads(header_list[3])
    frame_size = list(label_info['frame_sizes'])
    
    # [
    #     [l,l,l,l,...],
    #     [l,l,l,l,...],
    #     ...
    # ]

    label_list = []
    offset = 0
    for i in range(len(frame_size)):
        offset += frame_size[i-1] if i != 0 else 0
        label_list.append(
            np.array([
                decompress_data[offset + j] for j in range(frame_size[i])
            ], dtype=np.uint32)
        )

    sample_data_file = os.path.join(input_base_dir, "annotation", "sample_data.json")
    sample_data = json.load(open(sample_data_file, 'r'))
    sample_data = list(filter(lambda d : d["filename"].split(".")[-2] == "pcd", sample_data))

    labelled_points_dtype = np.dtype(
        {
            "names": ("x", "y", "z", "intensity", "reserve", "entity_id"),
            "formats": ("f4", "f4", "f4", "f4", "f4", "u4"),
        }
    )
    
    for i in tqdm.tqdm(range(0, len(sample_data))):
        pcd_file_path = os.path.join(input_base_dir, sample_data[i]["filename"])
        scan = np.fromfile(pcd_file_path, dtype=np.float32)
        points: np.ndarray = scan.reshape((-1, NUM_DIMENSIONS)) # 行数，列数
        # print(f"===== open {pcd_file_path} =====")
        # print(points)
        # return
        labelled_points: np.ndarray = np.concatenate([points, label_list[i].reshape(-1,1)],1, dtype=np.float32)
        # labelled_points = labelled_points.astype(labelled_points_dtype)
        labelled_points.tofile(pcd_file_path)

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
