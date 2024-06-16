import argparse
from datetime import date
import json
import os
from typing import List
import zlib

import numpy as np
import requests
from sensor_msgs.msg import PointField
import tqdm
import yaml

if os.environ.get("DEEPEN_CLIENT_ID") is None or os.environ.get("DEEPEN_ACCESS_TOKEN") is None:
    raise ValueError("You need to properly set the environment variables \"DEEPEN_CLIENT_ID\" and \"DEEPEN_ACCESS_TOKEN\"")
else:
    CLIENT_ID = os.environ["DEEPEN_CLIENT_ID"]
    ACCESS_TOKEN = os.environ["DEEPEN_ACCESS_TOKEN"]

today = str(date.today()).replace("-", "")
NUM_DIMENSIONS = 5


def get_datasets(dataset_id: str, input_base_dir: str) -> None:
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    if dataset_id is None:
        print("Dataset ID not found")
        return

    DATASET_URL = f"https://tools.deepen.ai/api/v2/datasets/{dataset_id}/label_types/3d_point/paint_labels?stageId=QA"

    try:
        response = requests.get(DATASET_URL, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    
    decompress_data = bytearray(zlib.decompress(bytearray(response.content)))

    header_list = list(response.headers.values())
    label_info = json.loads(header_list[3])
    frame_size = list(label_info["frame_sizes"])

    # [
    #     [l,l,l,l,...],
    #     [l,l,l,l,...],
    #     ...
    # ]

    label_list = []
    offset = 0
    for i in range(len(frame_size)):
        offset += frame_size[i - 1] if i != 0 else 0
        label_list.append(
            np.array([decompress_data[offset + j] for j in range(frame_size[i])], dtype=np.uint32)
        )

    sample_data_file = os.path.join(input_base_dir, "annotation", "sample_data.json")
    sample_data = json.load(open(sample_data_file, "r"))
    sample_data = list(filter(lambda d: d["filename"].split(".")[-2] == "pcd", sample_data))

    for i in tqdm.tqdm(range(0, len(sample_data))):
        pcd_file_path = os.path.join(input_base_dir, sample_data[i]["filename"])
        scan = np.fromfile(pcd_file_path, dtype=np.float32)
        points: np.ndarray = scan.reshape((-1, NUM_DIMENSIONS))  # 行数，列数
        labelled_points: np.ndarray = np.concatenate(
            [points, label_list[i].reshape(-1, 1)], 1, dtype=np.float32
        )
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
    dataset_id = list(config["conversion"]["dataset_corresponding"].values())
    input_base_dir = os.path.join(
        config["conversion"]["input_base"], os.listdir(config["conversion"]["input_base"])[0]
    )

    get_datasets(dataset_id[0], input_base_dir)
