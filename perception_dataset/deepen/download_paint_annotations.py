import argparse
from datetime import date
import json
import os
from typing import List
import pprint
import gzip
import zlib
import base64
import io
import sys
import requests
import yaml

CLIENT_ID = os.environ["DEEPEN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DEEPEN_ACCESS_TOKEN"]
DATSETS_URL = (
    f"https://tools.deepen.ai/api/v2/clients/{CLIENT_ID}/labels_of_dataset_ids?labelSetId=default"
)

today = str(date.today()).replace("-", "")


# def get_dataset():
#     URL = f"https://tools.deepen.ai/api/v2/datasets/{DATASET_ID}/labels?filter_existing_categories=true&final=true&all=true"
#     print(URL)

#     headers = {
#         "Authorization": f"Bearer {os.environ['DEEPEN_ACCESS_TOKEN']}",
#     }
#     response = requests.get(URL, headers=headers)
#     print(response.status_code)
#     pprint(response.json())


def get_datasets(dataset_ids: List[str], dataset_dir: str, output_name: str):
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    TMP_URL = (
        # f"https://tools.deepen.ai/api/v2/datasets/{dataset_ids[0]}/files/0.pcd/label_types/3d_point/paint_labels?stageId=QA"
        f"https://tools.deepen.ai/api/v2/datasets/{dataset_ids[0]}/label_types/3d_point/paint_labels?stageId=QA"
        # f"https://tools.deepen.ai/api/v2/datasets/{dataset_ids[0]}/labels?filter_existing_categories=true&final=true&all=true"
        # f"https://tools.deepen.ai/api/v2/datasets/{dataset_ids[0]}/label_tasks"
    )
    # TMP_URL = (
    #     f"https://tools.deepen.ai/api/v2/clients/{CLIENT_ID}/datasets_ids"
    # )
    # payload = {
    #     # "stageId" : "QA",
    #     "sensorId" : "lidar",
    #     "labelSetId" : "default"
    # }
    data = {"dataset_ids": dataset_ids}
    # print(data)
    # try:
    # response = requests.get(TMP_URL, headers=headers)
    response = requests.get(TMP_URL, headers=headers)
    print(response.status_code)
    decompress_data = zlib.decompress(bytearray(response.content))
    # print(decompress_data)
    # decoded = decompress_data.decode('utf-8')
    # print(f'inflated size : {len(decoded)}')
    # print(f'inflated contents : {decoded}',file=sys.stderr)
    print(response.headers.values)
    # print(json.dumps(json.loads(response.text), indent=2))
    # print(response.content)
    # print(response.)
    # response.raise_for_status()
    # except requests.exceptions.RequestException as e:
        # print(json.dumps(json.loads(e.response.text), indent=2))
        # raise SystemExit(e)
    return
    output_file = os.path.join(dataset_dir, output_name)
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(response.json(), f, indent=4)

    print(f"Annotation file is saved: {output_file}")


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

    get_datasets(dataset_ids, args.output_dir, output_name)
