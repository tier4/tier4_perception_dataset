import argparse
from datetime import date
import json
import os
from typing import List

import requests
import yaml

CLIENT_ID = os.environ["DEEPEN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DEEPEN_ACCESS_TOKEN"]
DATASETS_URL = (
    f"https://tools.deepen.ai/api/v2/clients/{CLIENT_ID}/labels_of_dataset_ids?labelSetId=default"
)

today = str(date.today()).replace("-", "")

def get_datasets(dataset_ids: List[str], dataset_dir: str, output_name: str):
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {"dataset_ids": dataset_ids}

    try:
        response = requests.post(DATASETS_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

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

    print("Requesting annotated json for: ", dataset_ids)

    get_datasets(dataset_ids, args.output_dir, output_name)
