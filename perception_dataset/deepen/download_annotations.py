import argparse
from datetime import date
import json
import os
from pathlib import Path
from typing import List
import zlib

import numpy as np
import requests
import yaml

CLIENT_ID = os.environ["DEEPEN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DEEPEN_ACCESS_TOKEN"]
DATASET_ID = "r82dS6VVkok98ZWr1CTkU4NE"
_HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json",
}

today = str(date.today()).replace("-", "")


def get_annotation_labels(dataset_ids: List[str], dataset_dir: str, output_name: str) -> None:
    """Get annotation labels except paint3d from deepen."""
    datasets_url = f"https://tools.deepen.ai/api/v2/clients/{CLIENT_ID}/labels_of_dataset_ids?labelSetId=default_semantic"
    data = {"dataset_ids": dataset_ids}

    try:
        response = requests.post(datasets_url, headers=_HEADERS, data=json.dumps(data))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

    output_file = os.path.join(dataset_dir, output_name)
    os.makedirs(os.path.dirname(output_name), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(response.json(), f, indent=4)

    print(f"Annotation file is saved: {output_file}")


def get_paint3d_labels(dataset_ids: List[str], dataset_dir: str, output_name: str) -> None:
    """
    Get 3d paint labels to lidar_seg/<dataset_id>_<frame_id>.bin and <output_name>.json.
    Example of a response:
        {
            'server': 'gunicorn', 'date': 'Wed, 15 Jan 2025 08:45:01 GMT',
            'content-type': 'text/html; charset=utf-8',
            'paint-metadata': '{
                "format": "pako_compressed", "paint_categories": ["car", "wall"],
                "frame_sizes": [173430, 173097, 172595, 172957, 173011, 173204, 173813, 173913, 174393, 175304, 175176, 174038, 174473, 174373, 172598,
                171886, 172377, 171724, 173322, 173566, 175911, 176165, 177180, 177774, 178544, 179606, 178347,
                177048, 176775, 176997, 177271, 177864, 179341, 178986, 179670, 179973, 178885, 178424, 179233, 179766, 178644, 179057, 177871, 178286, 178570,
                177410, 176688, 176961, 176871, 176862, 176740, 177872, 179313, 179414, 179496, 178974, 178653, 178992, 178861, 178823, 178690, 178815, 178966, 178960, 178907]
            }',
            'vary': 'Accept, Cookie, Accept-Encoding',
            'allow': 'POST, GET, OPTIONS',
            'x-frame-options': 'DENY',
            'x-content-type-options': 'nosniff',
            'referrer-policy': 'same-origin',
            'content-encoding': 'gzip',
            'via': '1.1 google',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Alt-Svc': 'h3=":443"; ma=2592000,h3-29=":443"; ma=2592000',
            'Transfer-Encoding': 'chunked'
        }

    <output_name>.json is in the following format:
        [
            {
                "dataset_id": "DOnC2vK05ojPr7qiqCsk2Ee7",
                "file_id": "0.pcd",
                "label_type": "3d_point",
                "label_id": "none",		# Keep it for consistency with downstream tasks
                "label_category_id": "none",	# Keep it for consistency with downstream tasks
                "total_lidar_points": 173430,
                        "sensor_id": "lidar",
                        "stage_id": "QA",
                "paint_categories": ["car", "wall", ...],
                "lidarseg_anno_file": "lidar_seg/DOnC2vK05ojPr7qiqCsk2Ee7_0.bin"
            },
            ...
        ]
    """
    dataset_path = Path(dataset_dir) / "lidar_seg"
    dataset_path.mkdir(parents=True, exist_ok=True)
    lidarseg_annos_info = []
    for dataset_id in dataset_ids:
        datasets_url = f"https://tools.deepen.ai/api/v2/datasets/{DATASET_ID}/label_types/3d_point/paint_labels?stageId=QA&sensor_id=lidar&labelSetId=default"
        try:
            response = requests.get(datasets_url, headers=_HEADERS)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        paint_metadata = json.loads(response.headers["paint-metadata"])
        decompressed_data = zlib.decompress(response.content)
        lidarseg_category_indices = np.frombuffer(decompressed_data, dtype=np.uint8)
        # Split based on pointcloud size
        previous_frame_size = 0
        for file_id, frame_size in enumerate(paint_metadata["frame_sizes"]):
            lidarseg_anno_filename = dataset_path / f"{dataset_id}_{str(file_id)}.bin"

            annos_info = {
                "dataset_id": dataset_id,
                "file_id": f"{file_id}.pcd",
                "label_id": "none",  # Keep it for consistency with downstream tasks
                "label_category_id": "none",  # Keep it for consistency with downstream tasks
                "total_lidar_points": frame_size,
                "sensor_id": "lidar",
                "stage_id": "QA",
                "paint_categories": paint_metadata["paint_categories"],
                "lidarseg_anno_file": str(lidarseg_anno_filename),
            }

            with open(lidarseg_anno_filename, "wb") as f:
                lidarseg_category_indices[
                    previous_frame_size : previous_frame_size + frame_size
                ].tofile(f)

            previous_frame_size = frame_size
            lidarseg_annos_info.append(annos_info)

    # Save lidarseg anno information to json
    output_file = os.path.join(dataset_dir, output_name)
    with open(output_file, "w") as f:
        json.dump(lidarseg_annos_info, f, indent=4)

    print(f"Annotation file is saved: {output_file}")


def get_datasets(dataset_ids: List[str], dataset_dir: str, output_name: str, label_type: str):
    if label_type == "labels":
        labels_request_func = get_annotation_labels
    elif label_type == "paint3d":
        labels_request_func = get_paint3d_labels
    else:
        raise ValueError(f"Label type: {label_type} not supported!")

    labels_request_func(dataset_ids=dataset_ids, dataset_dir=dataset_dir, output_name=output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/convert_deepen_to_t4.yaml",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="labels",
        help="Annotation type to request labels, supported values: ['labels', 'paint3d']",
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

    get_datasets(dataset_ids, args.output_dir, output_name, args.label_type)
