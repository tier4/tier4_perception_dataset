import argparse
from datetime import datetime
import glob
import json
import os
import os.path as osp
from typing import List, Dict, Any

import requests
import yaml

from perception_dataset.utils.logger import configure_logger
logger = configure_logger(modname=__name__)

RETRIES = 0
TIMEOUT = 100

CLIENT_ID = os.environ["DEEPEN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DEEPEN_ACCESS_TOKEN"]


class DeepenAccessException(Exception):
    MSG_TEMPLATE = "{status_code} from deepen API; request params except secrets: {params_except_secrets}; response {response}"

    def __init__(
        self,
        *,
        status_code: int,
        params_except_secrets_str: str,
        response_str: str,
    ):
        msg = self.MSG_TEMPLATE.format(
            status_code=str(status_code),
            params_except_secrets=params_except_secrets_str,
            response=response_str,
        )
        super().__init__(msg)

class AnnotationNotFoundException(Exception):
    pass

def mark_as_done(dataset_id: str) -> None:
    url = f"https://tools.deepen.ai/api/v2/datasets/{dataset_id}/mark_labelling_done"
    deepen_headers = {
        "Authorization": "Bearer " + ACCESS_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=deepen_headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

def get_dataset_status(dataset_id: str) -> str:
    url = f"https://tools.deepen.ai/api/v2/datasets/{dataset_id}"
    headers = {
        "Authorization": "Bearer " + ACCESS_TOKEN,
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

    return response.json()["current_stage_status"]

def import_label(
    dataset_id: str,
    label_set_id: str,
    label: List[Dict[str, Any]],
) -> str:
    logger.info("deepen import_label start.")
    # get labeling_profiles
    url = f"https://tools.deepen.ai/api/v2/datasets/{dataset_id}/labels?labelSetId={label_set_id}"
    logger.info(url)
    deepen_headers = {
        "Authorization": "Bearer " + ACCESS_TOKEN,
        "Content-Type": "application/json",
    }
    request_dict = {
        "type": "json",
        "labels": label,
    }
    try:
        response = requests.post(url, headers=deepen_headers, data=json.dumps(request_dict))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
 
    logger.info("Import 3D labels for a dataset end.")
    return dataset_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/import_labels.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    assert (
        config["task"] == "import_labels"
    ), f"use config file of import_labels task: {config['task']}"
    input_base = config["conversion"]["input_base"]
    label_set_id = config["conversion"]["label_set_id"]
    dataset_id_dict = config["conversion"]["dataset_ids"]
    change_to_done = config["conversion"]["change_status_to_done"]

    if not osp.exists(input_base):
        logger.error(f"input_base {input_base} does not exist.")
        raise FileNotFoundError

    log_file = "import_label.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for dataset_name, dataset_id in dataset_id_dict.items():
        json_file = f"{input_base}/{dataset_name}.json"
        if not osp.exists(json_file):
            logger.error(f"{json_file} does not exist.")
            json_file = f"{input_base}/Dataset - {dataset_name}.json"
            if not osp.exists(json_file):
                logger.error(f"{json_file} does not exist.")
                continue
        status = get_dataset_status(dataset_id)
        if status == "done":
            with open(log_file, "a") as log:
                log.write(f"{dataset_name}, {dataset_id}, already done, {timestamp}\n")
            continue
        with open(json_file) as f:
            label = yaml.safe_load(f)["labels"]
            import_label(dataset_id, label_set_id, label)
            if change_to_done:
                mark_as_done(dataset_id)

            # save log
            if change_to_done:
                status = "marked as done"
            else:
                status = "imported"

            with open(log_file, "a") as log:
                log.write(f"{dataset_name}, {dataset_id}, {status}, {timestamp}\n")
