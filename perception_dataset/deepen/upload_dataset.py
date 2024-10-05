import argparse
import glob
import json
import os
import os.path as osp
from typing import List
import urllib.parse

import httpx
import requests
import yaml

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)
RETRIES = 0
TIMEOUT = 100


CLIENT_ID = os.environ["DEEPEN_CLIENT_ID"]
ACCESS_TOKEN = os.environ["DEEPEN_ACCESS_TOKEN"]
DATSETS_URL = (
    f"https://tools.deepen.ai/api/v2/clients/{CLIENT_ID}/labels_of_dataset_ids?labelSetId=default"
)


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


def create_dataset_request(
    labeling_profile_id: str,
    labeling_mode: str,
    dataset_name: str,
    dataset_type: str,
    file_size: int,
    local_annotation_file: str,
    tags: List[str],
) -> str:
    logger.info("deepen create_dataset_request start.")
    # get labeling_profiles
    url = "https://tools.deepen.ai/api/v2/labelling_profiles/" + labeling_profile_id
    logger.info(url)
    transport = httpx.HTTPTransport(
        retries=RETRIES,
    )
    deepen_headers = {
        "Authorization": "Bearer " + ACCESS_TOKEN,
        "Content-Type": "application/json",
    }
    with httpx.Client(
        transport=transport,
        timeout=TIMEOUT,
    ) as client:
        response = client.get(
            headers=deepen_headers,
            url=url,
        )
    if response.status_code != 200:
        if response.status_code == 404:
            raise AnnotationNotFoundException()
        else:
            raise DeepenAccessException(
                status_code=response.status_code,
                params_except_secrets_str="",
                response_str=response.reason_phrase,
            )
    logger.info("deepen get labeling_profiles end.")
    # deepen datasets request
    labeling_profile = response.json()
    url = "https://tools.deepen.ai/api/v2/clients/" + CLIENT_ID + "/datasets"
    logger.info(url)
    request_dict = {
        "labelling_mode": labeling_mode,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "dataset_format": "json",
        "files": [{"file_size": file_size, "file_type": "application/zip"}],
        "labelling_profile": labeling_profile,
        "tags": tags,
    }
    with httpx.Client(
        transport=transport,
        timeout=TIMEOUT,
    ) as client:
        response = client.post(
            headers=deepen_headers,
            url=url,
            data=json.dumps(request_dict),
        )
    if response.status_code != 200:
        logger.info(response.text)
        if response.status_code == 404:
            raise AnnotationNotFoundException()
        else:
            raise DeepenAccessException(
                status_code=response.status_code,
                params_except_secrets_str="",
                response_str=response.reason_phrase,
            )
    result = response.json()
    logger.info("deepen create_dataset_request end.")
    logger.info(result)
    # datasets file upload
    dataset_id = result["dataset_id"]
    upload_file = result["files"][0]
    upload_url = upload_file["resumable_upload_url"]
    file_type = upload_file["file_type"]
    with open(local_annotation_file, "rb") as f:
        gcp_headers = {"Content-Length": str(file_size), "Content-Type": file_type}
        response = requests.put(upload_url, data=f, verify=False, headers=gcp_headers)
        if response.status_code != 200:
            if response.status_code == 404:
                raise AnnotationNotFoundException()
            else:
                raise DeepenAccessException(
                    status_code=response.status_code,
                    params_except_secrets_str="",
                    response_str=response.reason_phrase,
                )

    # datasets file upload notification
    url = "https://tools.deepen.ai/api/v2/datasets/" + dataset_id + "/process_uploaded_data"
    logger.info(url)
    upload_qs = urllib.parse.urlparse(upload_url).query
    params = urllib.parse.parse_qs(upload_qs)
    file_path = params["name"][0]
    file_name = file_path.split("/")[-1]
    request_dict = {"file_name": file_name}
    logger.info(request_dict)
    with httpx.Client(
        transport=transport,
        timeout=TIMEOUT,
    ) as client:
        response = client.post(
            headers=deepen_headers,
            url=url,
            data=json.dumps(request_dict),
        )
    if response.status_code != 200:
        if response.status_code == 404:
            raise AnnotationNotFoundException()
        else:
            raise DeepenAccessException(
                status_code=response.status_code,
                params_except_secrets_str="",
                response_str=response.reason_phrase,
            )
    return dataset_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/upload_dataset.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    assert (
        config["task"] == "upload_dataset"
    ), f"use config file of convert_deepen_to_t4 task: {config['task']}"
    input_base = config["conversion"]["input_base"]
    labeling_profile_id = config["conversion"]["labeling_profile_id"]
    labeling_mode = config["conversion"]["labeling_mode"]
    dataset_type = config["conversion"]["dataset_type"]
    tags = config["conversion"]["tags"]

    # get all dir in input_dir
    for input_dir in glob.glob(osp.join(input_base, "*")):
        file_size = os.path.getsize(input_dir)
        # get basename without extension
        dataset_name = osp.basename(input_dir)
        dataset_name = dataset_name.split(".")[0]
        dataset_id = create_dataset_request(
            labeling_profile_id,
            labeling_mode,
            dataset_name,
            dataset_type,
            file_size,
            input_dir,
            tags,
        )
        print(f"dataset_id: {dataset_id}")
