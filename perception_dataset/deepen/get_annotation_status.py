import argparse
from datetime import datetime

import yaml

from perception_dataset.deepen.import_labels import AnnotationNotFoundException, get_dataset_status
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


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

    dataset_id_dict = config["conversion"]["dataset_ids"]

    date = datetime.now().strftime("%Y-%m-%d")
    log_file = f"annotation_status_{date}.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for dataset_name, dataset_id in dataset_id_dict.items():
        status_msg = ""
        try:
            status = get_dataset_status(dataset_id)
            if status == "done":
                status_msg = "Done"
            elif status == "ready":
                status_msg = "Under Review"
            else:
                status_msg = status
        except AnnotationNotFoundException:
            status_msg = "Error"

        with open(log_file, "a") as log:
            log.write(f"{status_msg}\n")
            # log.write(f"{timestamp}, {dataset_name}, {dataset_id}, {status}\n")
