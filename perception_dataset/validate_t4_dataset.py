"""This tool validates that the token of schema can be accessed to the target schema and the camera and lidar data is more than 9 Hz.
2D annotation format is't supported yet.
"""

import argparse

from loguru import logger
from t4_devkit import Tier4

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME
from perception_dataset.t4_dataset.data_validator import validate_data_hz
from perception_dataset.t4_dataset.format_validator import (
    validate_directory_structure,
    validate_format,
)


@logger.catch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, help="path to T4Dataset")
    args = parser.parse_args()

    logger.info(f"Load {args.root_path}")

    validate_directory_structure(args.root_path)

    nusc = Tier4(
        data_root=args.root_path,
        verbose=False,
    )

    validate_format(nusc, args.root_path)
    validate_data_hz(nusc)


if __name__ == "__main__":
    main()
