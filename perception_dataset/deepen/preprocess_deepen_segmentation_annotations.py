import argparse
import logging
from pathlib import Path
import re
import shutil
from typing import Optional, Tuple
import zipfile


def extract_zip(zip_file: Path, out_dir: Path) -> Tuple[Path]:
    """
    Extracts a zip file and removes unnecessary directory levels.

    Args:
    - zip_file (Path): The path to the zip file to extract.
    - out_dir (Path): The directory where the extracted files will be placed.

    Returns:
    - base_dir (Path): The base directory after extraction. Returns None if it fails.
    - data_dir (Path): The path to the directory containing the data.
    """
    # Get the base name of the zip file (without the extension)
    base_name: str = zip_file.stem
    extract_path: Path = out_dir / base_name

    # Create the output directory
    extract_path.mkdir(parents=True, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Search for base_dir
    data_dirs: list[Path] = list(extract_path.glob("./tmp/deepenLabels-*"))
    if not data_dirs:
        logging.error("Could not find deepenLabels-* directory")
        return None
    if len(data_dirs) != 1:
        logging.error("Multiple data_dirs found in base_dir")
        return None
    data_dir: Path = data_dirs[0]
    base_dir: Path = data_dir.parent.parent

    return base_dir, data_dir


def reorganize_directory(base_dir: Path, data_dir: Path, logger: logging.Logger) -> None:
    """
    Reorganizes the directory structure.

    Process:
    - Removes prefixes from file names in base_dir.
    - Moves `.npy` files to sensor directories.

    Args:
    - base_dir (Path): The path to the directory to process.
    - data_dir (Path): The path to the directory containing the data.
    - logger (logging.Logger): Logger object for logging information.

    Returns:
    - None

    Example:

        Before
            base_dir/
                ├── tmp/
                │   └── data_dir/
                │       ├── Semantic Segmentation - metadata.json
                │       ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
                │       ├── Semantic Segmentation - sensor2 - data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
                │       ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy

        After
            base_dir/
                ├── metadata.json
                ├── CAM_TRAFFIC_LIGHT_NEAR/
                │   ├── data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
                │   ├── data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
                ├── CAM_TRAFFIC_LIGHT_FAR/
                │   └── data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy
    """
    # Flatten the directory structure
    flatten_directory_structure(base_dir=base_dir, data_dir=data_dir, logger=logger)

    for file_path in base_dir.iterdir():
        if file_path.is_file():
            # Remove prefix and rename file
            new_file_path: Path = remove_prefix(file_path)

            # If it's an npy file, move it to the appropriate sensor directory
            if new_file_path.suffix == ".npy":
                move_to_sensor_directory(new_file_path, base_dir)


def flatten_directory_structure(base_dir: Path, data_dir: Path, logger: logging.Logger) -> None:
    """
    Removes unnecessary directory levels and moves files up one directory.

    Args:
    - base_dir (Path): The parent directory where files will be moved.
    - data_dir (Path): The path to the directory containing the data.

    Returns:
    - None

    Example:

        Before
            base_dir/
                ├── tmp/
                │   └── data_dir/
                │       ├── Semantic Segmentation - metadata.json
                │       ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
                │       ├── Semantic Segmentation - sensor2 - data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
                │       ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy

        After
            base_dir/
                ├── Semantic Segmentation - metadata.json
                ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
                ├── Semantic Segmentation - sensor2 - data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
                ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy
    """
    # Move all files and directories inside data_dir to base_dir
    item: Path
    for item in data_dir.iterdir():
        dest_path: Path = base_dir / item.name
        if dest_path.exists():
            logger.warning(f"Warning: {dest_path} already exists. Skipping.")
            continue
        item.replace(dest_path)

    # Delete the parent directory of data_dir (tmp directory)
    tmp_dir: Path = data_dir.parent
    shutil.rmtree(tmp_dir)


def remove_prefix(file_path: Path) -> Path:
    """
    Removes the prefix from a file name and renames the file.

    - Changes 'Semantic Segmentation - metadata.json' to 'metadata.json'.
    - Removes the prefix 'Semantic Segmentation - sensorX - '.

    Args:
    - file_path (Path): The original file path.

    Returns:
    - new_file_path (Path): The new file path with the prefix removed.

    Example:

        Before
            base_dir/
                ├── Semantic Segmentation - metadata.json
                ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
                ├── Semantic Segmentation - sensor2 - data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
                ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy

        After
            base_dir/
                ├── metadata.json
                ├── data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
                ├── data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
                ├── data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy
    """
    filename: str = file_path.name
    new_filename: str
    if filename == "Semantic Segmentation - metadata.json":
        new_filename = "metadata.json"
    elif filename.endswith(".npy") and "Semantic Segmentation - sensor" in filename:
        # Remove the prefix 'Semantic Segmentation - sensorX - '
        new_filename = re.sub(r"^Semantic Segmentation - sensor\d+ - ", "", filename)
    else:
        new_filename = filename

    if new_filename != filename:
        new_file_path: Path = file_path.with_name(new_filename)
        file_path.rename(new_file_path)
        return new_file_path
    else:
        return file_path


def move_to_sensor_directory(npy_path: Path, base_dir: Path) -> None:
    """
    Moves npy files to the appropriate sensor directory.

    Args:
    - npy_path (Path): The path of the npy file.
    - base_dir (Path): The path to the base directory.

    Returns:
    - None

    Example:

        Before
            base_dir/
                ├── metadata.json
                ├── data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
                ├── data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
                ├── data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy

        After
            base_dir/
                ├── metadata.json
                ├── CAM_TRAFFIC_LIGHT_NEAR/
                │   ├── data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
                │   ├── data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
                ├── CAM_TRAFFIC_LIGHT_FAR/
                │   └── data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy
    """
    filename: str = npy_path.name
    match: Optional[re.Match] = re.search(r"data_(.+)_\d+_jpg.npy", filename)
    if match:
        sensor_dir_name: str = match.group(1)
        dir_path: Path = base_dir / sensor_dir_name
        dir_path.mkdir(exist_ok=True)
        dest_npy_path: Path = dir_path / filename
        npy_path.replace(dest_npy_path)


def preprocess_deepen_segmentation_annotation(
    zip_file: Path, out_dir: Path, logger: logging.Logger
) -> None:
    """
    Extracts and reorganizes the annotation data for Deepen's semantic segmentation.

    Args:
    - zip_file (Path): The path to the zip file to process.
    - out_dir (Path): The directory where the extracted files will be stored.
    - logger (logging.Logger): Logger object for logging information.

    Returns:
    - None
    """
    logger.info(f"Processing zip file: {zip_file}")

    # extract zip. .npy files are contained in base_dir/tmp/data_dir/
    base_dir, data_dir = extract_zip(zip_file, out_dir)
    if base_dir is None:
        logger.error(f"Failed to extract zip file: {zip_file}")
        return

    # Reorganize the directory structure
    reorganize_directory(base_dir, data_dir, logger)
    logger.info(f"Finish pre-processing. Output: {base_dir}")


def parse_args(logger: logging.Logger) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract annotation data for Deepen's semantic segmentation."
    )
    parser.add_argument("--zip-file", type=Path, help="Path to the zip file to process.")
    parser.add_argument(
        "--zip-files-dir", type=Path, help="Path to the directory containing multiple zip files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where the extracted files will be stored.",
    )
    args: argparse.Namespace = parser.parse_args()
    if not args.zip_file and not args.zip_files_dir:
        logger.error("Please specify either --zip-file or --zip-files-dir.")
        return
    if args.zip_file and args.zip_files_dir:
        logger.error("Please specify either --zip-file or --zip-files-dir, but not both")
        return

    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger: logging.Logger = logging.getLogger(__name__)
    args: Optional[argparse.Namespace] = parse_args(logger)

    if args and args.zip_file:
        # Process a single zip file
        preprocess_deepen_segmentation_annotation(args.zip_file, args.out_dir, logger)
    elif args and args.zip_files_dir:
        # Process all zip files in the directory
        zip_files_dir: Path = args.zip_files_dir
        zip_file_path: Path
        for zip_file_path in zip_files_dir.glob("*.zip"):
            preprocess_deepen_segmentation_annotation(zip_file_path, args.out_dir, logger)


if __name__ == "__main__":
    main()
