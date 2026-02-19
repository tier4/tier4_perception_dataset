import json
from pathlib import Path

from t4_devkit.cli.sanity import sanity_check
import yaml

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME
from tests.constants import (
    DB3_FILE_PATTERN,
    INPUT_BAG_DIR_NAME,
    JSON_EXTENSION,
    METADATA_YAML_FILENAME,
    TOKEN_FIELD_NAMES,
)


def _load_json_file(file_path: Path) -> dict:
    """Load and return JSON data from a file."""
    with open(file_path) as f:
        return json.load(f)


def _load_yaml_file(file_path: Path) -> dict:
    """Load and return YAML data from a file."""
    with open(file_path) as f:
        return yaml.safe_load(f)


def _remove_token_and_unused_fields(data):
    """
    Recursively remove key-value pairs where the key is in TOKEN_FIELD_NAMES or unused (None or empty).

    Args:
        data: Dictionary, list, or other data structure to process

    Returns:
        A new data structure with token fields removed
    """
    if isinstance(data, dict):
        return {
            key: _remove_token_and_unused_fields(value)
            for key, value in data.items()
            if key not in TOKEN_FIELD_NAMES and value not in (None, "", [], {}, ())
        }
    elif isinstance(data, list):
        return [_remove_token_and_unused_fields(item) for item in data]
    else:
        return data


def _normalize_numeric_values(data):
    """
    Recursively normalize numeric values to ensure 4.0 and 4 are treated as equal.
    Converts float values that are equivalent to integers into integers.

    Args:
        data: Dictionary, list, or other data structure to process

    Returns:
        A new data structure with normalized numeric values
    """
    if isinstance(data, dict):
        return {key: _normalize_numeric_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_normalize_numeric_values(item) for item in data]
    elif isinstance(data, float) and data.is_integer():
        return int(data)
    else:
        return data


def _compare_json_files(target_file: Path, source_file: Path) -> None:
    """Compare two JSON files after removing token fields."""
    target_data = _load_json_file(target_file)
    source_data = _load_json_file(source_file)

    # Remove token fields from both datasets
    target_filtered = _remove_token_and_unused_fields(target_data)
    source_filtered = _remove_token_and_unused_fields(source_data)

    # Normalize numeric values (e.g., 4.0 -> 4) for consistent comparison
    target_normalized = _normalize_numeric_values(target_filtered)
    source_normalized = _normalize_numeric_values(source_filtered)

    # For lists, check if intersection equals target
    target_set = {json.dumps(x, sort_keys=True) for x in target_normalized}
    source_set = {json.dumps(x, sort_keys=True) for x in source_normalized}
    intersection = target_set & source_set
    assert len(intersection) == len(target_set) == len(source_set), (
        f"Differences found in {target_file.name}: " f"Ssource and target elements are not same"
    )


def _count_files_recursively(directory: Path) -> int:
    """Count all files in a directory recursively."""
    return len([f for f in directory.rglob("*") if f.is_file()])


def _assert_file_counts_match(target_count: int, source_count: int, context: str = "") -> None:
    """Assert that file counts match between target and source."""
    assert target_count == source_count, (
        f"File count mismatch{f' in {context}' if context else ''}: "
        f"{target_count} files in target, {source_count} files in source"
    )


def diff_check_annotation(target_dir: Path, source_dir: Path) -> None:
    """Compare annotation directories between target and source datasets."""
    target_annotation_dir = target_dir / T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value
    source_annotation_dir = source_dir / T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value

    if not target_annotation_dir.exists() and not source_annotation_dir.exists():
        return  # Both annotation directories are absent; nothing to compare
    assert (
        target_annotation_dir.is_dir()
    ), f"Target annotation directory not found: {target_annotation_dir}"
    assert (
        source_annotation_dir.is_dir()
    ), f"Source annotation directory not found: {source_annotation_dir}"

    # Count and validate file counts
    source_files = [f for f in source_annotation_dir.glob("*") if f.is_file()]
    target_file_count = len(list(target_annotation_dir.glob("*")))
    source_file_count = len(source_files)

    _assert_file_counts_match(target_file_count, source_file_count, "annotation")

    # Compare each annotation file
    for source_file in source_files:
        target_file = target_annotation_dir / source_file.name
        assert target_file.is_file(), f"File {source_file.name} exists in source but not in target"

        _compare_json_files(target_file, source_file)


def diff_check_data(target_dir: Path, source_dir: Path) -> None:
    """Compare data directories between target and source datasets."""
    target_data_dir = target_dir / T4_FORMAT_DIRECTORY_NAME.DATA.value
    source_data_dir = source_dir / T4_FORMAT_DIRECTORY_NAME.DATA.value

    if not target_data_dir.exists() and not source_data_dir.exists():
        return  # Both data directories are absent; nothing to compare
    assert target_data_dir.is_dir(), f"Target data directory not found: {target_data_dir}"
    assert source_data_dir.is_dir(), f"Source data directory not found: {source_data_dir}"

    # Count and validate file counts
    target_file_count = _count_files_recursively(target_data_dir)
    source_file_count = _count_files_recursively(source_data_dir)
    _assert_file_counts_match(target_file_count, source_file_count, "data")

    # Compare each file in source data directory structure
    for source_file in source_data_dir.rglob("*"):
        if not source_file.is_file():
            continue

        relative_path = source_file.relative_to(source_data_dir)
        target_file = target_data_dir / relative_path

        assert target_file.is_file(), f"File {relative_path} exists in source but not in target"
        if str(relative_path).endswith(JSON_EXTENSION):
            _compare_json_files(target_file, source_file)
        else:
            source_content = source_file.read_bytes()
            target_content = target_file.read_bytes()
            assert source_content == target_content, f"File contents differ: {relative_path}"


def diff_check_rosbag(source_input_bag_path: Path, target_input_bag_path: Path) -> None:
    """Compare ROS bag directories between source and target."""
    # Verify exactly one .db3 file exists in source
    source_db3_files = list(source_input_bag_path.rglob(DB3_FILE_PATTERN))
    assert (
        len(source_db3_files) == 1
    ), f"Expected 1 .db3 file in source, found {len(source_db3_files)}"

    # Verify corresponding .db3 file exists in target
    target_db3_file = target_input_bag_path / source_db3_files[0].name
    assert target_db3_file.is_file(), f"DB3 file {source_db3_files[0].name} not found in target"

    # Load and compare metadata YAML files
    target_metadata_yaml_path = target_input_bag_path / METADATA_YAML_FILENAME
    source_metadata_yaml_path = source_input_bag_path / METADATA_YAML_FILENAME

    assert (
        target_metadata_yaml_path.is_file()
    ), f"Target metadata not found: {target_metadata_yaml_path}"
    assert (
        source_metadata_yaml_path.is_file()
    ), f"Source metadata not found: {source_metadata_yaml_path}"

    target_metadata = _load_yaml_file(target_metadata_yaml_path)
    source_metadata = _load_yaml_file(source_metadata_yaml_path)

    # Python's == operator works well for comparing dicts with standard data types
    assert target_metadata == source_metadata, (
        f"Differences found in {METADATA_YAML_FILENAME}: " f"Metadata files are not equal"
    )


def diff_check_and_validate_t4_datasets(target_dir: Path, source_dir: Path) -> None:
    """
    Comprehensive comparison and validation of T4 dataset directories.

    Compares annotation files, data files, and optionally input_bag directories
    between target and source datasets.
    """
    # Run sanity checks on the datasets to be compared
    sanity_check(target_dir)
    sanity_check(source_dir)

    diff_check_annotation(target_dir, source_dir)
    diff_check_data(target_dir, source_dir)

    # Check input_bag only if it exists in both or neither
    source_has_input_bag = (source_dir / INPUT_BAG_DIR_NAME).exists()
    target_has_input_bag = (target_dir / INPUT_BAG_DIR_NAME).exists()

    assert (
        source_has_input_bag == target_has_input_bag
    ), "Input bag mismatch: one dataset has input_bag while the other doesn't"

    if source_has_input_bag and target_has_input_bag:
        diff_check_rosbag(source_dir / INPUT_BAG_DIR_NAME, target_dir / INPUT_BAG_DIR_NAME)


def diff_check_folder(target_dir: Path, source_dir: Path) -> None:
    """
    Recursively compare all files in two directories for byte-level equality.

    This is a generic folder comparison that checks:
    - Both directories exist
    - File counts match
    - All files exist in both directories
    - File contents are identical byte-by-byte
    """
    assert target_dir.is_dir(), f"Target directory does not exist: {target_dir}"
    assert source_dir.is_dir(), f"Source directory does not exist: {source_dir}"

    # Count and validate file counts
    target_file_count = _count_files_recursively(target_dir)
    source_file_count = _count_files_recursively(source_dir)
    _assert_file_counts_match(target_file_count, source_file_count)

    # Compare each file byte-by-byte
    for source_file in source_dir.rglob("*"):
        if not source_file.is_file():
            continue

        relative_path = source_file.relative_to(source_dir)
        target_file = target_dir / relative_path

        assert target_file.is_file(), f"File {relative_path} exists in source but not in target"

        source_content = source_file.read_bytes()
        target_content = target_file.read_bytes()
        assert source_content == target_content, f"File contents differ: {relative_path}"


def diff_check_json_files(target_file: Path, source_file: Path) -> None:
    """Compare two JSON files after removing token fields."""
    _compare_json_files(target_file, source_file)
