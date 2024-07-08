# flake8: noqa
import json
from typing import Any

__all__ = ("load_json",)


def load_json(filename: str) -> Any:
    """Load json data from specified filepath.

    Args:
        filename (str): File path to .json file.

    Returns:
        Loaded data.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def save_json(data: Any, filename: str) -> None:
    """Save data into json file.

    Args:
        data (Any): Data to be saved.
        filename (str): File path to save as json.
    """
    with open(filename, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
