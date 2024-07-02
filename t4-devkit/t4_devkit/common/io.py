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
