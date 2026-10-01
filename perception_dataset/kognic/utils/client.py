"""Shared Kognic client helpers."""

from typing import Optional


def get_kognic_credentials(config_dict: dict) -> tuple[Optional[str], Optional[str]]:
    """Read organization and workspace IDs from a conversion config.

    Args:
        config_dict (dict): Parsed configuration mapping.

    Returns:
        tuple[Optional[str], Optional[str]]: Organization and workspace IDs.
    """
    conversion = config_dict["conversion"]
    return conversion.get("organization_id"), conversion.get("workspace_id")
