"""Shared Kognic client helpers."""

from typing import Optional

from kognic.io.client import KognicIOClient


def create_kognic_client(
    organization_id: Optional[str], workspace_id: Optional[str]
) -> KognicIOClient:
    """Create a Kognic client configured for an organization and workspace.

    Args:
        organization_id (Optional[str]): Kognic organization identifier.
        workspace_id (Optional[str]): Kognic write-workspace identifier.

    Returns:
        KognicIOClient: Configured Kognic client.
    """
    return KognicIOClient(
        client_organization_id=organization_id,
        write_workspace_id=workspace_id,
    )


def get_kognic_credentials(config_dict: dict) -> tuple[Optional[str], Optional[str]]:
    """Read organization and workspace IDs from a conversion config.

    Both the canonical keys and their legacy aliases are supported.

    Args:
        config_dict (dict): Parsed configuration mapping.

    Returns:
        tuple[Optional[str], Optional[str]]: Organization and workspace IDs.
    """
    conversion = config_dict["conversion"]
    organization_id = conversion.get("organization_id") or conversion.get(
        "client_organization_id"
    )
    workspace_id = conversion.get("workspace_id") or conversion.get(
        "write_workspace_id"
    )
    return organization_id, workspace_id
