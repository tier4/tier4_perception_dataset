"""Shared Kognic scene lookup helpers."""

from collections import OrderedDict
from typing import List, Optional, Set

from kognic.io.client import KognicIOClient


def resolve_scene_external_ids_to_uuids(
    client: KognicIOClient,
    external_ids: List[str],
    project: Optional[str] = None,
    batch: Optional[str] = None,
) -> "OrderedDict[str, List[str]]":
    """Resolve scene external IDs to scene UUIDs through their inputs.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        external_ids (List[str]): Scene external IDs to resolve.
        project (Optional[str]): Restrict matches to a project external ID.
        batch (Optional[str]): Restrict matches to a batch external ID.

    Returns:
        OrderedDict[str, List[str]]: Matching UUIDs for each requested external
        ID, preserving input ID order and sorting UUIDs within each result.
    """
    resolved: "OrderedDict[str, List[str]]" = OrderedDict(
        (external_id, []) for external_id in external_ids
    )
    if not external_ids:
        return resolved

    inputs = client.input.query_inputs(
        external_ids=external_ids,
        project=project,
        batch=batch,
    )
    matches: "OrderedDict[str, Set[str]]" = OrderedDict(
        (external_id, set()) for external_id in external_ids
    )
    for input_ in inputs:
        if input_.scene_uuid and input_.scene_external_id in matches:
            matches[input_.scene_external_id].add(input_.scene_uuid)

    for external_id, scene_uuids in matches.items():
        resolved[external_id] = sorted(scene_uuids)
    return resolved
