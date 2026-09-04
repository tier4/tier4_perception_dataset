"""Delete (invalidate) Kognic scenes that have no input.

The pre-annotation upload flow (see ``upload_dataset.py``) first creates a
*scene* and only afterwards creates an *input* from it. If that second step
fails, the scene lingers server-side as an orphan: it exists but no input
references it, so labelers never see it. Kognic has no hard scene delete; the
equivalent is :meth:`scene.invalidate_scenes`, which this script applies to
every orphaned scene it finds.

There is no "list all scenes" endpoint, so candidate scene uuids are taken from
a source you provide:

* ``--scene-uuids`` — an explicit comma/space separated list, or a path to a
  csv file with a ``scene_uuid`` column,
* ``--scene-external-ids`` — an explicit comma/space separated list of scene
  external ids, or a path to a csv file with a ``scene_external_id`` column.
  External ids are resolved to scene uuids via ``input.query_inputs``, so this
  only works for scenes that have an input (orphan scenes must be addressed by
  uuid), or
* ``--dataset-id-json`` — a ``dataset_id.json`` as written by the uploader
  (defaults to ``<input_base>/dataset_id.json`` from the upload config).

A scene is considered to have *no input* when ``input.query_inputs`` returns
nothing for it. Already-invalidated or non-existent scenes are skipped.

By default only empty (input-less) scenes are invalidated. Pass
``--delete-input`` to instead process *every* candidate scene: its inputs are
destructively deleted (:meth:`input.delete_input`) and the scene is then
invalidated. This is useful when you want to fully retract scenes that already
have inputs, not just clean up orphans.

By default the script only reports what it would do; pass ``--apply`` to
actually invalidate.
"""

import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
from typing import List, Optional, Set

from kognic.io.client import KognicIOClient
from kognic.io.model.scene.invalidated_reason import SceneInvalidatedReason
from kognic.io.model.scene.scene_entry import SceneStatus
from requests.exceptions import HTTPError
import yaml

from perception_dataset.kognic.upload_dataset import _load_upload_config
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


def _collect_values(explicit: str, csv_column: str) -> List[str]:
    """Parse values from text or a CSV column.

    Args:
        explicit (str): Comma- or space-separated values, or a CSV path.
        csv_column (str): Required column when ``explicit`` names a CSV file.

    Returns:
        List[str]: Parsed values in source order.

    Raises:
        ValueError: If a CSV file lacks the required column.
    """
    values: List[str] = []
    explicit_path = Path(explicit)
    if explicit_path.suffix.lower() == ".csv" and explicit_path.is_file():
        with open(explicit_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or csv_column not in reader.fieldnames:
                raise ValueError(
                    f"{explicit_path}: csv must have a '{csv_column}' column "
                    f"(found: {reader.fieldnames})"
                )
            for row in reader:
                values.append(row[csv_column] or "")
    else:
        values.extend(explicit.replace(",", " ").split())
    return values


def _collect_scene_uuids(
    explicit: Optional[str], dataset_id_json: Optional[Path]
) -> List[str]:
    """Gather candidate scene UUIDs from command-line sources.

    Args:
        explicit (Optional[str]): Scene UUID text or CSV path.
        dataset_id_json (Optional[Path]): Uploader-generated dataset ID file.

    Returns:
        List[str]: Unique scene UUIDs in discovery order.
    """
    uuids: List[str] = []
    seen: Set[str] = set()

    def _add(value: str) -> None:
        """Add a nonempty, unique scene UUID to the result.

        Args:
            value (str): Candidate scene UUID.

        Returns:
            None
        """
        value = value.strip()
        if value and value != "dryrun" and value not in seen:
            seen.add(value)
            uuids.append(value)

    if explicit:
        for value in _collect_values(explicit, csv_column="scene_uuid"):
            _add(value)

    if dataset_id_json is not None:
        if not dataset_id_json.exists():
            raise FileNotFoundError(f"dataset id file not found: {dataset_id_json}")
        with open(dataset_id_json) as f:
            data = json.load(f)
        # Current form: {external_id: {"scene_id": ..., "inputs": [...]}}.
        # Legacy form:  {dataset_name: scene_uuid}.
        for value in data.values():
            if isinstance(value, dict):
                scene_uuid = value.get("scene_id")
                if isinstance(scene_uuid, str):
                    _add(scene_uuid)
            elif isinstance(value, str):
                _add(value)

    return uuids


def _resolve_external_ids_to_scene_uuids(
    client: KognicIOClient, external_ids: List[str]
) -> List[str]:
    """Resolve scene external ids to scene uuids via the inputs that reference them.

    Kognic has no scene-by-external-id lookup, so scenes without any input
    cannot be resolved this way; those external ids are skipped with a warning.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        external_ids (List[str]): Scene external IDs to resolve.

    Returns:
        List[str]: Resolved scene UUIDs in input order.
    """
    if not external_ids:
        return []

    by_external_id: "OrderedDict[str, Set[str]]" = OrderedDict()
    for input_ in client.input.query_inputs(external_ids=external_ids):
        by_external_id.setdefault(input_.scene_external_id, set()).add(input_.scene_uuid)

    uuids: List[str] = []
    for external_id in external_ids:
        scene_uuids = by_external_id.get(external_id)
        if not scene_uuids:
            logger.warning(
                f"{external_id}: no input found with this external id; cannot "
                "resolve to a scene uuid (orphan scenes must be given by uuid), skipping"
            )
            continue
        for scene_uuid in sorted(scene_uuids):
            logger.info(f"{external_id}: resolved to scene {scene_uuid}")
            uuids.append(scene_uuid)

    return uuids


def _input_uuids(client: KognicIOClient, scene_uuid: str) -> List[str]:
    """Get UUIDs of inputs associated with a scene.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        scene_uuid (str): Scene UUID to query.

    Returns:
        List[str]: Input UUIDs associated with the scene.
    """
    inputs = client.input.query_inputs(scene_uuids=[scene_uuid])
    return [i.uuid for i in inputs]


def _live_scene_uuids(
    client: KognicIOClient, scene_uuids: List[str]
) -> List[str]:
    """Filter candidate UUIDs to live scenes.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        scene_uuids (List[str]): Candidate scene UUIDs.

    Returns:
        List[str]: UUIDs of scenes that exist and can be invalidated.
    """
    scenes = {s.uuid: s for s in client.scene.get_scenes_by_uuids(scene_uuids=scene_uuids)}

    live: List[str] = []
    for scene_uuid in scene_uuids:
        scene = scenes.get(scene_uuid)
        if scene is None:
            logger.warning(f"{scene_uuid}: not found (already deleted?); skipping")
            continue
        status = str(scene.status)
        if scene.status == SceneStatus.Failed or status.startswith("invalidated"):
            logger.info(f"{scene_uuid}: status={status}; already gone, skipping")
            continue
        if scene.status in (SceneStatus.Pending, SceneStatus.Processing):
            logger.warning(
                f"{scene_uuid}: status={status}; cannot be invalidated until scene "
                "creation finishes (becomes created or failed), skipping"
            )
            continue
        logger.info(f"{scene_uuid}: status={status}")
        live.append(scene_uuid)

    return live


def find_scenes_without_input(
    client: KognicIOClient, scene_uuids: List[str]
) -> List[str]:
    """Find live scenes that have no input.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        scene_uuids (List[str]): Candidate scene UUIDs.

    Returns:
        List[str]: UUIDs of live scenes without inputs.
    """
    if not scene_uuids:
        return []

    orphans: List[str] = []
    for scene_uuid in _live_scene_uuids(client, scene_uuids):
        if _input_uuids(client, scene_uuid):
            logger.info(f"{scene_uuid}: has an input; keeping")
            continue
        logger.info(f"{scene_uuid}: no input -> orphan")
        orphans.append(scene_uuid)

    return orphans


def find_scenes_with_inputs(
    client: KognicIOClient, scene_uuids: List[str]
) -> "OrderedDict[str, List[str]]":
    """Map live scene UUIDs to their input UUIDs.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        scene_uuids (List[str]): Candidate scene UUIDs.

    Returns:
        OrderedDict[str, List[str]]: Input UUIDs keyed by live scene UUID.
    """
    result: "OrderedDict[str, List[str]]" = OrderedDict()
    if not scene_uuids:
        return result

    for scene_uuid in _live_scene_uuids(client, scene_uuids):
        input_uuids = _input_uuids(client, scene_uuid)
        logger.info(f"{scene_uuid}: {len(input_uuids)} input(s) to delete")
        result[scene_uuid] = input_uuids

    return result


def main():
    """Run the scene-deletion command-line interface.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="config/upload_kognic_dataset_sample.yaml",
        help="Upload config yaml; provides organization_id, workspace_id, input_base.",
    )
    parser.add_argument(
        "--scene-uuids",
        type=str,
        default=None,
        help="Explicit comma/space separated scene uuids to consider, or a path "
        "to a csv file with a 'scene_uuid' column.",
    )
    parser.add_argument(
        "--scene-external-ids",
        type=str,
        default=None,
        help="Explicit comma/space separated scene external ids to consider, or "
        "a path to a csv file with a 'scene_external_id' column. Resolved to "
        "scene uuids via their inputs, so only scenes that have an input can "
        "be found this way.",
    )
    parser.add_argument(
        "--dataset-id-json",
        type=str,
        default=None,
        help="Path to a dataset_id.json. Defaults to <input_base>/dataset_id.json. "
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=SceneInvalidatedReason.INCORRECTLY_CREATED.value,
        choices=[r.value for r in SceneInvalidatedReason],
        help="Invalidation reason recorded on Kognic.",
    )
    parser.add_argument(
        "--delete-input",
        action="store_true",
        help="Also delete inputs: process every candidate scene (not just empty "
        "ones), destructively delete its inputs, then invalidate the scene.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually invalidate the scenes. Without this flag the script only reports.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    upload_config = _load_upload_config(config_dict)

    dataset_id_json: Optional[Path] = None
    if args.dataset_id_json:
        dataset_id_json = Path(args.dataset_id_json)

    scene_uuids = _collect_scene_uuids(args.scene_uuids, dataset_id_json)

    external_ids: List[str] = []
    if args.scene_external_ids:
        seen_external_ids: Set[str] = set()
        for value in _collect_values(args.scene_external_ids, csv_column="scene_external_id"):
            value = value.strip()
            if value and value not in seen_external_ids:
                seen_external_ids.add(value)
                external_ids.append(value)

    if not scene_uuids and not external_ids:
        logger.warning(
            "No candidate scenes found. Provide --scene-uuids, --scene-external-ids "
            "or a dataset_id.json."
        )
        return

    client = KognicIOClient(
        client_organization_id=upload_config.organization_id,
        write_workspace_id=upload_config.workspace_id,
    )

    if external_ids:
        logger.info(f"Resolving {len(external_ids)} scene external id(s) to scene uuids")
        for scene_uuid in _resolve_external_ids_to_scene_uuids(client, external_ids):
            if scene_uuid not in scene_uuids:
                scene_uuids.append(scene_uuid)

    if not scene_uuids:
        logger.warning("No scene uuids resolved; nothing to do.")
        return

    reason = SceneInvalidatedReason(args.reason)

    if args.delete_input:
        _delete_inputs_and_scenes(client, scene_uuids, reason, apply=args.apply)
    else:
        _delete_empty_scenes(client, scene_uuids, reason, apply=args.apply)


def _delete_empty_scenes(
    client: KognicIOClient,
    scene_uuids: List[str],
    reason: SceneInvalidatedReason,
    apply: bool,
) -> None:
    """Invalidate candidate scenes that have no inputs.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        scene_uuids (List[str]): Candidate scene UUIDs.
        reason (SceneInvalidatedReason): Reason recorded by Kognic.
        apply (bool): Whether to perform invalidation instead of a dry run.

    Returns:
        None
    """
    logger.info(f"Checking {len(scene_uuids)} candidate scene(s) for missing inputs")

    orphans = find_scenes_without_input(client, scene_uuids)
    if not orphans:
        logger.info("No scenes without an input found; nothing to delete.")
        return

    if not apply:
        logger.warning(
            f"[DRY RUN] {len(orphans)} scene(s) would be invalidated (reason={reason.value}): "
            f"{', '.join(orphans)}. Re-run with --apply to delete."
        )
        return

    logger.info(f"Invalidating {len(orphans)} scene(s) with reason {reason.value}")
    invalidated: List[str] = []
    failed: List[str] = []
    for scene_uuid in orphans:
        try:
            client.scene.invalidate_scenes(scene_uuids=[scene_uuid], reason=reason)
        except HTTPError as e:
            logger.error(f"{scene_uuid}: invalidate failed ({e}); continuing")
            failed.append(scene_uuid)
            continue
        logger.info(f"{scene_uuid}: invalidated")
        invalidated.append(scene_uuid)

    logger.info(f"Invalidated {len(invalidated)}/{len(orphans)} scene(s)")
    if failed:
        logger.warning(f"Failed to invalidate {len(failed)} scene(s): {', '.join(failed)}")


def _delete_inputs_and_scenes(
    client: KognicIOClient,
    scene_uuids: List[str],
    reason: SceneInvalidatedReason,
    apply: bool,
) -> None:
    """Delete inputs and invalidate their candidate scenes.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        scene_uuids (List[str]): Candidate scene UUIDs.
        reason (SceneInvalidatedReason): Reason recorded by Kognic.
        apply (bool): Whether to perform deletion instead of a dry run.

    Returns:
        None
    """
    logger.info(
        f"Checking {len(scene_uuids)} candidate scene(s); inputs will be deleted too"
    )

    targets = find_scenes_with_inputs(client, scene_uuids)
    if not targets:
        logger.info("No live scenes found; nothing to delete.")
        return

    total_inputs = sum(len(v) for v in targets.values())

    if not apply:
        logger.warning(
            f"[DRY RUN] {total_inputs} input(s) would be deleted and "
            f"{len(targets)} scene(s) invalidated (reason={reason.value}): "
            f"{', '.join(targets)}. Re-run with --apply to delete."
        )
        return

    logger.info(
        f"Deleting {total_inputs} input(s) and invalidating {len(targets)} scene(s) "
        f"with reason {reason.value}"
    )
    for scene_uuid, input_uuids in targets.items():
        for input_uuid in input_uuids:
            logger.info(f"{scene_uuid}: deleting input {input_uuid}")
            client.input.delete_input(input_uuid=input_uuid)
        client.scene.invalidate_scenes(scene_uuids=[scene_uuid], reason=reason)
        logger.info(f"{scene_uuid}: invalidated")

    logger.info(
        f"Deleted {total_inputs} input(s) and invalidated {len(targets)} scene(s): "
        f"{', '.join(targets)}"
    )


if __name__ == "__main__":
    main()
