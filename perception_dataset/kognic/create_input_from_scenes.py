"""Create Kognic inputs from already-uploaded scenes.

The pre-annotation upload flow (see ``upload_dataset.py``) first creates a
*scene* and only afterwards creates an *input* from it. When that second step
was skipped or failed (see ``delete_scenes.py`` for the cleanup counterpart),
this script performs it after the fact via
:meth:`input.create_from_scene`, optionally attaching a pre-annotation.

The pre-annotation can be given either as ``--pre-annotation-uuid`` (already
uploaded to the scene) or as ``--pre-annotation-file`` (a local OpenLabel
JSON, e.g. ``semseg_pre_annotation.json``); a file is uploaded to each scene
first and the resulting uuid is attached to that scene's input.

Candidate scene uuids are taken from a source you provide:

* ``--scene-uuids`` — an explicit comma/space separated list, or
* ``--dataset-id-json`` — a ``dataset_id.json`` as written by the uploader.

Scenes that are failed/invalidated, or that already have an input in the
target project/batch, are skipped; inputs in other project/batch
combinations do not block creation, so the same scene can be sent to
several projects or batches. By default the script only reports what it would do; pass ``--apply``
to actually create the inputs.
"""

import argparse
import json
from pathlib import Path
import time
from typing import List, Optional, Set

from kognic.io.client import KognicIOClient
from kognic.io.model.scene.scene_entry import Scene, SceneStatus
from kognic.openlabel.models import OpenLabelAnnotation
import yaml

from perception_dataset.kognic.upload_dataset import _load_upload_config
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


def _collect_scene_uuids(explicit: Optional[str], dataset_id_json: Optional[Path]) -> List[str]:
    """Gather candidate scene UUIDs from command-line sources.

    Args:
        explicit (Optional[str]): Comma- or space-separated scene UUIDs.
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
        for token in explicit.replace(",", " ").split():
            _add(token)

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


def find_scenes_ready_for_input(
    client: KognicIOClient,
    scene_uuids: List[str],
    project: str,
    batch: Optional[str],
) -> List[Scene]:
    """Return the subset of *scene_uuids* that exist, are live and have no input
    in the target *project*/*batch*.

    A scene may hold inputs in several project/batch combinations; only an
    existing input in the same project (and batch, when one is given — with
    ``batch=None`` any input in the project counts, since the default batch
    cannot be queried by name) makes it a duplicate and skips the scene.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        scene_uuids (List[str]): Candidate scene UUIDs.
        project (str): Target project external ID.
        batch (Optional[str]): Target batch, or ``None`` for any project input.

    Returns:
        List[Scene]: Live scenes without an input in the target scope.
    """
    if not scene_uuids:
        return []

    scenes = {s.uuid: s for s in client.scene.get_scenes_by_uuids(scene_uuids=scene_uuids)}

    ready: List[Scene] = []
    for scene_uuid in scene_uuids:
        scene = scenes.get(scene_uuid)
        if scene is None:
            logger.warning(f"{scene_uuid}: not found; skipping")
            continue
        status = str(scene.status)
        if scene.status == SceneStatus.Failed or status.startswith("invalidated"):
            logger.info(f"{scene_uuid}: status={status}; skipping")
            continue
        duplicates = client.input.query_inputs(
            scene_uuids=[scene_uuid], project=project, batch=batch
        )
        if duplicates:
            logger.info(
                f"{scene_uuid}: already has {len(duplicates)} input(s) in project "
                f"{project} (batch={batch or 'any'}); skipping"
            )
            continue
        others = client.input.query_inputs(scene_uuids=[scene_uuid])
        if others:
            logger.info(
                f"{scene_uuid}: has {len(others)} input(s) in other project/batch "
                "combinations; creating another one here"
            )
        ready.append(scene)

    return ready


def load_pre_annotation(pre_annotation_path: Path) -> OpenLabelAnnotation:
    """Load and validate an OpenLABEL pre-annotation file.

    Args:
        pre_annotation_path (Path): Path to the OpenLABEL JSON file.

    Returns:
        OpenLabelAnnotation: Validated pre-annotation model.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the JSON does not match the OpenLABEL model.
    """
    if not pre_annotation_path.exists():
        raise FileNotFoundError(f"pre-annotation file not found: {pre_annotation_path}")

    with open(pre_annotation_path) as f:
        pre_annotation = OpenLabelAnnotation.model_validate(json.load(f))

    frames = pre_annotation.openlabel.frames or {}
    objects = pre_annotation.openlabel.objects or {}
    logger.info(f"Loaded {pre_annotation_path}: {len(objects)} objects over {len(frames)} frames")
    return pre_annotation


# Pre-annotation statuses that mean server-side processing has not finished
# yet. Lifecycle: created -> processing -> indexed | failed ("Pre-annotations
# can only be deleted from status=Indexed" pins indexed as the success state).
_PRE_ANNOTATION_PENDING_STATUSES = {
    "created",
    "pending",
    "pending_for_scene",
    "processing",
    "registered",
    "importing",
}


def _wait_for_pre_annotation(
    client: KognicIOClient,
    pre_annotation_uuid: str,
    timeout_s: float = 120.0,
    poll_s: float = 5.0,
) -> dict:
    """Poll until the pre-annotation leaves processing; return its final record.

    Uploading a pre-annotation only queues it: Kognic processes it
    asynchronously (sometimes only once an input references it) and an input
    created against one that later flips to ``failed`` is silently dropped.
    Raises ``RuntimeError`` on failure, with the full record in the message
    since it may carry the server's error details. On timeout the last record
    is returned with a warning — input creation will then surface the verdict.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        pre_annotation_uuid (str): Uploaded pre-annotation UUID.
        timeout_s (float): Maximum polling duration in seconds.
        poll_s (float): Delay between status requests in seconds.

    Returns:
        dict: Final or last observed pre-annotation record.

    Raises:
        RuntimeError: If the record is missing or processing fails.
    """
    deadline = time.time() + timeout_s
    while True:
        records = client.pre_annotation.list(ids=[pre_annotation_uuid])
        if not records:
            raise RuntimeError(f"pre-annotation {pre_annotation_uuid} not found")
        record = records[0]
        status = str(record.get("status", "")).lower()
        if status == "failed":
            raise RuntimeError(
                f"pre-annotation {pre_annotation_uuid} failed server-side "
                f"processing: {json.dumps(record, default=str)}"
            )
        if status not in _PRE_ANNOTATION_PENDING_STATUSES:
            return record
        if time.time() >= deadline:
            logger.warning(
                f"pre-annotation {pre_annotation_uuid} still {status} after "
                f"{timeout_s:.0f}s; proceeding (input creation will surface the "
                f"verdict): {json.dumps(record, default=str)}"
            )
            return record
        logger.info(f"pre-annotation {pre_annotation_uuid}: status={status}; waiting")
        time.sleep(poll_s)


def create_inputs_from_scenes(
    client: KognicIOClient,
    scenes: List[Scene],
    project: str,
    batch: Optional[str],
    pre_annotation_uuid: Optional[str],
    pre_annotation: Optional[OpenLabelAnnotation] = None,
    pre_annotation_name: Optional[str] = None,
) -> List[str]:
    """Create one input per scene; a failure on one scene does not stop the rest.

    When *pre_annotation* (a loaded OpenLabel file) is given, it is uploaded to
    each scene first — a pre-annotation is bound to one scene — and the created
    uuid is attached to that scene's input. If the scene already has a
    pre-annotation with the same external id (e.g. from an earlier run whose
    input creation failed), it is reused when live; when it failed server-side
    processing the file is re-uploaded under a fresh ``-r<N>`` suffixed
    external id, since Kognic refuses to delete failed pre-annotations
    ("can only be deleted from status=Indexed") yet external ids must be
    unique per scene. The input is only created once the
    pre-annotation finished processing, since an input referencing one that
    later fails is silently dropped by Kognic. Otherwise *pre_annotation_uuid*
    (or no pre-annotation at all) is attached as-is.

    Returns the uuids of scenes whose pre-annotation upload or input creation
    failed.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        scenes (List[Scene]): Scenes for which to create inputs.
        project (str): Target project external ID.
        batch (Optional[str]): Target batch.
        pre_annotation_uuid (Optional[str]): Existing pre-annotation UUID.
        pre_annotation (Optional[OpenLabelAnnotation]): Local pre-annotation
            to upload separately for each scene.
        pre_annotation_name (Optional[str]): Name used in uploaded external IDs.

    Returns:
        List[str]: Scene UUIDs whose upload or input creation failed.
    """
    failed: List[str] = []
    for scene in scenes:
        scene_pre_annotation_uuid = pre_annotation_uuid
        if pre_annotation is not None:
            base_external_id = f"{scene.external_id}-{pre_annotation_name}-pre-annotation"
            try:
                taken = {
                    record.get("externalId"): record
                    for record in client.pre_annotation.list(scene_uuids=[scene.uuid])
                }
                existing = taken.get(base_external_id)
                if existing and str(existing.get("status", "")).lower() != "failed":
                    scene_pre_annotation_uuid = existing["id"]
                    logger.info(
                        f"{scene.uuid}: reusing existing pre-annotation "
                        f"{scene_pre_annotation_uuid} ({base_external_id}, "
                        f"status={existing.get('status')})"
                    )
                else:
                    external_id = base_external_id
                    if existing:
                        # Failed pre-annotations cannot be deleted, and external
                        # ids are unique per scene: retry under a fresh suffix.
                        retry = 2
                        while f"{base_external_id}-r{retry}" in taken:
                            retry += 1
                        external_id = f"{base_external_id}-r{retry}"
                        logger.warning(
                            f"{scene.uuid}: existing pre-annotation {existing['id']} "
                            f"({base_external_id}) failed server-side processing; "
                            f"re-uploading as {external_id}. Failed record: "
                            f"{json.dumps(existing, default=str)}"
                        )
                    created = client.pre_annotation.create(
                        scene_uuid=scene.uuid,
                        pre_annotation=pre_annotation,
                        external_id=external_id,
                        dryrun=False,
                    )
                    scene_pre_annotation_uuid = created.id
                    logger.info(
                        f"{scene.uuid}: uploaded pre-annotation {scene_pre_annotation_uuid}"
                    )
                _wait_for_pre_annotation(client, scene_pre_annotation_uuid)
            except Exception as exc:
                logger.error(f"{scene.uuid}: failed to upload pre-annotation: {exc}")
                failed.append(scene.uuid)
                continue

        logger.info(
            f"{scene.uuid}: creating input in project {project} "
            f"(batch={batch or 'default'}, "
            f"pre_annotation={scene_pre_annotation_uuid or 'none'})"
        )
        try:
            created_input = client.input.create_from_scene(
                scene_uuid=scene.uuid,
                pre_annotation_uuid=scene_pre_annotation_uuid,
                project=project,
                batch=batch,
            )
            input_uuid = str(created_input.uuid) if created_input else None
            logger.info(f"{scene.uuid}: created input {input_uuid}")
        except Exception as exc:
            logger.error(f"{scene.uuid}: failed to create input: {exc}")
            if pre_annotation is not None:
                logger.error(
                    f"{scene.uuid}: its pre-annotation {scene_pre_annotation_uuid} was "
                    "uploaded; retry with --pre-annotation-uuid to avoid re-uploading"
                )
            failed.append(scene.uuid)
    return failed


def main():
    """Run the create-inputs command-line interface.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="config/upload_kognic_dataset_sample.yaml",
        help="Upload config yaml; provides organization_id, workspace_id.",
    )
    parser.add_argument(
        "--scene-uuids",
        type=str,
        default=None,
        help="Explicit comma/space separated scene uuids to create inputs for.",
    )
    parser.add_argument(
        "--dataset-id-json",
        type=str,
        default=None,
        help="Path to a dataset_id.json as written by the uploader.",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="External id of the Kognic project to create the inputs in.",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Batch within the project; omit for the project default batch.",
    )
    pre_annotation_group = parser.add_mutually_exclusive_group()
    pre_annotation_group.add_argument(
        "--pre-annotation-uuid",
        type=str,
        default=None,
        help="Uuid of an already-uploaded pre-annotation to attach to every input.",
    )
    pre_annotation_group.add_argument(
        "--pre-annotation-file",
        type=str,
        default=None,
        help=(
            "Path to an OpenLabel pre-annotation JSON (e.g. "
            "semseg_pre_annotation.json); it is uploaded to each scene and "
            "attached to that scene's input."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create the inputs. Without this flag the script only reports.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    upload_config = _load_upload_config(config_dict)

    dataset_id_json: Optional[Path] = None
    if args.dataset_id_json:
        dataset_id_json = Path(args.dataset_id_json)

    scene_uuids = _collect_scene_uuids(args.scene_uuids, dataset_id_json)
    if not scene_uuids:
        logger.warning(
            "No candidate scene uuids found. Provide --scene-uuids or a dataset_id.json."
        )
        return

    # Load (and fail fast on) the pre-annotation file before touching the API.
    pre_annotation: Optional[OpenLabelAnnotation] = None
    pre_annotation_name: Optional[str] = None
    if args.pre_annotation_file:
        pre_annotation_path = Path(args.pre_annotation_file)
        pre_annotation = load_pre_annotation(pre_annotation_path)
        pre_annotation_name = pre_annotation_path.stem

    client = KognicIOClient(
        client_organization_id=upload_config.organization_id,
        write_workspace_id=upload_config.workspace_id,
    )

    logger.info(f"Checking {len(scene_uuids)} candidate scene(s)")
    ready = find_scenes_ready_for_input(client, scene_uuids, args.project, args.batch)
    if not ready:
        logger.info("No scenes ready for input creation; nothing to do.")
        return

    if not args.apply:
        pre_annotation_desc = (
            f"upload of {args.pre_annotation_file}"
            if pre_annotation is not None
            else args.pre_annotation_uuid or "none"
        )
        logger.warning(
            f"[DRY RUN] {len(ready)} input(s) would be created in project "
            f"{args.project} (batch={args.batch or 'default'}, "
            f"pre_annotation={pre_annotation_desc}): "
            f"{', '.join(scene.uuid for scene in ready)}. Re-run with --apply to create."
        )
        return

    failed = create_inputs_from_scenes(
        client,
        ready,
        args.project,
        args.batch,
        args.pre_annotation_uuid,
        pre_annotation=pre_annotation,
        pre_annotation_name=pre_annotation_name,
    )
    created = len(ready) - len(failed)
    logger.info(f"Created {created}/{len(ready)} input(s) in project {args.project}")
    if failed:
        logger.error(f"Failed scene(s): {', '.join(failed)}")


if __name__ == "__main__":
    main()
