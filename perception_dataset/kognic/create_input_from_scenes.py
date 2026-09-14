"""Create Kognic inputs from already-uploaded scenes.

The pre-annotation upload flow (see ``upload_dataset.py``) first creates a
*scene* and only afterwards creates an *input* from it. When that second step
was skipped or failed (see ``delete_scenes.py`` for the cleanup counterpart),
this script performs it after the fact via
:meth:`input.create_from_scene`, optionally attaching a pre-annotation.

An existing pre-annotation can be attached with ``--pre-annotation-uuid``.
Pre-annotation files are uploaded through ``upload_dataset.py``.
When pre-annotation is added, please pass ``--pre-annotation-uuid`` with the corresponding UUID.

Exactly one scene UUID must be provided with ``--scene-uuid``.

Scenes that are failed/invalidated, or that already have an input in the
target project/batch, are skipped; inputs in other project/batch
combinations do not block creation, so the same scene can be sent to
several projects or batches. By default the script only reports what it would do; pass ``--apply``
to actually create the inputs.
"""

import argparse
from typing import List, Optional
from uuid import UUID

from kognic.io.client import KognicIOClient
from kognic.io.model.scene.scene_entry import Scene, SceneStatus
import yaml

from perception_dataset.kognic.utils.client import create_kognic_client
from perception_dataset.kognic.utils.upload_config import load_upload_config
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


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


def create_inputs_from_scenes(
    client: KognicIOClient,
    scenes: List[Scene],
    project: str,
    batch: Optional[str],
    pre_annotation_uuid: Optional[str],
) -> List[str]:
    """Create inputs, optionally attaching an existing pre-annotation.

    Pre-annotation uploading and availability checks are handled by the uploader.
    An input creation failure is recorded for the scene, and processing continues
    with the remaining scenes.

    Returns:
        List[str]: Scene UUIDs whose input creation failed.
    """
    failed: List[str] = []
    for scene in scenes:
        logger.info(
            f"{scene.uuid}: creating input in project {project} "
            f"(batch={batch or 'default'}, "
            f"pre_annotation={pre_annotation_uuid or 'none'})"
        )
        try:
            created_input = client.input.create_from_scene(
                scene_uuid=scene.uuid,
                pre_annotation_uuid=pre_annotation_uuid,
                project=project,
                batch=batch,
            )
            input_uuid = str(created_input.uuid) if created_input else None
            logger.info(f"{scene.uuid}: created input {input_uuid}")
        except Exception as exc:
            logger.error(f"{scene.uuid}: failed to create input: {exc}")
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
        "--scene-uuid",
        type=UUID,
        action="append",
        required=True,
        help="UUID of the single scene to create an input from; specify exactly once.",
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
    parser.add_argument(
        "--pre-annotation-uuid",
        type=str,
        default=None,
        help="Uuid of an already-uploaded pre-annotation to attach to every input.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create the inputs. Without this flag the script only reports.",
    )
    args = parser.parse_args()
    if len(args.scene_uuid) != 1:
        parser.error("--scene-uuid must be specified exactly once")
    scene_uuids = [str(args.scene_uuid[0])]

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    upload_config = load_upload_config(config_dict)

    client = create_kognic_client(
        upload_config.organization_id,
        upload_config.workspace_id,
    )

    logger.info(f"Checking {len(scene_uuids)} candidate scene(s)")
    ready = find_scenes_ready_for_input(client, scene_uuids, args.project, args.batch)
    if not ready:
        logger.info("No scenes ready for input creation; nothing to do.")
        return

    if not args.apply:
        logger.warning(
            f"[DRY RUN] {len(ready)} input(s) would be created in project "
            f"{args.project} (batch={args.batch or 'default'}, "
            f"pre_annotation={args.pre_annotation_uuid or 'none'}): "
            f"{', '.join(scene.uuid for scene in ready)}. Re-run with --apply to create."
        )
        return

    failed = create_inputs_from_scenes(
        client,
        ready,
        args.project,
        args.batch,
        args.pre_annotation_uuid,
    )
    created = len(ready) - len(failed)
    logger.info(f"Created {created}/{len(ready)} input(s) in project {args.project}")
    if failed:
        logger.error(f"Failed scene(s): {', '.join(failed)}")


if __name__ == "__main__":
    main()
