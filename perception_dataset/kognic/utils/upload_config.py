"""Shared configuration for Kognic upload and scene management commands."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from kognic.auth.credentials_parser import ApiCredentials

from perception_dataset.kognic.utils.client import get_kognic_credentials

# What ``KognicIOClient(auth=...)`` accepts (see ``kognic.auth.resolve_credentials``):
# a path to a credentials JSON, a ``(client_id, client_secret)`` pair, or a
# parsed ``ApiCredentials``. ``None`` means "resolve from the environment".
KognicAuth = Union[str, Tuple[str, str], ApiCredentials]


@dataclass(frozen=True)
class ProjectTarget:
    """A Kognic project an input is created in, with its own optional batch.

    ``pre_annotation`` is the filename (relative to each sequence dir) of the
    OpenLabel pre-annotation to attach for this target. It is optional and
    defaults to ``None``, meaning no pre-annotation is uploaded for this batch.
    Targets that request the same pre-annotation (including ``None``) share a
    single scene; targets requesting different pre-annotations each get their
    own scene, since a pre-annotation is attached scene-wide and applied to all
    inputs created from that scene.
    """

    external_id: str
    batch: Optional[str] = None
    pre_annotation: Optional[str] = None


@dataclass(frozen=True)
class KognicUploadConfig:
    """Configuration for uploading Kognic staging sequences.

    Attributes:
        input_base (Path): Base staging directory.
        organization_id (Optional[str]): Kognic organization identifier.
        workspace_id (Optional[str]): Kognic write-workspace identifier.
        auth (Optional[KognicAuth]): Credentials forwarded to ``KognicIOClient``.
        project_targets (List[ProjectTarget]): Projects and batches to receive inputs.
        dryrun (bool): Whether Kognic should validate without persisting scenes.
        motion_compensate (bool): Whether to enable motion compensation.
        generate_tsv_report (bool): Whether to write an upload outcome report.
        scene_creation_timeout_s (int): Scene-processing timeout in seconds.
        scene_creation_poll_interval_s (int): Status polling interval in seconds.
        pre_annotation_timeout_s (int): Pre-annotation-processing timeout in seconds.
        pre_annotation_poll_interval_s (int): Pre-annotation status polling interval in seconds.
    """

    input_base: Path
    # Both optional: Kognic derives the organization from the auth credentials
    # and infers the write workspace when not provided.
    organization_id: Optional[str] = None
    workspace_id: Optional[str] = None
    # Credentials forwarded to ``KognicIOClient(auth=...)``. ``None`` falls back
    # to the credentials in the environment (``KOGNIC_CREDENTIALS`` or
    # ``KOGNIC_CLIENT_ID``/``KOGNIC_CLIENT_SECRET``).
    auth: Optional[KognicAuth] = None
    project_targets: List[ProjectTarget] = field(default_factory=list)
    dryrun: bool = False
    motion_compensate: bool = False
    generate_tsv_report: bool = False
    scene_creation_timeout_s: int = 3600
    scene_creation_poll_interval_s: int = 10
    pre_annotation_timeout_s: int = 300
    pre_annotation_poll_interval_s: int = 5


def _parse_project_targets(conversion_config: Dict) -> List[ProjectTarget]:
    """Resolve the projects (and their optional batches) a scene's inputs go to.

    Configured via a ``projects`` list. ``batch_external_id`` and
    ``pre_annotation`` are both optional per project: ``batch_external_id``
    defaults to the latest open batch on the Kognic side, and ``pre_annotation``
    defaults to ``None`` (no pre-annotation uploaded for that batch). Listing
    several projects shares one scene across those that request the same
    ``pre_annotation``::

        projects:
          - project_external_id: project_a
            batch_external_id: cuboid_batch
            pre_annotation: pre_annotation.json   # cuboids -> attached to its own scene
          - project_external_id: project_b
            batch_external_id: semseg_batch        # no pre_annotation -> separate, plain scene

    Args:
        conversion_config (Dict): Upload conversion settings.

    Returns:
        List[ProjectTarget]: Validated project targets.

    Raises:
        ValueError: If a project ID is missing or a project/batch pair repeats.
    """
    targets: List[ProjectTarget] = []
    seen: set = set()
    for entry in conversion_config.get("projects") or []:
        if isinstance(entry, str):
            external_id, batch, pre_annotation = entry, None, None
        else:
            external_id = entry.get("project_external_id")
            batch = entry.get("batch_external_id")
            pre_annotation = entry.get("pre_annotation")
        if not external_id or not str(external_id).strip():
            raise ValueError(
                f"conversion.projects entry is missing project_external_id: {entry!r}"
            )
        target = ProjectTarget(
            external_id=str(external_id).strip(), batch=batch, pre_annotation=pre_annotation
        )
        combination = (target.external_id, target.batch)
        if combination in seen:
            raise ValueError(
                "conversion.projects has a duplicate project/batch combination: "
                f"project_external_id={target.external_id!r}, "
                f"batch_external_id={target.batch!r}"
            )
        seen.add(combination)
        targets.append(target)
    return targets


def _parse_auth(conversion_config: Dict) -> Optional[KognicAuth]:
    """Resolve ``conversion.auth`` into something ``KognicIOClient`` accepts.

    YAML can express two of the three forms: a path to a credentials JSON, or a
    ``[client_id, client_secret]`` pair. The pair arrives as a list, which
    ``resolve_credentials`` rejects (it checks for a ``tuple``), so convert it
    here rather than letting it fail as "Bad auth credentials".
    """
    auth = conversion_config.get("auth")
    if auth is None or isinstance(auth, str):
        return auth
    if isinstance(auth, (list, tuple)):
        if len(auth) != 2:
            raise ValueError(
                "conversion.auth as a credentials pair must be "
                f"[client_id, client_secret], got {len(auth)} item(s)"
            )
        return (str(auth[0]), str(auth[1]))
    raise ValueError(
        "conversion.auth must be a path to a credentials JSON or a "
        f"[client_id, client_secret] pair, got {type(auth).__name__}"
    )


def load_upload_config(config_dict: Dict) -> KognicUploadConfig:
    """Load uploader configuration from parsed YAML.

    Args:
        config_dict (Dict): Parsed configuration mapping.

    Returns:
        KognicUploadConfig: Normalized upload settings.
    """
    conversion_config = config_dict["conversion"]
    organization_id, workspace_id = get_kognic_credentials(config_dict)

    return KognicUploadConfig(
        input_base=Path(conversion_config["input_base"]),
        organization_id=organization_id,
        workspace_id=workspace_id,
        auth=_parse_auth(conversion_config),
        project_targets=_parse_project_targets(conversion_config),
        dryrun=conversion_config.get("dryrun", False),
        motion_compensate=conversion_config.get("motion_compensate", False),
        generate_tsv_report=conversion_config.get("generate_tsv_report", False),
        scene_creation_timeout_s=conversion_config.get("scene_creation_timeout_s", 1800),
        scene_creation_poll_interval_s=conversion_config.get("scene_creation_poll_interval_s", 10),
        pre_annotation_timeout_s=conversion_config.get("pre_annotation_timeout_s", 300),
        pre_annotation_poll_interval_s=conversion_config.get("pre_annotation_poll_interval_s", 5),
    )
