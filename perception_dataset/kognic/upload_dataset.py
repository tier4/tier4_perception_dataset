"""Upload converted T4 staging sequences to Kognic."""

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, Generator, List, Optional, Tuple
import uuid

from kognic.io.client import KognicIOClient
import kognic.io.model as KognicModel
from kognic.io.model.ego.imu_data import IMUData
from kognic.io.model.scene.feature_flags import FeatureFlags
from kognic.io.model.scene.invalidated_reason import SceneInvalidatedReason
from kognic.io.model.scene.lidars_and_cameras_sequence.frame import (
    Frame as LidarsAndCamerasSequenceFrame,
)
from kognic.io.model.scene.metadata.metadata import FrameMetaData, MetaData
from kognic.io.model.scene.resources.image import ImageMetadata
from kognic.io.model.scene.scene_entry import SceneStatus
from kognic.openlabel.models.models import OpenLabelAnnotation
import numpy as np
from requests.exceptions import HTTPError
import yaml

from perception_dataset.constants import (
    IMU_EXTRAPOLATE_S,
    IMU_TARGET_HZ,
    PREFERRED_CAMERA_SENSORS,
    PREFERRED_LIDAR_SENSORS,
)
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

SceneUUID = str

_UPLOAD_REPORT_FILENAME = "upload_report.tsv"
_UPLOAD_REPORT_FIELDS = (
    "scene",
    "status",
    "stage",
    "project",
    "batch",
    "scene_uuid",
    "input_id",
    "invalidated",
    "error_code",
    "error_type",
    "error_message",
    "duration_seconds",
)


class SceneInputError(RuntimeError):
    """A scene was created on Kognic but a later step failed.

    Once ``lidars_and_cameras_sequence.create()`` returns a ``scene_uuid`` the
    scene is persisted server-side. If the pre-annotation upload or input
    creation then fails, the scene would linger as an orphan (no input, invisible
    to labelers), so ``upload_one`` invalidates it before raising this. The
    ``scene_uuid`` is kept for logging/traceability only.
    """

    def __init__(
        self,
        external_id: str,
        scene_uuid: str,
        stage: str,
        cause: BaseException,
        invalidated: bool = False,
        failed_input_errors: Optional[List[Tuple["ProjectTarget", BaseException]]] = None,
    ):
        """Initialize an error for a failed post-creation stage.

        Args:
            external_id (str): Scene external ID.
            scene_uuid (str): UUID of the created scene.
            stage (str): Post-creation stage that failed.
            cause (BaseException): Original failure.
            invalidated (bool): Whether orphan cleanup succeeded.
            failed_input_errors: Original per-project input failures, when the
                scene failed because every input creation failed.
        """
        super().__init__(f"{external_id}: scene {scene_uuid} created but {stage} failed: {cause}")
        self.external_id = external_id
        self.scene_uuid = scene_uuid
        self.stage = stage
        self.cause = cause
        # Whether the orphaned scene was successfully invalidated during cleanup.
        # If False the scene remains on Kognic and needs manual invalidation.
        self.invalidated = invalidated
        self.failed_input_errors = failed_input_errors or []


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


@dataclass
class SceneUploadResult:
    """Outcome of creating one scene (and its inputs) for a sequence."""

    external_id: str
    scene_uuid: Optional[SceneUUID]
    # One entry per created input: {"project_name", "batch_name", "input_id"}.
    inputs: List[Dict[str, Optional[str]]] = field(default_factory=list)
    # ``project/batch`` of inputs that failed while the scene itself succeeded.
    failed_inputs: List[str] = field(default_factory=list)
    # Project/batch target and the original exception for each failed input.
    failed_input_errors: List[Tuple[ProjectTarget, BaseException]] = field(default_factory=list)
    failed: bool = False
    # When ``failed``: whether the orphaned scene was successfully invalidated.
    invalidated: bool = False
    error: Optional[BaseException] = None


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
_PRE_ANNOTATION_SUCCESS_STATUS = "indexed"


def _wait_for_pre_annotation(
    client: KognicIOClient,
    pre_annotation_uuid: str,
    timeout_s: float = 120.0,
    poll_s: float = 5.0,
    raise_on_timeout: bool = False,
) -> dict:
    """Poll until the pre-annotation leaves processing; return its final record.

    Uploading a pre-annotation only queues it: Kognic processes it
    asynchronously and an input created against one that later fails is
    silently dropped, so a successful upload response alone cannot be treated
    as final success. Any status outside the known pending/failed/success set
    is treated as a processing failure rather than silently accepted, since the
    documented lifecycle only has those three outcomes.

    Args:
        client (KognicIOClient): Authenticated Kognic client.
        pre_annotation_uuid (str): Uploaded pre-annotation UUID.
        timeout_s (float): Maximum polling duration in seconds.
        poll_s (float): Delay between status requests in seconds.
        raise_on_timeout (bool): Whether a still-pending status at the deadline
            raises instead of returning the last observed (pending) record.

    Returns:
        dict: Final pre-annotation record (status ``indexed``), or the last
            observed record on a non-raising timeout.

    Raises:
        RuntimeError: If the record is missing, processing fails, or an
            unrecognized status is observed.
        TimeoutError: If ``raise_on_timeout`` and processing does not finish
            within ``timeout_s``.
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
        if status == _PRE_ANNOTATION_SUCCESS_STATUS:
            return record
        if status not in _PRE_ANNOTATION_PENDING_STATUSES:
            raise RuntimeError(
                f"pre-annotation {pre_annotation_uuid} entered unrecognized status "
                f"{status!r} (expected one of {sorted(_PRE_ANNOTATION_PENDING_STATUSES)} "
                f"or {_PRE_ANNOTATION_SUCCESS_STATUS!r}): {json.dumps(record, default=str)}"
            )
        if time.time() >= deadline:
            message = (
                f"pre-annotation {pre_annotation_uuid} still {status} after {timeout_s:.0f}s"
            )
            if raise_on_timeout:
                raise TimeoutError(
                    f"{message}; not attaching it to an input: "
                    f"{json.dumps(record, default=str)}"
                )
            logger.warning(
                f"{message}; proceeding (input creation will surface the verdict): "
                f"{json.dumps(record, default=str)}"
            )
            return record
        logger.info(f"pre-annotation {pre_annotation_uuid}: status={status}; waiting")
        time.sleep(poll_s)


@dataclass(frozen=True)
class KognicUploadConfig:
    """Configuration for uploading Kognic staging sequences.

    Attributes:
        input_base (Path): Base staging directory.
        organization_id (Optional[str]): Kognic organization identifier.
        workspace_id (Optional[str]): Kognic write-workspace identifier.
        project_targets (List[ProjectTarget]): Projects and batches to receive inputs.
        dryrun (bool): Whether Kognic should validate without persisting scenes.
        motion_compensate (bool): Whether to enable motion compensation.
        include_imu_data (bool): Whether to derive dense IMU pose data.
        write_debug_frames (bool): Whether to save serialized frames locally.
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
    project_targets: List[ProjectTarget] = field(default_factory=list)
    dryrun: bool = False
    motion_compensate: bool = False
    include_imu_data: bool = True
    write_debug_frames: bool = False
    generate_tsv_report: bool = False
    scene_creation_timeout_s: int = 3600
    scene_creation_poll_interval_s: int = 10
    pre_annotation_timeout_s: int = 120
    pre_annotation_poll_interval_s: int = 5

    @property
    def project_external_id(self) -> Optional[str]:
        """Get the first configured project external ID.

        Returns:
            Optional[str]: Project external ID, or ``None`` when unconfigured.
        """
        return self.project_targets[0].external_id if self.project_targets else None

    @property
    def batch(self) -> Optional[str]:
        """Get the first configured project batch.

        Returns:
            Optional[str]: Batch external ID, or ``None``.
        """
        return self.project_targets[0].batch if self.project_targets else None


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


def _sort_key(path: Path) -> Tuple[int, str]:
    """Build a stable sort key for timestamp-named sensor files.

    Args:
        path (Path): Sensor file path.

    Returns:
        Tuple[int, str]: Numeric-first filename sort key.
    """
    try:
        return (0, f"{int(path.stem):030d}")
    except ValueError:
        return (1, path.stem)


def _sensor_sort_key(sensor_name: str, preferred_order: List[str]) -> Tuple[int, str]:
    """Build a sort key that prioritizes preferred sensors.

    Args:
        sensor_name (str): Sensor channel name.
        preferred_order (List[str]): Channel names in preferred order.

    Returns:
        Tuple[int, str]: Preference rank and channel name.
    """
    if sensor_name in preferred_order:
        return (preferred_order.index(sensor_name), sensor_name)
    return (len(preferred_order), sensor_name)


def _validate_sensor_file_counts(
    sequence_path: Path,
    sensor_files: Dict[str, List[Path]],
    anchor_sensor: str,
    expected_count: int,
) -> None:
    """Require every sensor to contribute exactly one file per frame.

    Args:
        sequence_path (Path): Staging sequence directory.
        sensor_files (Dict[str, List[Path]]): Sorted files keyed by sensor channel.
        anchor_sensor (str): Channel used to define the frame count.
        expected_count (int): Number of frames in the anchor channel.

    Returns:
        None

    Raises:
        ValueError: If any channel has a different number of files.
    """
    mismatched = {
        name: len(files) for name, files in sensor_files.items() if len(files) != expected_count
    }
    if mismatched:
        raise ValueError(
            f"Sensor file counts disagree in {sequence_path}: anchor {anchor_sensor} has "
            f"{expected_count} files but {mismatched} differ. Frames are paired with files by "
            "position, so re-convert the sequence instead of uploading a partial staging directory."
        )


def _validate_sensor_timestamps(
    sequence_path: Path,
    sensor_files: Dict[str, List[Path]],
    anchor_files: List[Path],
) -> None:
    """Require each sensor's timestamps to track the anchor sensor's frames.

    Args:
        sequence_path (Path): Staging sequence directory.
        sensor_files (Dict[str, List[Path]]): Sorted files keyed by sensor channel.
        anchor_files (List[Path]): Anchor channel files in frame order.

    Returns:
        None

    Raises:
        ValueError: If a channel's file deviates from its frame by more than
            90% of the anchor frame interval, which indicates an off-by-N pairing.
    """
    try:
        anchor_ts = [int(path.stem) for path in anchor_files]
    except ValueError:
        return

    intervals = sorted(b - a for a, b in zip(anchor_ts, anchor_ts[1:]))
    if not intervals:
        return

    # 90% of a frame interval: observed sensor sync jitter (e.g. a camera vs.
    # the lidar anchor) can approach half the interval, so a tighter bound
    # flagged legitimate frames; an off-by-one pairing still lands a full
    # interval away and stays well outside this bound.
    tolerance_ns = intervals[len(intervals) // 2] * 0.9
    if tolerance_ns <= 0:
        raise ValueError(
            f"Anchor timestamps in {sequence_path} are not strictly increasing; "
            "frame ordering cannot be trusted."
        )

    for name, files in sensor_files.items():
        for frame_idx, (path, expected_ns) in enumerate(zip(files, anchor_ts)):
            try:
                actual_ns = int(path.stem)
            except ValueError:
                break
            if abs(actual_ns - expected_ns) > tolerance_ns:
                raise ValueError(
                    f"Sensor {name} in {sequence_path} is misaligned at frame {frame_idx}: "
                    f"file {path.name} is {abs(actual_ns - expected_ns)}ns from the anchor "
                    f"timestamp {expected_ns} (tolerance {tolerance_ns:.0f}ns). The staging "
                    "directory likely mixes files from different conversion runs."
                )


def _load_upload_config(config_dict: Dict) -> KognicUploadConfig:
    """Load uploader configuration from parsed YAML.

    Args:
        config_dict (Dict): Parsed configuration mapping.

    Returns:
        KognicUploadConfig: Normalized upload settings.
    """
    conversion_config = config_dict["conversion"]
    organization_id = conversion_config.get("organization_id") or conversion_config.get(
        "client_organization_id"
    )
    workspace_id = conversion_config.get("workspace_id") or conversion_config.get(
        "write_workspace_id"
    )

    return KognicUploadConfig(
        input_base=Path(conversion_config["input_base"]),
        organization_id=organization_id,
        workspace_id=workspace_id,
        project_targets=_parse_project_targets(conversion_config),
        dryrun=conversion_config.get("dryrun", False),
        motion_compensate=conversion_config.get("motion_compensate", False),
        include_imu_data=conversion_config.get("include_imu_data", True),
        write_debug_frames=conversion_config.get("write_debug_frames", False),
        generate_tsv_report=conversion_config.get("generate_tsv_report", False),
        scene_creation_timeout_s=conversion_config.get("scene_creation_timeout_s", 1800),
        scene_creation_poll_interval_s=conversion_config.get("scene_creation_poll_interval_s", 10),
        pre_annotation_timeout_s=conversion_config.get("pre_annotation_timeout_s", 120),
        pre_annotation_poll_interval_s=conversion_config.get("pre_annotation_poll_interval_s", 5),
    )


def find_sequence_paths(input_base: Path) -> List[Path]:
    """Find Kognic staging sequences below a base path.

    Args:
        input_base (Path): Staging sequence or parent directory.

    Returns:
        List[Path]: Sequence directories containing ``calibration.json``.

    Raises:
        FileNotFoundError: If no staging sequences are found.
    """
    if (input_base / "calibration.json").exists():
        return [input_base]

    sequence_paths = [
        path
        for path in sorted(input_base.iterdir())
        if path.is_dir() and (path / "calibration.json").exists()
    ]
    if not sequence_paths:
        raise FileNotFoundError(
            f"No Kognic staging sequences found under {input_base}. "
            "Expected calibration.json in input_base or in its child directories."
        )
    return sequence_paths


class KognicDatasetUploader:
    """Build and upload Kognic scenes from local staging sequences."""

    def __init__(self, config: KognicUploadConfig):
        """Initialize the uploader.

        Args:
            config (KognicUploadConfig): Upload settings.
        """
        self.config = config
        self._kognic_io_client: Optional[KognicIOClient] = None
        self._calibration_cache: Dict[str, str] = {}  # content hash → calibration_id

    @property
    def kognic_io_client(self) -> KognicIOClient:
        """Get a lazily initialized Kognic client.

        Returns:
            KognicIOClient: Client configured for the requested workspace.
        """
        if self._kognic_io_client is None:
            self._kognic_io_client = KognicIOClient(
                client_organization_id=self.config.organization_id,
                write_workspace_id=self.config.workspace_id,
            )
        return self._kognic_io_client

    def _get_or_upload_calibration(self, sequence_path: Path, external_id: str) -> str:
        """Reuse or upload a sequence calibration.

        Args:
            sequence_path (Path): Staging sequence directory.
            external_id (str): Scene external ID used in logs.

        Returns:
            str: Kognic calibration ID.
        """
        raw = (sequence_path / "calibration.json").read_bytes()
        content_hash = hashlib.sha256(raw).hexdigest()
        if content_hash in self._calibration_cache:
            calibration_id = self._calibration_cache[content_hash]
            logger.info(f"Reusing cached calibration {calibration_id} for {external_id}")
            return calibration_id
        calibration = self._load_calibration(sequence_path)
        cal_response = self.kognic_io_client.calibration.create_calibration(
            sensor_calibration=calibration
        )
        calibration_id = cal_response.id
        self._calibration_cache[content_hash] = calibration_id
        logger.info(f"Calibration uploaded for {external_id}: {calibration_id}")
        return calibration_id

    def upload_one(self, sequence_path: Path, external_id: str) -> List[SceneUploadResult]:
        """Upload a sequence as a single scene shared across all its projects.

        The sensor data is uploaded once as one scene. Every distinct
        pre-annotation requested by the projects is attached to that scene, then
        one input is created per project via ``client.input.create_from_scene``,
        which lets each input pick its own pre-annotation (or none) -- so e.g. a
        3D-cuboid project and a semseg project can share the same scene, the
        cuboid input referencing the cuboid pre-annotation and the semseg input
        referencing none.

        Returns a single ``SceneUploadResult`` (in a list). It is marked
        ``failed`` only if the scene ends up with no input at all (orphan, then
        invalidated); if some inputs succeed and others fail, the scene is kept
        and the failed projects are listed in ``failed_inputs``.

        Args:
            sequence_path (Path): Staging sequence directory.
            external_id (str): External ID for the uploaded scene.

        Returns:
            List[SceneUploadResult]: Upload outcomes for the sequence.
        """
        logger.info(f"Loading ego poses for {external_id}")
        ego_poses = self._load_ego_poses(sequence_path)

        # Validate and build the staging data before creating any remote resource.
        logger.info(f"Building frames and IMU data for {external_id}")
        frames = self._build_frames(sequence_path, ego_poses)
        imu_data = self._build_imu_data(sequence_path, ego_poses)
        annotated = sum(1 for f in frames if f.metadata.annotate)
        logger.info(
            f"{external_id}: {len(frames)} frames ({annotated} annotate=True), "
            f"{len(imu_data)} IMU samples"
        )

        if self.config.write_debug_frames:
            self._write_debug_frames(sequence_path, external_id, frames)

        start_time = time.time()
        logger.info(f"Uploading calibration for {external_id}")
        calibration_id = self._get_or_upload_calibration(sequence_path, external_id)
        logger.info(
            f"Time taken to upload calibration for {external_id}: "
            f"{time.time() - start_time} seconds"
        )

        feature_flags = FeatureFlags() if not self.config.motion_compensate else None

        # Load each distinct pre-annotation file referenced by the projects once.
        pre_annotations: Dict[str, OpenLabelAnnotation] = {}
        for target in self.config.project_targets:
            if target.pre_annotation and target.pre_annotation not in pre_annotations:
                pre_annotations[target.pre_annotation] = self._load_pre_annotation(
                    sequence_path / target.pre_annotation
                )

        scene = KognicModel.LidarsAndCamerasSequence(
            external_id=external_id,
            frames=frames,
            calibration_id=calibration_id,
            imu_data=imu_data,
            metadata=MetaData(
                source_filename=sequence_path.name,
                dataset_id=sequence_path.name,
                inner_uuid=str(uuid.uuid4()),
            ),
        )

        try:
            scene_uuid, inputs, failed_inputs, failed_input_errors = self._upload_scene(
                scene, external_id, pre_annotations, self.config.project_targets, feature_flags
            )
            return [
                SceneUploadResult(
                    external_id,
                    scene_uuid,
                    inputs=inputs,
                    failed_inputs=failed_inputs,
                    failed_input_errors=failed_input_errors,
                )
            ]
        except SceneInputError as exc:
            # The scene was created but a step failed and it ended up with no
            # input. _upload_scene tried to invalidate it; record whether that
            # worked so main() can report orphans needing manual cleanup.
            if exc.invalidated:
                logger.error(f"{exc}. Orphaned scene invalidated; re-upload to retry.")
            else:
                logger.error(
                    f"{exc}. Orphaned scene {exc.scene_uuid} could NOT be invalidated "
                    "and remains on Kognic; invalidate it manually."
                )
            return [
                SceneUploadResult(
                    external_id,
                    exc.scene_uuid,
                    failed=True,
                    invalidated=exc.invalidated,
                    error=exc,
                    failed_input_errors=exc.failed_input_errors,
                )
            ]

    def _upload_scene(
        self,
        scene: KognicModel.LidarsAndCamerasSequence,
        external_id: str,
        pre_annotations: Dict[str, OpenLabelAnnotation],
        targets: List[ProjectTarget],
        feature_flags: Optional[FeatureFlags],
    ) -> Tuple[
        SceneUUID,
        List[Dict[str, Optional[str]]],
        List[str],
        List[Tuple[ProjectTarget, BaseException]],
    ]:
        """Create the scene, attach pre-annotations, and create one input/project.

        Returns ``(scene_uuid, input_records, failed_inputs, failed_input_errors)``.
        Each input record is ``{"project_name", "batch_name", "input_id"}``;
        ``failed_inputs`` lists ``project/batch`` labels and
        ``failed_input_errors`` retains their original exceptions for reporting.
        On dryrun the scene_uuid is ``"dryrun"`` and all three lists are empty.

        Steps: create the scene (no project) and wait for Created; attach each
        distinct pre-annotation (capturing its uuid); then create one input per
        project, each referencing its project's pre-annotation uuid (or ``None``
        for no pre-annotation). A failure before any input exists invalidates the
        orphaned scene and raises; a failure once at least one input exists leaves
        the scene in place and is reported via ``failed_inputs``.

        Args:
            scene (KognicModel.LidarsAndCamerasSequence): Scene payload.
            external_id (str): Scene external ID.
            pre_annotations (Dict[str, OpenLabelAnnotation]): Annotations keyed
                by staged filename.
            targets (List[ProjectTarget]): Projects that receive scene inputs.
            feature_flags (Optional[FeatureFlags]): Kognic scene feature flags.

        Returns:
            Tuple: Scene UUID, successful input records, failed project/batch
                labels, and failed target/exception pairs.
        """
        logger.info(
            f"Uploading {external_id} as scene without input (dryrun={self.config.dryrun})"
        )
        response = self.kognic_io_client.lidars_and_cameras_sequence.create(
            scene,
            dryrun=self.config.dryrun,
            feature_flags=feature_flags,
        )
        if response is None:
            logger.info(
                f"{external_id}: dryrun OK; scene validated locally, "
                "pre-annotation upload and input creation skipped"
            )
            return "dryrun", [], [], []

        scene_uuid = response.scene_uuid

        # Until at least one input is attached the scene is an orphan (no input,
        # invisible to labelers), so invalidate it if scene processing or the
        # pre-annotation upload fails.
        stage = "scene processing"
        try:
            self._wait_for_scene_created(scene_uuid, external_id)

            stage = "pre-annotation upload"
            pre_annotation_uuids = self._upload_pre_annotations(
                scene_uuid, external_id, pre_annotations
            )
        except Exception as exc:
            logger.error(
                f"{external_id}: scene {scene_uuid} created but {stage} failed: {exc}. "
                "Invalidating the orphaned scene."
            )
            invalidated = self._invalidate_scene(scene_uuid, external_id)
            raise SceneInputError(
                external_id, scene_uuid, stage, exc, invalidated=invalidated
            ) from exc

        if not targets:
            logger.warning(
                f"{external_id}: no project configured; scene uploaded but no input "
                "created. Create one later with client.input.create_from_scene()."
            )
            return scene_uuid, [], [], []

        inputs, failed_inputs, failed_input_errors = self._create_inputs_from_scene(
            scene_uuid, external_id, targets, pre_annotation_uuids
        )

        # Every input failed: the scene is an orphan, so invalidate and raise.
        if not inputs:
            exc = RuntimeError(f"all {len(failed_inputs)} input(s) failed: {failed_inputs}")
            logger.error(
                f"{external_id}: scene {scene_uuid} created but no input could be "
                "created. Invalidating the orphaned scene."
            )
            invalidated = self._invalidate_scene(scene_uuid, external_id)
            raise SceneInputError(
                external_id,
                scene_uuid,
                "input creation",
                exc,
                invalidated=invalidated,
                failed_input_errors=failed_input_errors,
            ) from exc

        return scene_uuid, inputs, failed_inputs, failed_input_errors

    def _upload_pre_annotations(
        self,
        scene_uuid: SceneUUID,
        external_id: str,
        pre_annotations: Dict[str, OpenLabelAnnotation],
    ) -> Dict[str, str]:
        """Attach each distinct pre-annotation to a scene and wait for it to process.

        An input created against a pre-annotation that later fails or is still
        processing is silently dropped by Kognic, so each upload is confirmed
        ``indexed`` here, before any input references it, rather than trusting
        the create response alone.

        Args:
            scene_uuid (SceneUUID): Target scene UUID.
            external_id (str): Scene external ID used to build annotation IDs.
            pre_annotations (Dict[str, OpenLabelAnnotation]): Annotations keyed
                by staged filename.

        Returns:
            Dict[str, str]: Uploaded pre-annotation UUIDs keyed by filename.

        Raises:
            RuntimeError: If a pre-annotation fails or reaches an unrecognized status.
            TimeoutError: If a pre-annotation is still processing after the
                configured timeout.
        """
        pre_annotation_uuids: Dict[str, str] = {}
        for filename, pre_annotation in pre_annotations.items():
            logger.info(
                f"Uploading pre-annotation {filename} for {external_id} (scene {scene_uuid})"
            )
            created = self.kognic_io_client.pre_annotation.create(
                scene_uuid=scene_uuid,
                pre_annotation=pre_annotation,
                external_id=f"{external_id}-{Path(filename).stem}-pre-annotation",
                dryrun=False,
            )
            _wait_for_pre_annotation(
                self.kognic_io_client,
                created.id,
                timeout_s=self.config.pre_annotation_timeout_s,
                poll_s=self.config.pre_annotation_poll_interval_s,
                raise_on_timeout=True,
            )
            pre_annotation_uuids[filename] = created.id
        return pre_annotation_uuids

    def _invalidate_scene(self, scene_uuid: SceneUUID, external_id: str) -> bool:
        """Invalidate an orphaned scene; return True only if it actually worked.

        If invalidation fails (e.g. the scene is not yet queryable and the API
        404s) the scene is left behind on Kognic with no input, so log loudly and
        report it as ``False`` rather than pretending it was cleaned up.

        Args:
            scene_uuid (SceneUUID): Orphaned scene UUID.
            external_id (str): Scene external ID used in logs.

        Returns:
            bool: ``True`` only when invalidation succeeds.
        """
        try:
            self.kognic_io_client.scene.invalidate_scenes(
                scene_uuids=[scene_uuid],
                reason=SceneInvalidatedReason.INCORRECTLY_CREATED,
            )
            logger.info(f"{external_id}: invalidated orphaned scene {scene_uuid}")
            return True
        except Exception as cleanup_exc:
            logger.error(
                f"{external_id}: FAILED to invalidate orphaned scene {scene_uuid}: "
                f"{cleanup_exc}. The scene remains on Kognic with no input; invalidate "
                "it manually or re-run once it is queryable."
            )
            return False

    def _create_inputs_from_scene(
        self,
        scene_uuid: SceneUUID,
        external_id: str,
        projects: List[ProjectTarget],
        pre_annotation_uuids: Dict[str, str],
    ) -> Tuple[
        List[Dict[str, Optional[str]]],
        List[str],
        List[Tuple[ProjectTarget, BaseException]],
    ]:
        """Create one input per project from the shared scene.

        Each input references its project's pre-annotation (resolved from
        ``pre_annotation_uuids`` via the target's ``pre_annotation`` filename) or
        ``None`` for no pre-annotation -- so projects with incompatible task
        definitions (e.g. cuboid vs semseg) can share the same scene.

        Inputs are created independently: a failure on one project is recorded
        and the rest still proceed (the scene already exists and other inputs may
        be valid). Returns ``(input_records, failed, failed_errors)`` where each
        record is ``{"project_name", "batch_name", "input_id"}``, ``failed``
        lists the ``project/batch`` of inputs that could not be created, and
        ``failed_errors`` retains the target and original exception.

        Args:
            scene_uuid (SceneUUID): Source scene UUID.
            external_id (str): Scene external ID used in logs.
            projects (List[ProjectTarget]): Target projects and batches.
            pre_annotation_uuids (Dict[str, str]): Annotation UUIDs keyed by
                staged filename.

        Returns:
            Tuple: Successful input records, failed project/batch labels, and
                failed target/exception pairs.
        """
        records: List[Dict[str, Optional[str]]] = []
        failed: List[str] = []
        failed_errors: List[Tuple[ProjectTarget, BaseException]] = []
        for target in projects:
            pre_annotation_uuid = pre_annotation_uuids.get(target.pre_annotation)
            logger.info(
                f"{external_id}: creating input from scene {scene_uuid} in project "
                f"{target.external_id} (batch={target.batch}, "
                f"pre_annotation={target.pre_annotation or 'none'})"
            )
            try:
                created_input = self.kognic_io_client.input.create_from_scene(
                    scene_uuid=scene_uuid,
                    pre_annotation_uuid=pre_annotation_uuid,
                    project=target.external_id,
                    batch=target.batch,
                )
                records.append(
                    {
                        "project_name": target.external_id,
                        "batch_name": target.batch,
                        "input_id": str(created_input.uuid) if created_input else None,
                    }
                )
            except Exception as exc:
                logger.error(
                    f"{external_id}: failed to create input in project "
                    f"{target.external_id} (batch={target.batch}): {exc}"
                )
                failed.append(f"{target.external_id}/{target.batch}")
                failed_errors.append((target, exc))
        return records, failed, failed_errors

    def _load_pre_annotation(self, pre_annotation_path: Path) -> OpenLabelAnnotation:
        """Load and validate a configured pre-annotation OpenLabel file.

        Raises ``FileNotFoundError`` if the path is missing: a pre-annotation is
        only loaded when a project target explicitly requested it, so a missing
        file is a configuration error rather than an "absent, skip it" signal.

        Args:
            pre_annotation_path (Path): Staged OpenLABEL JSON path.

        Returns:
            OpenLabelAnnotation: Validated pre-annotation.

        Raises:
            FileNotFoundError: If the configured file is missing.
            ValidationError: If the JSON does not match the OpenLABEL model.
        """
        if not pre_annotation_path.exists():
            raise FileNotFoundError(
                f"configured pre_annotation file not found: {pre_annotation_path}"
            )

        with open(pre_annotation_path) as f:
            pre_annotation = OpenLabelAnnotation.model_validate(json.load(f))

        frames = pre_annotation.openlabel.frames or {}
        objects = pre_annotation.openlabel.objects or {}
        logger.info(
            f"Loaded {pre_annotation_path}: {len(objects)} objects over {len(frames)} frames"
        )
        return pre_annotation

    def _wait_for_scene_created(self, scene_uuid: str, external_id: str) -> None:
        """Poll until a scene finishes server-side processing.

        Args:
            scene_uuid (str): Scene UUID to query.
            external_id (str): Scene external ID used in logs.

        Returns:
            None

        Raises:
            RuntimeError: If scene processing fails.
            TimeoutError: If processing exceeds the configured timeout.
        """
        deadline = time.time() + self.config.scene_creation_timeout_s

        while True:
            try:
                scenes = self.kognic_io_client.scene.get_scenes_by_uuids(scene_uuids=[scene_uuid])
            except HTTPError as exc:
                # create() returns the scene_uuid before the scene becomes
                # queryable, so the query 404s for a short window right after
                # creation. The scene does exist, so treat 404 as "not yet
                # queryable" and keep polling instead of failing.
                if exc.response is not None and exc.response.status_code == 404:
                    scenes = []
                else:
                    raise
            status = scenes[0].status if scenes else None

            if status == SceneStatus.Created:
                logger.info(f"{external_id}: scene {scene_uuid} created")
                return
            if status is not None and (
                status == SceneStatus.Failed or str(status).startswith("invalidated")
            ):
                raise RuntimeError(
                    f"{external_id}: scene {scene_uuid} ended in status {status}: "
                    f"{scenes[0].error_message}"
                )
            if time.time() >= deadline:
                raise TimeoutError(
                    f"{external_id}: scene {scene_uuid} not created within "
                    f"{self.config.scene_creation_timeout_s}s (last status: {status})"
                )

            logger.info(
                f"{external_id}: scene {scene_uuid} status={status or 'not yet queryable'}; "
                f"waiting {self.config.scene_creation_poll_interval_s}s"
            )
            time.sleep(self.config.scene_creation_poll_interval_s)

    def _load_calibration(self, sequence_path: Path) -> KognicModel.SensorCalibration:
        """Load a staged calibration into the Kognic model.

        Args:
            sequence_path (Path): Staging sequence directory.

        Returns:
            KognicModel.SensorCalibration: Parsed sensor calibration.
        """
        with open(sequence_path / "calibration.json") as f:
            json_calibration = json.load(f)

        return KognicModel.SensorCalibration(
            external_id=str(uuid.uuid4()),
            calibration={
                sensor_name: KognicModel.SensorCalibrationEntry._parse_calibration(calib)
                for sensor_name, calib in json_calibration.items()
            },
        )

    def _load_ego_poses(
        self, sequence_path: Path
    ) -> Optional[Dict[str, KognicModel.EgoVehiclePose]]:
        """Load optional staged ego poses.

        Args:
            sequence_path (Path): Staging sequence directory.

        Returns:
            Optional[Dict[str, KognicModel.EgoVehiclePose]]: Poses keyed by
                frame ID, or ``None`` when the file is absent.
        """
        poses_file = sequence_path / "ego_poses.json"
        if not poses_file.exists():
            return None

        with open(poses_file) as f:
            poses_data = json.load(f)

        return {
            frame_id: KognicModel.EgoVehiclePose.from_json(pose_data)
            for frame_id, pose_data in poses_data.items()
        }

    def _collect_sensor_files(
        self, sequence_path: Path, root_name: str, suffix: str
    ) -> Dict[str, List[Path]]:
        """Collect staged files grouped by sensor channel.

        Args:
            sequence_path (Path): Staging sequence directory.
            root_name (str): Sensor root such as ``lidar`` or ``cameras``.
            suffix (str): File suffix to include.

        Returns:
            Dict[str, List[Path]]: Sorted files keyed by sensor channel.
        """
        root = sequence_path / root_name
        if not root.exists():
            return {}

        sensor_files = {}
        for sensor_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            files = sorted(sensor_dir.glob(f"*{suffix}"), key=_sort_key)
            if files:
                sensor_files[sensor_dir.name] = files
        return sensor_files

    def iterate_frames(
        self, sequence_path: Path
    ) -> Generator[Tuple[str, int, Dict[str, Path]], None, None]:
        """Yield synchronized staging frames in anchor-sensor order.

        Args:
            sequence_path (Path): Staging sequence directory.

        Yields:
            Tuple[str, int, Dict[str, Path]]: Frame ID, timestamp in
                nanoseconds, and sensor files keyed by channel.

        Raises:
            FileNotFoundError: If the sequence contains no supported sensor files.
            ValueError: If sensor file counts differ or a sensor's file
                timestamps do not line up with the anchor sensor's frames.
        """
        lidar_files = self._collect_sensor_files(sequence_path, "lidar", ".csv")
        camera_files = self._collect_sensor_files(sequence_path, "cameras", ".jpg")

        if not lidar_files and not camera_files:
            raise FileNotFoundError(f"No lidar CSVs or camera JPGs found in {sequence_path}")

        if lidar_files:
            anchor_sensors = sorted(
                lidar_files,
                key=lambda name: _sensor_sort_key(name, PREFERRED_LIDAR_SENSORS),
            )
            anchor_files = lidar_files[anchor_sensors[0]]
        else:
            anchor_sensors = sorted(
                camera_files,
                key=lambda name: _sensor_sort_key(name, PREFERRED_CAMERA_SENSORS),
            )
            anchor_files = camera_files[anchor_sensors[0]]

        all_files = {**lidar_files, **camera_files}
        _validate_sensor_file_counts(sequence_path, all_files, anchor_sensors[0], len(anchor_files))
        _validate_sensor_timestamps(sequence_path, all_files, anchor_files)

        for frame_idx, anchor_file in enumerate(anchor_files):
            timestamp_ns = int(anchor_file.stem)
            sensor_files: Dict[str, Path] = {
                name: files[frame_idx] for name, files in all_files.items()
            }

            yield str(frame_idx), timestamp_ns, sensor_files

    def _build_frames(
        self,
        sequence_path: Path,
        ego_poses: Optional[Dict[str, KognicModel.EgoVehiclePose]],
    ) -> List[LidarsAndCamerasSequenceFrame]:
        """Build Kognic frame models for a staging sequence.

        Args:
            sequence_path (Path): Staging sequence directory.
            ego_poses (Optional[Dict[str, KognicModel.EgoVehiclePose]]): Optional
                poses keyed by frame ID.

        Returns:
            List[LidarsAndCamerasSequenceFrame]: Ordered Kognic scene frames.
        """
        frames = []
        reference_timestamp = None

        frame_records = list(self.iterate_frames(sequence_path))
        keyframe_indices = self._load_keyframe_indices(sequence_path, len(frame_records))
        annotate_indices = set(keyframe_indices)

        for frame_idx, (frame_id, timestamp_ns, sensor_files) in enumerate(frame_records):
            if reference_timestamp is None:
                reference_timestamp = timestamp_ns

            annotate = frame_idx in annotate_indices
            relative_timestamp = int((timestamp_ns - reference_timestamp) / 1e6)
            point_clouds = [
                KognicModel.PointCloud(sensor_name=name, filename=str(path))
                for name, path in sensor_files.items()
                if path.suffix == ".csv"
            ]
            images = [
                KognicModel.Image(
                    sensor_name=name,
                    filename=str(path),
                    metadata=ImageMetadata(
                        shutter_time_start_ns=int(path.stem),
                        shutter_time_end_ns=int(path.stem),
                    ),
                )
                for name, path in sensor_files.items()
                if path.suffix == ".jpg"
            ]

            frames.append(
                LidarsAndCamerasSequenceFrame(
                    frame_id=frame_id,
                    relative_timestamp=relative_timestamp,
                    unix_timestamp=timestamp_ns,
                    ego_vehicle_pose=ego_poses.get(frame_id) if ego_poses else None,
                    point_clouds=point_clouds,
                    images=images,
                    metadata=FrameMetaData(annotate=annotate),
                )
            )

        return frames

    @staticmethod
    def _load_keyframe_indices(sequence_path: Path, frame_count: int) -> List[int]:
        """Load T4 keyframe positions exported by the T4-to-Kognic converters.

        ``keyframes.json`` holds the staging frame indices whose source T4
        ``sample_data`` records have ``is_key_frame`` set. When present, exactly
        those frames are marked ``annotate=True`` so the annotatable frames
        coincide with the T4 keyframes (and, for annotated scenes, with the
        pre-annotation frames). The file is required, and its recorded frame
        count must match the current staging data to prevent stale annotation
        flags from being applied to changed sensor files.

        Args:
            sequence_path (Path): Staging sequence directory.
            frame_count (int): Current number of staging frames.

        Returns:
            List[int]: Staging frame indices to mark for annotation.

        Raises:
            FileNotFoundError: If ``keyframes.json`` is missing.
            ValueError: If the file's frame count or keyframe indices are invalid.
        """
        keyframes_path = sequence_path / "keyframes.json"
        if not keyframes_path.exists():
            raise FileNotFoundError(
                f"Required keyframe metadata is missing: {keyframes_path}. "
                "Re-run the T4-to-Kognic converter before uploading."
            )

        with open(keyframes_path) as f:
            data = json.load(f)

        if data.get("frame_count") != frame_count:
            raise ValueError(
                f"{keyframes_path} was generated for {data.get('frame_count')} frames but "
                f"the current staging data has {frame_count}. Re-run the T4-to-Kognic "
                "converter before uploading."
            )

        keyframe_indices = data.get("keyframe_indices")
        if not isinstance(keyframe_indices, list) or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < frame_count
            for index in keyframe_indices
        ):
            raise ValueError(
                f"{keyframes_path} contains invalid keyframe_indices for {frame_count} frames. "
                "Re-run the T4-to-Kognic converter before uploading."
            )

        logger.info(
            f"Marking the {len(keyframe_indices)} T4 keyframes from "
            f"{keyframes_path.name} as annotate=True"
        )
        return keyframe_indices

    def _build_imu_data(
        self,
        sequence_path: Path,
        ego_poses: Optional[Dict[str, KognicModel.EgoVehiclePose]],
    ) -> List[IMUData]:
        """Interpolate ego poses into dense Kognic IMU data.

        Args:
            sequence_path (Path): Staging sequence directory.
            ego_poses (Optional[Dict[str, KognicModel.EgoVehiclePose]]): Sparse
                poses keyed by frame ID.

        Returns:
            List[IMUData]: Dense pose samples, or an empty list when disabled
                or insufficient data is available.
        """
        if not self.config.include_imu_data or not ego_poses:
            logger.info("Skipping building the IMU data...")
            return []

        try:
            from scipy.interpolate import interp1d
            from scipy.spatial.transform import Rotation, Slerp
        except ModuleNotFoundError:
            logger.warning("scipy is not installed; skipping optional IMU data generation")
            return []

        logger.info("Building IMU data for (this may take a while if there are many frames)")

        sparse: List[Tuple[int, KognicModel.EgoVehiclePose]] = []
        for frame_id, timestamp_ns, _ in self.iterate_frames(sequence_path):
            pose = ego_poses.get(frame_id)
            if pose is not None:
                sparse.append((timestamp_ns, pose))
        if len(sparse) < 2:
            return []

        sparse.sort(key=lambda x: x[0])
        ts_sparse = np.array([s[0] for s in sparse], dtype=np.float64)
        pos_sparse = np.array(
            [[s[1].position.x, s[1].position.y, s[1].position.z] for s in sparse],
            dtype=np.float64,
        )
        rot_sparse = Rotation.concatenate(
            [
                Rotation.from_quat(
                    [
                        s[1].rotation.x,
                        s[1].rotation.y,
                        s[1].rotation.z,
                        s[1].rotation.w,
                    ]
                )
                for s in sparse
            ]
        )

        dense_dt_ns = int(1e9 / IMU_TARGET_HZ)
        extrap_ns = IMU_EXTRAPOLATE_S * 1e9
        t0 = ts_sparse[0] - extrap_ns
        t1 = ts_sparse[-1] + extrap_ns
        ts_dense = np.arange(t0, t1 + dense_dt_ns, dense_dt_ns)

        pos_interp = interp1d(
            ts_sparse,
            pos_sparse,
            axis=0,
            fill_value="extrapolate",
            bounds_error=False,
        )
        pos_dense = pos_interp(ts_dense)

        slerp = Slerp(ts_sparse, rot_sparse)
        rot_dense_list = []
        for t in ts_dense:
            if t <= ts_sparse[0]:
                dt = ts_sparse[1] - ts_sparse[0]
                diff = rot_sparse[0].inv() * rot_sparse[1]
                scale = (ts_sparse[0] - t) / dt
                rot_dense_list.append(rot_sparse[0] * (diff.inv() ** scale))
            elif t >= ts_sparse[-1]:
                dt = ts_sparse[-1] - ts_sparse[-2]
                diff = rot_sparse[-2].inv() * rot_sparse[-1]
                scale = (t - ts_sparse[-1]) / dt
                rot_dense_list.append(rot_sparse[-1] * (diff**scale))
            else:
                rot_dense_list.append(slerp(t))

        quats = Rotation.concatenate(rot_dense_list).as_quat()
        return [
            IMUData(
                timestamp=float(ts_dense[i]),
                position=KognicModel.Position(
                    x=float(pos_dense[i, 0]),
                    y=float(pos_dense[i, 1]),
                    z=float(pos_dense[i, 2]),
                ),
                rotation_quaternion=KognicModel.RotationQuaternion(
                    w=float(quats[i, 3]),
                    x=float(quats[i, 0]),
                    y=float(quats[i, 1]),
                    z=float(quats[i, 2]),
                ),
            )
            for i in range(len(ts_dense))
        ]

    def _write_debug_frames(
        self,
        sequence_path: Path,
        external_id: str,
        frames: List[LidarsAndCamerasSequenceFrame],
    ) -> None:
        """Write serialized frame models for local debugging.

        Args:
            sequence_path (Path): Staging sequence directory.
            external_id (str): Scene external ID.
            frames (List[LidarsAndCamerasSequenceFrame]): Frames to serialize.

        Returns:
            None
        """
        debug_path = sequence_path / "frames_debug.json"
        with open(debug_path, "w") as f:
            json.dump(
                {
                    "external_id": external_id,
                    "frames": [frame.model_dump() for frame in frames],
                },
                f,
                indent=2,
                default=str,
            )


def _scene_report_path(input_base: Path, sequence_path: Path) -> str:
    """Return a complete nested scene path relative to the upload input base."""
    resolved_base = input_base.resolve()
    resolved_scene = sequence_path.resolve()
    try:
        relative_scene = resolved_scene.relative_to(resolved_base)
    except ValueError:
        return str(resolved_scene)
    if relative_scene == Path("."):
        return resolved_scene.name
    return relative_scene.as_posix()


def _exception_report_fields(exc: BaseException) -> Dict[str, str]:
    """Extract a stable type, message, and HTTP/SDK error code from an exception."""
    stage = "upload"
    if isinstance(exc, SceneInputError):
        stage = exc.stage

    current: Optional[BaseException] = exc
    visited: set = set()
    error_code = ""
    leaf: BaseException = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        leaf = current
        response = getattr(current, "response", None)
        candidate = getattr(response, "status_code", None)
        if candidate is None:
            candidate = getattr(current, "status_code", None)
        if candidate is None:
            candidate = getattr(current, "code", None)
        if candidate is not None and not callable(candidate):
            error_code = str(candidate)
            break
        current = getattr(current, "cause", None) or current.__cause__

    return {
        "stage": stage,
        "error_code": error_code,
        "error_type": type(leaf).__name__,
        "error_message": str(leaf),
    }


def _upload_report_row(
    *,
    scene: str,
    status: str,
    duration_seconds: float,
    stage: str = "",
    project: str = "",
    batch: Optional[str] = None,
    scene_uuid: Optional[str] = None,
    input_id: Optional[str] = None,
    invalidated: Optional[bool] = None,
    error: Optional[BaseException] = None,
) -> Dict[str, str]:
    """Build one normalized upload report row."""
    error_fields = _exception_report_fields(error) if error is not None else {}
    return {
        "scene": scene,
        "status": status,
        "stage": stage or error_fields.get("stage", ""),
        "project": project,
        "batch": batch or "",
        "scene_uuid": scene_uuid or "",
        "input_id": input_id or "",
        "invalidated": "" if invalidated is None else str(invalidated).lower(),
        "error_code": error_fields.get("error_code", ""),
        "error_type": error_fields.get("error_type", ""),
        "error_message": error_fields.get("error_message", ""),
        "duration_seconds": f"{duration_seconds:.3f}",
    }


def _result_report_rows(
    scene: str, result: SceneUploadResult, duration_seconds: float
) -> List[Dict[str, str]]:
    """Build the scene summary and per-input rows for an upload result."""
    if result.failed:
        status = "failed"
    elif result.failed_inputs:
        status = "partial_success"
    elif result.scene_uuid == "dryrun":
        status = "dryrun_successful"
    else:
        status = "successful"

    rows = [
        _upload_report_row(
            scene=scene,
            status=status,
            duration_seconds=duration_seconds,
            scene_uuid=result.scene_uuid,
            invalidated=result.invalidated if result.failed else None,
            error=result.error,
        )
    ]
    rows.extend(
        _upload_report_row(
            scene=scene,
            status="input_successful",
            stage="input creation",
            project=str(input_record.get("project_name") or ""),
            batch=input_record.get("batch_name"),
            scene_uuid=result.scene_uuid,
            input_id=input_record.get("input_id"),
            duration_seconds=duration_seconds,
        )
        for input_record in result.inputs
    )
    rows.extend(
        _upload_report_row(
            scene=scene,
            status="input_failed",
            stage="input creation",
            project=target.external_id,
            batch=target.batch,
            scene_uuid=result.scene_uuid,
            duration_seconds=duration_seconds,
            error=error,
        )
        for target, error in result.failed_input_errors
    )
    return rows


def _write_upload_report(input_base: Path, rows: List[Dict[str, str]]) -> Path:
    """Write upload report rows below the staging input base."""
    input_base.mkdir(parents=True, exist_ok=True)
    report_path = input_base / _UPLOAD_REPORT_FILENAME
    with open(report_path, "w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=_UPLOAD_REPORT_FIELDS, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Upload report saved to {report_path}")
    return report_path


def read_upload_report_scene_uuids(
    report_path: Path, statuses: set[str]
) -> List[str]:
    """Read unique, non-dry-run scene UUIDs for selected summary statuses.

    Args:
        report_path (Path): Uploader-generated ``upload_report.tsv``.
        statuses (set[str]): Scene-summary statuses to include.

    Returns:
        List[str]: Unique scene UUIDs in report order.

    Raises:
        FileNotFoundError: If the report does not exist.
        ValueError: If required report columns are missing.
    """
    if not report_path.exists():
        raise FileNotFoundError(f"upload report not found: {report_path}")

    scene_uuids: List[str] = []
    seen: set[str] = set()
    with open(report_path, newline="", encoding="utf-8") as report_file:
        reader = csv.DictReader(report_file, dialect="excel-tab")
        required_columns = {"status", "scene_uuid"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{report_path}: upload report is missing required column(s): "
                f"{', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            scene_uuid = (row.get("scene_uuid") or "").strip()
            if (
                row.get("status") in statuses
                and scene_uuid
                and scene_uuid != "dryrun"
                and scene_uuid not in seen
            ):
                seen.add(scene_uuid)
                scene_uuids.append(scene_uuid)
    return scene_uuids


def main():
    """Run the dataset-upload command-line interface.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/upload_kognic_dataset_sample.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    assert (
        config_dict["task"] == "upload_kognic_dataset"
    ), f"use config file of upload_kognic_dataset task: {config_dict['task']}"

    upload_config = _load_upload_config(config_dict)
    uploader = KognicDatasetUploader(upload_config)

    report_rows: List[Dict[str, str]] = []

    failures: List[str] = []
    pre_creation_failures: List[str] = []
    orphans: List[str] = []  # failed AND not invalidated -> need manual cleanup
    partial: List[str] = []  # scene OK but some project inputs failed
    try:
        sequence_paths = find_sequence_paths(upload_config.input_base)
        for sequence_path in sequence_paths:
            dataset_name = sequence_path.name
            report_scene = _scene_report_path(upload_config.input_base, sequence_path)
            time_start = time.time()
            logger.info(f"Uploading dataset {dataset_name} from {sequence_path}")
            try:
                results = uploader.upload_one(sequence_path, external_id=dataset_name)
            except Exception as exc:
                if not upload_config.generate_tsv_report:
                    raise
                duration = time.time() - time_start
                logger.exception(f"Failed to upload dataset {dataset_name}")
                failures.append(dataset_name)
                pre_creation_failures.append(dataset_name)
                report_rows.append(
                    _upload_report_row(
                        scene=report_scene,
                        status="failed",
                        duration_seconds=duration,
                        error=exc,
                    )
                )
                continue

            duration = time.time() - time_start
            for result in results:
                report_rows.extend(_result_report_rows(report_scene, result, duration))
                if result.failed:
                    # Scene was created but ended up with no input. If invalidation
                    # also failed the scene is left behind and needs manual cleanup.
                    failures.append(result.external_id)
                    if not result.invalidated:
                        orphans.append(f"{result.external_id}={result.scene_uuid}")
                    continue
                logger.info(
                    f"dataset_id ({result.external_id}): scene {result.scene_uuid}, "
                    f"{len(result.inputs)} input(s)"
                )
                if result.failed_inputs:
                    partial.append(
                        f"{result.external_id} (scene kept): {', '.join(result.failed_inputs)}"
                    )
            logger.info(f"Time taken to upload {dataset_name}: {duration} seconds")
    finally:
        if upload_config.generate_tsv_report:
            _write_upload_report(upload_config.input_base, report_rows)

    if failures or partial:
        parts: List[str] = []
        if failures:
            msg = f"{len(failures)} scene upload(s) failed: {', '.join(failures)}."
            if pre_creation_failures:
                msg += (
                    f" {len(pre_creation_failures)} failed before scene/input completion: "
                    f"{', '.join(pre_creation_failures)}."
                )
            msg += " Re-upload to retry."
            if orphans:
                msg += (
                    f" {len(orphans)} could NOT be invalidated and remain orphaned on "
                    f"Kognic (external_id=scene_uuid): {', '.join(orphans)}. "
                    "Invalidate them manually."
                )
            parts.append(msg)
        if partial:
            parts.append(
                f"{len(partial)} scene(s) created but some project inputs failed "
                f"(project/batch): {'; '.join(partial)}. Re-run to create the missing inputs."
            )
        raise SystemExit(" ".join(parts))


if __name__ == "__main__":
    main()
