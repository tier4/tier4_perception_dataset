import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os.path as osp
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
from requests import HTTPError
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
    ):
        super().__init__(f"{external_id}: scene {scene_uuid} created but {stage} failed: {cause}")
        self.external_id = external_id
        self.scene_uuid = scene_uuid
        self.stage = stage
        self.cause = cause
        # Whether the orphaned scene was successfully invalidated during cleanup.
        # If False the scene remains on Kognic and needs manual invalidation.
        self.invalidated = invalidated


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
    failed: bool = False
    # When ``failed``: whether the orphaned scene was successfully invalidated.
    invalidated: bool = False


@dataclass(frozen=True)
class KognicUploadConfig:
    input_base: Path
    organization_id: str
    workspace_id: str
    project_targets: List[ProjectTarget] = field(default_factory=list)
    target_hz: Optional[int] = None
    annotation_interval_tolerance_s: Optional[float] = None
    dryrun: bool = False
    motion_compensate: bool = False
    include_imu_data: bool = True
    write_debug_frames: bool = False
    scene_creation_timeout_s: int = 1800
    scene_creation_poll_interval_s: int = 10

    @property
    def project_external_id(self) -> Optional[str]:
        """First configured project, for callers that only handle one project."""
        return self.project_targets[0].external_id if self.project_targets else None

    @property
    def batch(self) -> Optional[str]:
        """Batch of the first project, for callers that only handle one project."""
        return self.project_targets[0].batch if self.project_targets else None


def _parse_project_targets(conversion_config: Dict) -> List[ProjectTarget]:
    """Resolve the projects (and their optional batches) a scene's inputs go to.

    Configured via a ``projects`` list. ``batch`` and ``pre_annotation`` are both
    optional per project: ``batch`` defaults to the latest open batch on the
    Kognic side, and ``pre_annotation`` defaults to ``None`` (no pre-annotation
    uploaded for that batch). Listing several projects shares one scene across
    those that request the same ``pre_annotation``::

        projects:
          - external_id: project_a
            batch: cuboid_batch
            pre_annotation: pre_annotation.json   # cuboids -> attached to its own scene
          - external_id: project_b
            batch: semseg_batch                    # no pre_annotation -> separate, plain scene
    """
    targets: List[ProjectTarget] = []
    seen: set = set()
    for entry in conversion_config.get("projects") or []:
        if isinstance(entry, str):
            external_id, batch, pre_annotation = entry, None, None
        else:
            external_id = entry.get("external_id") or entry.get("project_external_id")
            batch = entry.get("batch")
            pre_annotation = entry.get("pre_annotation")
        if not external_id or not str(external_id).strip():
            raise ValueError(f"conversion.projects entry is missing external_id: {entry!r}")
        target = ProjectTarget(
            external_id=str(external_id).strip(), batch=batch, pre_annotation=pre_annotation
        )
        combination = (target.external_id, target.batch)
        if combination in seen:
            raise ValueError(
                "conversion.projects has a duplicate project/batch combination: "
                f"external_id={target.external_id!r}, batch={target.batch!r}"
            )
        seen.add(combination)
        targets.append(target)
    return targets


def _sort_key(path: Path) -> Tuple[int, str]:
    try:
        return (0, f"{int(path.stem):030d}")
    except ValueError:
        return (1, path.stem)


def _sensor_sort_key(sensor_name: str, preferred_order: List[str]) -> Tuple[int, str]:
    if sensor_name in preferred_order:
        return (preferred_order.index(sensor_name), sensor_name)
    return (len(preferred_order), sensor_name)


def _load_upload_config(config_dict: Dict) -> KognicUploadConfig:
    conversion_config = config_dict["conversion"]
    organization_id = conversion_config.get("organization_id") or conversion_config.get(
        "client_organization_id"
    )
    workspace_id = conversion_config.get("workspace_id") or conversion_config.get(
        "write_workspace_id"
    )

    if not organization_id:
        raise ValueError("conversion.organization_id is required for Kognic upload")
    if not workspace_id:
        raise ValueError("conversion.workspace_id is required for Kognic upload")

    return KognicUploadConfig(
        input_base=Path(conversion_config["input_base"]),
        organization_id=organization_id,
        workspace_id=workspace_id,
        project_targets=_parse_project_targets(conversion_config),
        target_hz=conversion_config.get("target_hz"),
        annotation_interval_tolerance_s=conversion_config.get("annotation_interval_tolerance_s"),
        dryrun=conversion_config.get("dryrun", False),
        motion_compensate=conversion_config.get("motion_compensate", False),
        include_imu_data=conversion_config.get("include_imu_data", True),
        write_debug_frames=conversion_config.get("write_debug_frames", False),
        scene_creation_timeout_s=conversion_config.get("scene_creation_timeout_s", 1800),
        scene_creation_poll_interval_s=conversion_config.get("scene_creation_poll_interval_s", 10),
    )


def find_sequence_paths(input_base: Path) -> List[Path]:
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
    def __init__(self, config: KognicUploadConfig):
        self.config = config
        self._kognic_io_client: Optional[KognicIOClient] = None
        self._calibration_cache: Dict[str, str] = {}  # content hash → calibration_id

    @property
    def kognic_io_client(self) -> KognicIOClient:
        if self._kognic_io_client is None:
            self._kognic_io_client = KognicIOClient(
                client_organization_id=self.config.organization_id,
                write_workspace_id=self.config.workspace_id,
            )
        return self._kognic_io_client

    def _get_or_upload_calibration(self, sequence_path: Path, external_id: str) -> str:
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
        """Upload a sequence, creating one scene per distinct pre-annotation.

        Calibration, frames and IMU data are shared across every scene created
        from this sequence, so they are built once. Projects are then grouped by
        the pre-annotation they request (see ``_resolve_scene_groups``) and each
        group becomes its own scene with its own inputs. Returns one
        ``SceneUploadResult`` per scene; a group whose input creation failed is
        marked ``failed`` (its orphaned scene has already been invalidated).
        """
        # create calibration first since the frames reference the calibration_id
        start_time = time.time()
        logger.info(f"Uploading calibration for {external_id}")
        calibration_id = self._get_or_upload_calibration(sequence_path, external_id)
        logger.info(
            f"Time taken to upload calibration for {external_id}: "
            f"{time.time() - start_time} seconds"
        )

        logger.info(f"Loading ego poses for {external_id}")
        ego_poses = self._load_ego_poses(sequence_path)

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

        feature_flags = FeatureFlags() if not self.config.motion_compensate else None

        groups = self._resolve_scene_groups(sequence_path, external_id)
        if len(groups) > 1:
            logger.info(
                f"{external_id}: creating {len(groups)} scenes, one per distinct "
                f"pre-annotation: {[scene_external_id for scene_external_id, _, _ in groups]}"
            )

        results: List[SceneUploadResult] = []
        for scene_external_id, pre_annotation, targets in groups:
            scene = KognicModel.LidarsAndCamerasSequence(
                external_id=scene_external_id,
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
                scene_uuid, inputs = self._upload_scene_group(
                    scene, scene_external_id, pre_annotation, targets, feature_flags
                )
                results.append(SceneUploadResult(scene_external_id, scene_uuid, inputs=inputs))
            except SceneInputError as exc:
                # The scene was created but a later step failed. _upload_scene_group
                # tried to invalidate it; record whether that worked and continue
                # with the remaining groups for this sequence.
                if exc.invalidated:
                    logger.error(f"{exc}. Orphaned scene invalidated; re-upload to retry.")
                else:
                    logger.error(
                        f"{exc}. Orphaned scene {exc.scene_uuid} could NOT be invalidated "
                        "and remains on Kognic; invalidate it manually."
                    )
                results.append(
                    SceneUploadResult(
                        scene_external_id,
                        exc.scene_uuid,
                        failed=True,
                        invalidated=exc.invalidated,
                    )
                )
        return results

    def _resolve_scene_groups(
        self, sequence_path: Path, external_id: str
    ) -> List[Tuple[str, Optional[OpenLabelAnnotation], List[ProjectTarget]]]:
        """Group project targets into scenes by the pre-annotation they request.

        A pre-annotation is attached to the scene and applied to *all* inputs
        created from it, so a scene can only be shared by targets whose task
        definition matches that pre-annotation. Each distinct ``pre_annotation``
        value (including ``None`` -> no pre-annotation) therefore becomes its own
        scene. The ``None`` group keeps the sequence's external_id; each
        pre-annotated group is suffixed with the pre-annotation file stem to keep
        scene external_ids unique.

        Returns ``(scene_external_id, pre_annotation_or_None, targets)`` tuples.
        """
        targets = self.config.project_targets
        if not targets:
            # No projects configured: still create a bare scene (no input).
            return [(external_id, None, [])]

        grouped: Dict[Optional[str], List[ProjectTarget]] = {}
        for target in targets:  # dict preserves config order of first appearance
            grouped.setdefault(target.pre_annotation, []).append(target)

        groups: List[Tuple[str, Optional[OpenLabelAnnotation], List[ProjectTarget]]] = []
        for pre_annotation_file, group_targets in grouped.items():
            if pre_annotation_file is None:
                groups.append((external_id, None, group_targets))
            else:
                scene_external_id = f"{external_id}-{Path(pre_annotation_file).stem}"
                pre_annotation = self._load_pre_annotation(sequence_path / pre_annotation_file)
                groups.append((scene_external_id, pre_annotation, group_targets))
        return groups

    def _upload_scene_group(
        self,
        scene: KognicModel.LidarsAndCamerasSequence,
        external_id: str,
        pre_annotation: Optional[OpenLabelAnnotation],
        targets: List[ProjectTarget],
        feature_flags: Optional[FeatureFlags],
    ) -> Tuple[SceneUUID, List[Dict[str, Optional[str]]]]:
        """Create one scene and its input(s), attaching ``pre_annotation`` if given.

        Returns ``(scene_uuid, input_records)`` where each input record is
        ``{"project_name", "batch_name", "input_id"}``. On dryrun the scene_uuid
        is ``"dryrun"`` and the records are empty.

        Always scene-first: create the scene without a project, wait for it to
        reach Created, attach the pre-annotation if any, then create one input per
        project from the scene (``create_from_scene`` returns the created inputs,
        which is the only way to capture their ids).
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
            return "dryrun", []

        scene_uuid = response.scene_uuid

        # The scene now exists server-side. If any step below fails it would be
        # orphaned (no input, invisible to labelers), so invalidate it before
        # re-raising rather than leaving it behind.
        stage = "scene processing"
        try:
            self._wait_for_scene_created(scene_uuid, external_id)

            if pre_annotation is not None:
                stage = "pre-annotation upload"
                logger.info(f"Uploading pre-annotation for {external_id} (scene {scene_uuid})")
                self.kognic_io_client.pre_annotation.create(
                    scene_uuid=scene_uuid,
                    pre_annotation=pre_annotation,
                    external_id=f"{external_id}-pre-annotation",
                    dryrun=False,
                )

            stage = "input creation"
            if targets:
                inputs = self._create_inputs_from_scene(scene_uuid, external_id, targets)
            else:
                inputs = []
                logger.warning(
                    f"{external_id}: no project configured; scene "
                    "uploaded but no input created. Create one later with "
                    "client.lidars_and_cameras_sequence.create_from_scene()."
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

        return scene_uuid, inputs

    def _invalidate_scene(self, scene_uuid: SceneUUID, external_id: str) -> bool:
        """Invalidate an orphaned scene; return True only if it actually worked.

        If invalidation fails (e.g. the scene is not yet queryable and the API
        404s) the scene is left behind on Kognic with no input, so log loudly and
        report it as ``False`` rather than pretending it was cleaned up.
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
        self, scene_uuid: SceneUUID, external_id: str, projects: List[ProjectTarget]
    ) -> List[Dict[str, Optional[str]]]:
        """Create one input per project from an already-created scene.

        All inputs share the single scene (and its pre-annotation), but each is
        created in its own project/batch. A failure on any project propagates so
        ``_upload_scene_group`` invalidates the whole scene; the next re-upload
        recreates the scene and every input, keeping the projects consistent
        rather than leaving a partially-attached scene behind.

        Returns one ``{"project_name", "batch_name", "input_id"}`` record per
        created input (a project/batch may yield several inputs, one per
        annotation type).
        """
        records: List[Dict[str, Optional[str]]] = []
        for target in projects:
            logger.info(
                f"{external_id}: creating input from scene {scene_uuid} in project "
                f"{target.external_id} (batch={target.batch})"
            )
            created = self.kognic_io_client.lidars_and_cameras_sequence.create_from_scene(
                scene_uuid=scene_uuid,
                project=target.external_id,
                batch=target.batch,
            )
            for created_input in created or []:
                records.append(
                    {
                        "project_name": target.external_id,
                        "batch_name": target.batch,
                        "input_id": str(created_input.uuid),
                    }
                )
        return records

    def _load_pre_annotation(self, pre_annotation_path: Path) -> OpenLabelAnnotation:
        """Load and validate a configured pre-annotation OpenLabel file.

        Raises ``FileNotFoundError`` if the path is missing: a pre-annotation is
        only loaded when a project target explicitly requested it, so a missing
        file is a configuration error rather than an "absent, skip it" signal.
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
        """Poll until the scene finished server-side processing."""
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

        for frame_idx, anchor_file in enumerate(anchor_files):
            timestamp_ns = int(anchor_file.stem)
            sensor_files: Dict[str, Path] = {}

            for lidar_name, files in lidar_files.items():
                if frame_idx < len(files):
                    sensor_files[lidar_name] = files[frame_idx]

            for camera_name, files in camera_files.items():
                if frame_idx < len(files):
                    sensor_files[camera_name] = files[frame_idx]

            yield str(frame_idx), timestamp_ns, sensor_files

    def _build_frames(
        self,
        sequence_path: Path,
        ego_poses: Optional[Dict[str, KognicModel.EgoVehiclePose]],
    ) -> List[LidarsAndCamerasSequenceFrame]:
        frames = []
        reference_timestamp = None
        min_interval_ns = int(1e9 / self.config.target_hz) if self.config.target_hz else 0
        last_annotated_ts: Optional[int] = None

        frame_records = list(self.iterate_frames(sequence_path))

        # T4 source frames are not spaced at exactly 1/source_hz: a small
        # per-frame drift (e.g. ~0.09997s instead of 0.1s) accumulates, so the
        # frame at a nominal 1.0s boundary can actually land at ~0.99999s.
        # Allow a tolerance when deciding whether to annotate, taken from
        # annotation_interval_tolerance_s. If not set, use half of the median frame interval as tolerance.
        if min_interval_ns:
            if self.config.annotation_interval_tolerance_s is not None:
                tolerance_ns = int(self.config.annotation_interval_tolerance_s * 1e9)
            else:
                timestamps = [ts for _, ts, _ in frame_records]
                deltas = sorted(b - a for a, b in zip(timestamps, timestamps[1:]) if b > a)
                if deltas:
                    tolerance_ns = deltas[len(deltas) // 2] // 2

        for frame_id, timestamp_ns, sensor_files in frame_records:
            if reference_timestamp is None:
                reference_timestamp = timestamp_ns

            if (
                min_interval_ns == 0
                or last_annotated_ts is None
                or (timestamp_ns - last_annotated_ts) >= (min_interval_ns - tolerance_ns)
            ):
                annotate = True
                last_annotated_ts = timestamp_ns
            else:
                annotate = False

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

    def _build_imu_data(
        self,
        sequence_path: Path,
        ego_poses: Optional[Dict[str, KognicModel.EgoVehiclePose]],
    ) -> List[IMUData]:
        if not self.config.include_imu_data or not ego_poses:
            logger.info(f"Skipping building the IMU data...")
            return []

        try:
            from scipy.interpolate import interp1d
            from scipy.spatial.transform import Rotation, Slerp
        except ModuleNotFoundError:
            logger.warning("scipy is not installed; skipping optional IMU data generation")
            return []

        logger.info(f"Building IMU data for (this may take a while if there are many frames)")

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


def main():
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
        config_dict["task"] == "upload_dataset"
    ), f"use config file of upload_dataset task: {config_dict['task']}"

    upload_config = _load_upload_config(config_dict)
    uploader = KognicDatasetUploader(upload_config)
    sequence_paths = find_sequence_paths(upload_config.input_base)

    dataset_name_id_dict = {}

    def _persist_dataset_ids() -> None:
        with open(osp.join(upload_config.input_base, "dataset_id.json"), "w") as f:
            json.dump(dataset_name_id_dict, f)

    failures: List[str] = []
    orphans: List[str] = []  # failed AND not invalidated -> need manual cleanup
    for sequence_path in sequence_paths:
        dataset_name = sequence_path.name
        time_start = time.time()
        logger.info(f"Uploading dataset {dataset_name} from {sequence_path}")
        # One sequence can produce several scenes (one per distinct pre-annotation).
        results = uploader.upload_one(sequence_path, external_id=dataset_name)
        for result in results:
            if result.failed:
                # Scene was created but a later step failed. If invalidation also
                # failed the scene is left behind and needs manual cleanup.
                failures.append(result.external_id)
                if not result.invalidated:
                    orphans.append(f"{result.external_id}={result.scene_uuid}")
                continue
            logger.info(
                f"dataset_id ({result.external_id}): scene {result.scene_uuid}, "
                f"{len(result.inputs)} input(s)"
            )
            dataset_name_id_dict[result.external_id] = {
                "scene_id": result.scene_uuid,
                "inputs": result.inputs,
            }
        _persist_dataset_ids()
        time_end = time.time()
        logger.info(f"Time taken to upload {dataset_name}: {time_end - time_start} seconds")

    if failures:
        message = (
            f"{len(failures)} scene(s) created without an input: "
            f"{', '.join(failures)}. Re-upload to retry."
        )
        if orphans:
            message += (
                f" {len(orphans)} could NOT be invalidated and remain orphaned on "
                f"Kognic (external_id=scene_uuid): {', '.join(orphans)}. "
                "Invalidate them manually."
            )
        raise SystemExit(message)


if __name__ == "__main__":
    main()
