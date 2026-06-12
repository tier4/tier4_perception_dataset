import argparse
from dataclasses import dataclass
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
from kognic.io.model.scene.lidars_and_cameras_sequence.frame import (
    Frame as LidarsAndCamerasSequenceFrame,
)
from kognic.io.model.scene.metadata.metadata import FrameMetaData, MetaData
from kognic.io.model.scene.resources.image import ImageMetadata
from kognic.io.model.scene.scene_entry import SceneStatus
from kognic.openlabel.models.models import OpenLabelAnnotation
import numpy as np
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


@dataclass(frozen=True)
class KognicUploadConfig:
    input_base: Path
    organization_id: str
    workspace_id: str
    project_external_id: Optional[str] = None
    batch: Optional[str] = None
    target_hz: Optional[int] = None
    dryrun: bool = False
    motion_compensate: bool = False
    include_imu_data: bool = True
    write_debug_frames: bool = False
    scene_creation_timeout_s: int = 1800
    scene_creation_poll_interval_s: int = 10


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
        project_external_id=conversion_config.get("project_external_id"),
        batch=conversion_config.get("batch"),
        target_hz=conversion_config.get("target_hz"),
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

    def upload_one(self, sequence_path: Path, external_id: str) -> SceneUUID:

        # create calibration first since the frames reference the calibration_id
        start_time = time.time()
        logger.info(f"Uploading calibration for {external_id}")
        calibration_id = self._get_or_upload_calibration(sequence_path, external_id)
        end_time = time.time()
        logger.info(
            f"Time taken to upload calibration for {external_id}: {end_time - start_time} seconds"
        )

        logger.info(f"Loading ego poses for {external_id}")
        ego_poses = self._load_ego_poses(sequence_path)

        logger.info(f"Building frames and IMU data for {external_id}")
        frames = self._build_frames(sequence_path, ego_poses)

        logger.info(
            f"Building IMU data for {external_id} (this may take a while if there are many frames)"
        )
        imu_data = self._build_imu_data(sequence_path, ego_poses)

        annotated = sum(1 for f in frames if f.metadata.annotate)
        logger.info(
            f"{external_id}: {len(frames)} frames ({annotated} annotate=True), "
            f"{len(imu_data)} IMU samples"
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

        if self.config.write_debug_frames:
            self._write_debug_frames(sequence_path, external_id, frames)

        feature_flags = FeatureFlags() if not self.config.motion_compensate else None

        pre_annotation = self._load_pre_annotation(sequence_path)
        if pre_annotation is None:
            # No pre-annotation, so we can create the scene and input in one step without waiting.
            logger.info(f"Uploading {external_id} to Kognic (dryrun={self.config.dryrun})")
            response = self.kognic_io_client.lidars_and_cameras_sequence.create(
                scene,
                project=self.config.project_external_id,
                batch=self.config.batch,
                dryrun=self.config.dryrun,
                feature_flags=feature_flags,
            )
            return "dryrun" if response is None else response.scene_uuid

        # Pre-annotation flow (https://docs.kognic.com/api-guide/pre-annotations):
        # The scene must reach Created status before the pre-annotation can be attached, so:
        # 1) create the scene to validate files and metadata and upload resources, and wait for the scene creation status turn to be created
        # 2) upload the pre-annotation with reference to the created scene (the pre-annotation will be attached to the scene since it has the same external_id),
        # 3) Then create the input from the scene (attach the scene to a project and batch) so that the pre-annotation is visible to labelers.
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
                f"{external_id}: dryrun OK; pre-annotation validated locally, "
                "pre-annotation upload and input creation skipped"
            )
            return "dryrun"

        scene_uuid = response.scene_uuid
        self._wait_for_scene_created(scene_uuid, external_id)

        logger.info(f"Uploading pre-annotation for {external_id} (scene {scene_uuid})")
        self.kognic_io_client.pre_annotation.create(
            scene_uuid=scene_uuid,
            pre_annotation=pre_annotation,
            external_id=f"{external_id}-pre-annotation",
            dryrun=False,
        )

        if self.config.project_external_id:
            logger.info(
                f"Creating input from scene {scene_uuid} in project "
                f"{self.config.project_external_id}"
            )
            self.kognic_io_client.lidars_and_cameras_sequence.create_from_scene(
                scene_uuid=scene_uuid,
                project=self.config.project_external_id,
                batch=self.config.batch,
            )
        else:
            logger.warning(
                f"{external_id}: no project_external_id configured; scene and "
                "pre-annotation uploaded but no input created. Create one later with "
                "client.lidars_and_cameras_sequence.create_from_scene()."
            )

        return scene_uuid

    def _load_pre_annotation(self, sequence_path: Path) -> Optional[OpenLabelAnnotation]:
        """Load and validate <sequence>/pre_annotation.json if present."""
        pre_annotation_path = sequence_path / "pre_annotation.json"
        if not pre_annotation_path.exists():
            return None

        with open(pre_annotation_path) as f:
            pre_annotation = OpenLabelAnnotation.model_validate(json.load(f))

        frames = pre_annotation.openlabel.frames or {}
        objects = pre_annotation.openlabel.objects or {}
        logger.info(
            f"Found {pre_annotation_path}: {len(objects)} objects over {len(frames)} frames"
        )
        return pre_annotation

    def _wait_for_scene_created(self, scene_uuid: str, external_id: str) -> None:
        """Poll until the scene finished server-side processing."""
        deadline = time.time() + self.config.scene_creation_timeout_s

        while True:
            scenes = self.kognic_io_client.scene.get_scenes_by_uuids(scene_uuids=[scene_uuid])
            status = scenes[0].status if scenes else None

            if status == SceneStatus.Created:
                logger.info(f"{external_id}: scene {scene_uuid} created")
                return
            if status in (SceneStatus.Failed,) or (
                status is not None and str(status).startswith("invalidated")
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
                f"{external_id}: scene {scene_uuid} status={status}; waiting "
                f"{self.config.scene_creation_poll_interval_s}s"
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

        for frame_id, timestamp_ns, sensor_files in self.iterate_frames(sequence_path):
            if reference_timestamp is None:
                reference_timestamp = timestamp_ns

            if (
                min_interval_ns == 0
                or last_annotated_ts is None
                or (timestamp_ns - last_annotated_ts) >= min_interval_ns
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
            return []

        try:
            from scipy.interpolate import interp1d
            from scipy.spatial.transform import Rotation, Slerp
        except ModuleNotFoundError:
            logger.warning("scipy is not installed; skipping optional IMU data generation")
            return []

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
    for sequence_path in sequence_paths:
        dataset_name = sequence_path.name
        time_start = time.time()
        logger.info(f"Uploading dataset {dataset_name} from {sequence_path}")
        scene_uuid = uploader.upload_one(sequence_path, external_id=dataset_name)
        logger.info(f"dataset_id: {scene_uuid}")
        time_end = time.time()
        logger.info(f"Time taken to upload {dataset_name}: {time_end - time_start} seconds")
        dataset_name_id_dict[dataset_name] = scene_uuid

        with open(osp.join(upload_config.input_base, "dataset_id.json"), "w") as f:
            json.dump(dataset_name_id_dict, f)


if __name__ == "__main__":
    main()
