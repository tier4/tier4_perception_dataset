import glob
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from pyquaternion import Quaternion
from segments import SegmentsClient
from segments.exceptions import NotFoundError, SegmentsError
from segments.typing import CameraConvention, InputType, PCDType, TaskType
from t4_devkit import Tier4

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import SENSOR_ENUM
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.transform import transform_matrix

logger = configure_logger(modname=__name__)

API_KEY = "3a66126fb3320bec2362cce84cc8c3940ff871fb"  # jacob's API KEY

DEFAULT_CAMERA_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests",
    "config",
    "convert_non_annotated_t4_to_deepen.yaml",
)


class NonAnnotatedT4ToSegmentsAIConverter(AbstractConverter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        camera_sensors: Optional[List[Dict[str, str]]] = None,
        annotation_hz: int = 10,
        dataset_name: str = "test_tier4_dataset",
        dataset_description: str = "",
        dataset_owner: str = "tier4",
        dataset_task_type: TaskType = TaskType.MULTISENSOR_SEQUENCE,
        dataset_task_attributes_path: Optional[str] = None,
        create_dataset: bool = False,
        sample_prefix: str = None,
        pcd_type: PCDType = PCDType.NUSCENES,
        lidar_sensor_name: str = "Lidar",
    ):
        super().__init__(input_base, output_base)

        self._annotation_hz = annotation_hz
        self._camera_sensor_types: List[SENSOR_ENUM] = []
        self._camera_channel_positions: Dict[str, int] = {}
        camera_sensors = self._resolve_camera_sensors(camera_sensors)
        self._camera_sensors: List[Dict[str, str]] = camera_sensors
        for index, cam in enumerate(camera_sensors):
            channel = cam.get("channel")
            if channel is None:
                continue
            if not SENSOR_ENUM.has_channel(channel):
                logger.warning(f"Unknown camera channel {channel}. Skipping.")
                continue
            sensor_enum = SENSOR_ENUM[channel]
            self._camera_sensor_types.append(sensor_enum)
            self._camera_channel_positions[channel] = index

        self._client = SegmentsClient(API_KEY)
        self._dataset_owner = dataset_owner
        self._dataset_name = dataset_name
        self._dataset_description = dataset_description
        self._dataset_task_type = TaskType(dataset_task_type)
        self._task_attributes = self._load_task_attributes(dataset_task_attributes_path)
        self._create_dataset_flag = create_dataset
        self._sample_prefix = sample_prefix
        self._pcd_type = PCDType(pcd_type)
        self._lidar_sensor_name = lidar_sensor_name
        self._dataset_identifier = f"{self._dataset_owner}/{self._dataset_name}"
        self._asset_cache: Dict[str, str] = {}

    def _resolve_camera_sensors(
        self, camera_sensors: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        if camera_sensors:
            return camera_sensors

        if not os.path.exists(DEFAULT_CAMERA_CONFIG):
            logger.warning(
                f"No camera sensors provided and default config not found at {DEFAULT_CAMERA_CONFIG}."
            )
            return []

        try:
            with open(DEFAULT_CAMERA_CONFIG, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.error(
                f"Failed to load default camera sensors from {DEFAULT_CAMERA_CONFIG}: {exc}"
            )
            return []

        sensors = config.get("conversion", {}).get("camera_sensors", []) or []
        if sensors:
            logger.info(
                f"Loaded {len(sensors)} camera sensors from {os.path.basename(DEFAULT_CAMERA_CONFIG)}."
            )
        else:
            logger.warning(
                f"Default config {DEFAULT_CAMERA_CONFIG} did not define camera sensors."
            )
        return sensors

    def convert(self):
        start_time = time.time()

        if self._create_dataset_flag:
            self._ensure_dataset_exists()
        else:
            self._verify_dataset_access()

        for scene_dir in glob.glob(os.path.join(self._input_base, "*")):
            if not os.path.isdir(scene_dir):
                continue

            out_dir = os.path.join(self._output_base, os.path.basename(scene_dir).replace(".", "-"))
            self._convert_one_scene(scene_dir, out_dir)

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.1f} [sec]")

    def _convert_one_scene(self, scene_dir: str, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        scene_name = os.path.basename(scene_dir)
        logger.info(f"Converting scene {scene_name}")

        t4_dataset = Tier4(data_root=scene_dir, verbose=False)
        frame_stride = max(1, int(10 / self._annotation_hz)) if self._annotation_hz else 1

        pointcloud_frames: List[Dict[str, Any]] = []
        camera_frames: Dict[str, List[Dict[str, Any]]] = {
            sensor.value["channel"]: [] for sensor in self._camera_sensor_types
        }

        for frame_index, sample in enumerate(t4_dataset.sample):
            if frame_index % frame_stride != 0:
                continue

            camera_only_mode = SENSOR_ENUM.LIDAR_CONCAT.value["channel"] not in sample.data
            lidar_frame_entry: Optional[Dict[str, Any]] = None
            lidar_metadata: Optional[Dict[str, Any]] = None

            if not camera_only_mode:
                lidar_token: str = sample.data[SENSOR_ENUM.LIDAR_CONCAT.value["channel"]]
                lidar_path, _, _ = t4_dataset.get_sample_data(lidar_token)
                lidar_metadata = self._get_sensor_metadata(t4_dataset, lidar_token)
                lidar_asset_fp = f"{scene_name}/{frame_index:06d}_{os.path.basename(lidar_path)}"
                if self._sample_prefix:
                    lidar_asset_fp = f"{self._sample_prefix}/{lidar_asset_fp}"
                lidar_asset_url = self._upload_asset(
                    lidar_path,
                    filename=lidar_asset_fp,
                )

                lidar_frame_entry = {
                    "pcd": {
                        "url": lidar_asset_url,
                        "type": self._pcd_type.value,
                    },
                    "images": [],
                    "name": lidar_token,
                    "timestamp": lidar_metadata["timestamp"],
                    "ego_pose": self._build_ego_pose(lidar_metadata),
                }
                lidar_frame_entry["default_z"] = lidar_frame_entry["ego_pose"]["position"]["z"]

            for sensor_enum in self._camera_sensor_types:
                camera_channel = sensor_enum.value["channel"]
                camera_token = self._get_camera_token(camera_channel, sample, t4_dataset)
                if camera_token is None:
                    logger.warning(
                        f"Camera token not found for channel {camera_channel} in frame {frame_index}. Skipping."
                    )
                    continue

                camera_path, _, cam_intrinsic = t4_dataset.get_sample_data(camera_token)
                camera_metadata = self._get_sensor_metadata(t4_dataset, camera_token)
                camera_asset_fp = (
                    f"{scene_name}/{camera_channel}/{frame_index:06d}_{os.path.basename(camera_path)}"
                )
                if self._sample_prefix:
                    camera_asset_fp = f"{self._sample_prefix}/{camera_asset_fp}"
                image_asset_url = self._upload_asset(
                    camera_path,
                    filename=camera_asset_fp,
                )

                camera_frame_entry = {
                    "image": {"url": image_asset_url},
                    "name": f"{camera_channel}_{frame_index:06d}",
                    "timestamp": camera_metadata["timestamp"],
                }
                camera_frames[camera_channel].append(camera_frame_entry)

                if lidar_frame_entry is not None and lidar_metadata is not None:
                    calibrated_image: Dict[str, Any] = {
                        "url": image_asset_url,
                        "name": camera_channel,
                        "row": 0,
                        "col": self._camera_channel_positions.get(camera_channel, 0),
                    }
                    if cam_intrinsic is not None:
                        calibrated_image["intrinsics"] = {
                            "intrinsic_matrix": cam_intrinsic.tolist()
                        }
                    extrinsics = self._build_camera_extrinsics(camera_metadata)
                    if extrinsics is not None:
                        calibrated_image["extrinsics"] = extrinsics
                    calibrated_image["camera_convention"] = CameraConvention.OPEN_CV.value
                    lidar_frame_entry["images"].append(calibrated_image)

            if lidar_frame_entry is not None:
                pointcloud_frames.append(lidar_frame_entry)

        sensors: List[Dict[str, Any]] = []
        if pointcloud_frames:
            sensors.append(
                {
                    "name": self._lidar_sensor_name,
                    "task_type": TaskType.POINTCLOUD_CUBOID_SEQUENCE.value,
                    "attributes": {
                        "frames": pointcloud_frames,
                    },
                }
            )

        for channel, frames in camera_frames.items():
            if not frames:
                continue
            sensors.append(
                {
                    "name": channel,
                    "task_type": TaskType.IMAGE_VECTOR_SEQUENCE.value,
                    "attributes": {
                        "frames": frames,
                    },
                }
            )

        if not sensors:
            logger.warning(f"No sensors collected for scene {scene_name}. Skipping sample creation.")
            return

        sample_attributes = {"sensors": sensors}
        sample_name = scene_name.replace(".", "-")

        sample_attributes_path = os.path.join(out_dir, f"{sample_name}_attributes.json")
        with open(sample_attributes_path, "w", encoding="utf-8") as f:
            json.dump(sample_attributes, f, indent=2)

        metadata = {
            "source_scene_dir": scene_dir,
            "frame_stride": frame_stride,
            "camera_channels": [channel for channel, frames in camera_frames.items() if frames],
            "pointcloud_frame_count": len(pointcloud_frames),
        }

        try:
            self._client.add_sample(
                self._dataset_identifier,
                sample_name,
                sample_attributes,
                metadata=metadata,
            )
            logger.info(
                f"Added sample {sample_name} with {len(pointcloud_frames)} pointcloud frames and "
                f"{sum(len(frames) for frames in camera_frames.values())} camera streams."
            )
        except NotFoundError:
            logger.error(
                f"Dataset {self._dataset_identifier} was not found while adding sample {sample_name}. "
                "Use create_dataset=True if you want to create it automatically."
            )
            raise
        except SegmentsError as exc:
            logger.error(f"Failed to add sample {sample_name}: {exc}")
            raise

    def _upload_asset(self, file_path: str, filename: str):
        if file_path in self._asset_cache:
            return self._asset_cache[file_path]

        with open(file_path, "rb") as f:
            asset = self._client.upload_asset(f, filename=filename)
        self._asset_cache[file_path] = asset.url
        return asset.url

    def _create_dataset(
        self,
        name: str,
        description: str,
        task_type: TaskType,
        task_attributes: dict,
    ):
        dataset = self._client.add_dataset(
            name=name,
            description=description,
            task_type=task_type,
            task_attributes=task_attributes,
            organization=self._dataset_owner,
        )
        return dataset

    def _ensure_dataset_exists(self):
        try:
            self._client.get_dataset(self._dataset_identifier)
            logger.info(f"Dataset {self._dataset_identifier} already exists.")
        except NotFoundError:
            logger.info(f"Dataset {self._dataset_identifier} not found. Creating new dataset.")
            self._create_dataset(
                name=self._dataset_name,
                description=self._dataset_description,
                task_type=self._dataset_task_type,
                task_attributes=self._task_attributes,
            )
        except SegmentsError as exc:
            logger.error(f"Error while ensuring dataset existence: {exc}")
            raise
        logger.info(f"Ready to add samples to {self._dataset_identifier}.")

    def _verify_dataset_access(self):
        try:
            self._client.get_dataset(self._dataset_identifier)
            logger.info(f"Using existing dataset {self._dataset_identifier}.")
        except SegmentsError as exc:
            logger.error(
                f"Unable to access dataset {self._dataset_identifier} ({exc}). "
                "Set create_dataset=True if it does not exist."
            )
            raise

    def _load_task_attributes(self, path: Optional[str]) -> Dict[str, Any]:
        if path is None:
            default_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "segmentsai_dataset_config.json"
            )
            path = default_path

        with open(path, "r", encoding="utf-8") as f:
            task_attributes = json.load(f)

        self._normalize_task_attributes(task_attributes)
        return task_attributes

    def _normalize_task_attributes(self, node: Any) -> None:
        if isinstance(node, dict):
            input_type = node.get("input_type")
            if input_type is not None:
                node["input_type"] = self._normalize_input_type(input_type)
            for value in node.values():
                self._normalize_task_attributes(value)
        elif isinstance(node, list):
            for item in node:
                self._normalize_task_attributes(item)

    def _normalize_input_type(self, input_type: Any) -> str:
        if isinstance(input_type, InputType):
            return input_type.value
        if not isinstance(input_type, str):
            return str(input_type)

        normalized = input_type.strip().lower()
        mapping = {
            "multi_select": InputType.MULTISELECT.value,
            "multi-select": InputType.MULTISELECT.value,
            "multiselect": InputType.MULTISELECT.value,
            "radio": InputType.SELECT.value,
            "select": InputType.SELECT.value,
            "checkbox": InputType.CHECKBOX.value,
            "text": InputType.TEXT.value,
            "number": InputType.NUMBER.value,
            "vector3": InputType.VECTOR3.value,
            "quaternion": InputType.QUATERNION.value,
            "points": InputType.POINTS.value,
        }

        return mapping.get(normalized, normalized)

    def _build_ego_pose(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        translation = metadata["sensor2global_translation"]
        rotation: Quaternion = metadata["sensor2global_quaternion"]
        return {
            "position": {
                "x": float(translation[0]),
                "y": float(translation[1]),
                "z": float(translation[2]),
            },
            "heading": {
                "qx": float(rotation.x),
                "qy": float(rotation.y),
                "qz": float(rotation.z),
                "qw": float(rotation.w),
            },
        }

    def _build_camera_extrinsics(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        translation = metadata.get("sensor2ego_translation")
        quaternion: Optional[Quaternion] = metadata.get("sensor2ego_quaternion")
        if translation is None or quaternion is None:
            return None

        translation = np.asarray(translation, dtype=float)

        return {
            "translation": {
                "x": float(translation[0]),
                "y": float(translation[1]),
                "z": float(translation[2]),
            },
            "rotation": {
                "qx": float(quaternion.x),
                "qy": float(quaternion.y),
                "qz": float(quaternion.z),
                "qw": float(quaternion.w),
            },
        }

    @staticmethod
    def _get_camera_token(camera_channel: str, sample, t4_dataset) -> Optional[str]:
        camera_token: Optional[str] = None
        if camera_channel in sample.data.keys():
            camera_token = sample.data[camera_channel]
        else:
            sample_data = [s for s in t4_dataset.sample_data if s.sample_token == sample.token]
            for sensor in sample_data:
                if sensor.channel == camera_channel:
                    camera_token = sensor.token
                    break
        return camera_token

    def _get_sensor_metadata(self, t4_dataset: Tier4, sensor_channel_token: str) -> Dict[str, Any]:
        sd_record = t4_dataset.get("sample_data", sensor_channel_token)
        cs_record = t4_dataset.get("calibrated_sensor", sd_record.calibrated_sensor_token)
        ep_record = t4_dataset.get("ego_pose", sd_record.ego_pose_token)

        sensor2ego_transform = transform_matrix(
            translation=cs_record.translation,
            rotation=cs_record.rotation,
        )
        sensor2ego_translation = sensor2ego_transform[:3, 3]
        sensor2ego_quaternion = Quaternion(matrix=sensor2ego_transform[:3, :3])

        ego2global_transform = transform_matrix(
            translation=ep_record.translation,
            rotation=ep_record.rotation,
        )

        sensor2global_transform = ego2global_transform @ sensor2ego_transform
        sensor2global_translation = sensor2global_transform[:3, 3]
        sensor2global_quaternion = Quaternion(matrix=sensor2global_transform[:3, :3])

        metadata = {
            "fileformat": sd_record.fileformat,
            "unix_timestamp": self._timestamp_to_sec(sd_record.timestamp),
            "timestamp": int(sd_record.timestamp),
            "sensor2global_transform": sensor2global_transform,
            "sensor2global_translation": sensor2global_translation,
            "sensor2global_quaternion": sensor2global_quaternion,
            "sensor2ego_translation": sensor2ego_translation,
            "sensor2ego_quaternion": sensor2ego_quaternion,
        }

        return metadata

    @staticmethod
    def _timestamp_to_sec(timestamp: int) -> float:
        return float(timestamp) * 1e-6


if __name__ == "__main__":
    converter = NonAnnotatedT4ToSegmentsAIConverter(
        input_base="test_data/unannotated/",
        output_base="test_data/segmentsai/",
        camera_sensors=None,
        annotation_hz=1,
        dataset_name="test_tier4_dataset",
        dataset_description="Test dataset converted from unannotated Tier4 data",
        create_dataset=True,
    )
    converter.convert()
