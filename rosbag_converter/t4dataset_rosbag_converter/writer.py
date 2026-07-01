from __future__ import annotations

import json
import os
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from autoware_sensing_msgs.msg import ConcatenatedPointCloudInfo
from sensor_msgs.msg import CompressedImage
from t4_devkit.schema.tables import (
    Attribute,
    CalibratedSensor,
    Category,
    EgoPose,
    Instance,
    Log,
    Map,
    Sample,
    SampleAnnotation,
    SampleData,
    Scene,
    Sensor,
    Visibility,
)

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME
from perception_dataset.t4_dataset.table_handler import TableHandler
from perception_dataset.utils import misc as misc_utils

from .calibration import CalibrationSet
from .camera_calibration import CameraCalibration
from .concat import ConcatenatedFrame
from .geometry import RigidTransform
from .geometry import identity
from .pcd import save_lidar_pointcloud_pcd
from .pointcloud import pointcloud_to_lidar_features
from .pointcloud import stamp_to_seconds
from .tf_manager import TfManager


@dataclass(frozen=True)
class CameraWriteRequest:
    topic: str
    image_index: int
    filename: str
    calibration: CameraCalibration
    sample_data_token: str


@dataclass
class CameraSyncState:
    timestamps: list[float]
    delay_sec: float
    window_sec: float
    current_index: int = 0

    def match(self, lidar_timestamp: float) -> int | None:
        while self.current_index < len(self.timestamps):
            adjusted = self.timestamps[self.current_index] - self.delay_sec
            if adjusted - lidar_timestamp >= -self.window_sec:
                break
            self.current_index += 1
        if self.current_index >= len(self.timestamps):
            return None
        adjusted = self.timestamps[self.current_index] - self.delay_sec
        if adjusted - lidar_timestamp >= self.window_sec:
            return None
        image_index = self.current_index
        self.current_index += 1
        return image_index


class T4DatasetWriter:
    def __init__(
        self,
        *,
        output_base: Path,
        scene_name: str,
        conversion: dict[str, Any],
        scene_description: str,
        calibration: CalibrationSet,
        tf_manager: TfManager,
        camera_calibrations: dict[str, CameraCalibration],
        camera_timestamps: dict[str, list[float]],
    ) -> None:
        self.output_base = output_base
        self.scene_name = scene_name
        self.conversion = conversion
        self.scene_description = scene_description
        self.calibration = calibration
        self.tf_manager = tf_manager
        self.camera_calibrations = camera_calibrations
        self.camera_timestamps = camera_timestamps
        self.output_scene_dir = self._scene_dir()
        self.output_anno_dir = self.output_scene_dir / T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value
        self.output_data_dir = self.output_scene_dir / T4_FORMAT_DIRECTORY_NAME.DATA.value
        self.tables: list[TableHandler] = []
        self._sensor_tokens: dict[str, str] = {}
        self._calibrated_tokens: dict[str, str] = {}
        self._sample_data_tokens_by_channel: dict[str, list[str]] = defaultdict(list)
        self._camera_sync: dict[str, CameraSyncState] = {}
        self._camera_buffers: dict[str, deque[tuple[int, float, CompressedImage]]] = defaultdict(deque)
        self._pending_camera_requests: dict[tuple[str, int], CameraWriteRequest] = {}
        self._rectify_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._scene_token = ""
        self._lidar_frame_index = 0
        self._max_points = 0
        self._ego_pose_fallbacks = 0
        self._started = False

    @property
    def frame_count(self) -> int:
        return self._lidar_frame_index

    def start(self) -> None:
        self._init_tables()
        self._make_directories()
        self._scene_token = self._write_static_tables()
        self._camera_sync = self._build_camera_sync()
        self._started = True

    def write_lidar_frame(self, frame: ConcatenatedFrame) -> bool:
        if not self._started:
            raise RuntimeError("T4DatasetWriter.start() must be called before writing frames")
        max_frames = int(self.conversion.get("num_load_frames", 0) or 0)
        if max_frames > 0 and self._lidar_frame_index >= max_frames:
            return False

        frame_index = self._lidar_frame_index
        self._lidar_frame_index += 1
        self._write_lidar_frame(frame, frame_index)
        self._request_camera_frames(frame, frame_index)
        return True

    def write_camera_message(self, topic: str, image_index: int, msg: CompressedImage) -> None:
        timestamp = stamp_to_seconds(msg.header.stamp)
        buffer = self._camera_buffers[topic]
        buffer.append((image_index, timestamp, msg))
        self._prune_camera_buffer(topic, timestamp)
        request = self._pending_camera_requests.pop((topic, image_index), None)
        if request is not None:
            self._write_camera_file(request, msg)

    def finalize(self) -> Path:
        if self._pending_camera_requests:
            for request in list(self._pending_camera_requests.values()):
                self.sample_data_table.update_record_from_token(
                    request.sample_data_token,
                    is_key_frame=False,
                    is_valid=False,
                )
                self._write_dummy_camera_file(request.filename, request.calibration)
            self._pending_camera_requests.clear()
        self._connect_samples()
        self._connect_sample_data()
        self._finalize_scene()
        self._save_tables()
        self._save_status()
        return self.output_scene_dir

    def _scene_dir(self) -> Path:
        base = self.output_base / self.scene_name
        if self.conversion.get("make_t4_dataset_dir", True):
            return base / "t4_dataset"
        return base

    def _init_tables(self) -> None:
        self.log_table = TableHandler(Log)
        self.map_table = TableHandler(Map)
        self.sensor_table = TableHandler(Sensor)
        self.calibrated_sensor_table = TableHandler(CalibratedSensor)
        self.scene_table = TableHandler(Scene)
        self.sample_table = TableHandler(Sample)
        self.sample_data_table = TableHandler(SampleData)
        self.ego_pose_table = TableHandler(EgoPose)
        self.instance_table = TableHandler(Instance)
        self.sample_annotation_table = TableHandler(SampleAnnotation)
        self.category_table = TableHandler(Category)
        self.attribute_table = TableHandler(Attribute)
        self.visibility_table = TableHandler(Visibility)
        self.tables = [
            self.log_table,
            self.map_table,
            self.sensor_table,
            self.calibrated_sensor_table,
            self.scene_table,
            self.sample_table,
            self.sample_data_table,
            self.ego_pose_table,
            self.instance_table,
            self.sample_annotation_table,
            self.category_table,
            self.attribute_table,
            self.visibility_table,
        ]

    def _make_directories(self) -> None:
        self.output_anno_dir.mkdir(parents=True, exist_ok=True)
        self.output_data_dir.mkdir(parents=True, exist_ok=True)
        channels = [self.conversion["lidar_sensor"]["channel"]]
        if self._pcd_output_format() == "both":
            channels.append(f"{self.conversion['lidar_sensor']['channel']}_PCD")
        lidar_info_channel = self.conversion["lidar_sensor"].get("lidar_info_channel")
        if lidar_info_channel:
            channels.append(lidar_info_channel)
        channels.extend(camera["channel"] for camera in self.conversion.get("camera_sensors", []))
        if self._write_individual_lidar():
            for mapping in self.conversion["lidar_sensor"].get("lidar_sources_mapping", []):
                channels.append(mapping["channel"])
        for channel in sorted(set(channels)):
            (self.output_data_dir / channel).mkdir(parents=True, exist_ok=True)

    def _write_static_tables(self) -> str:
        log_token = self.log_table.insert_into_table(
            logfile="",
            vehicle=self.calibration.vehicle_id,
            data_captured="",
            location="",
        )
        self.map_table.insert_into_table(log_tokens=[log_token], category="", filename="")
        scene_token = self.scene_table.insert_into_table(
            name=self.scene_name,
            description=self.scene_description,
            log_token=log_token,
            nbr_samples=0,
            first_sample_token="tmp_token",
            last_sample_token="tmp_token",
        )
        self._ensure_sensor(self.conversion["lidar_sensor"]["channel"], "lidar")
        if self._write_individual_lidar():
            for mapping in self.conversion["lidar_sensor"].get("lidar_sources_mapping", []):
                self._ensure_sensor(mapping["channel"], "lidar")
        for camera in self.conversion.get("camera_sensors", []):
            self._ensure_sensor(camera["channel"], "camera")
        return scene_token

    def _build_camera_sync(self) -> dict[str, CameraSyncState]:
        sync = {}
        window = (
            float(self.conversion.get("system_scan_period_sec", 0.1)) * 0.5
            + float(self.conversion.get("max_camera_jitter_sec", 0.005))
        )
        for camera in self.conversion.get("camera_sensors", []):
            topic = camera["topic"]
            sync[topic] = CameraSyncState(
                timestamps=self.camera_timestamps.get(topic, []),
                delay_sec=float(camera.get("delay_msec", 0.0)) * 1e-3,
                window_sec=window,
            )
        return sync

    def _write_lidar_frame(self, frame: ConcatenatedFrame, frame_index: int) -> None:
        lidar_sensor = self.conversion["lidar_sensor"]
        lidar_channel = lidar_sensor["channel"]
        pcd_output_format = self._pcd_output_format()
        sample_extension = "pcd" if pcd_output_format == "pcd" else "pcd.bin"
        calibrated_token = self._ensure_calibrated_sensor(lidar_channel, self.calibration.base_frame)
        self._validate_lidar_points(frame, frame_index)

        timestamp = stamp_to_seconds(frame.cloud.header.stamp)
        sample_token = self.sample_table.insert_into_table(
            timestamp=misc_utils.unix_timestamp_to_nusc_timestamp(timestamp),
            scene_token=self._scene_token,
            next="",
            prev="",
        )
        ego_pose_token = self._ego_pose(timestamp)
        info_filename = ""
        lidar_info_channel = lidar_sensor.get("lidar_info_channel")
        if lidar_info_channel:
            info_filename = misc_utils.get_sample_data_filename(lidar_info_channel, frame_index, "json")
        filename = misc_utils.get_sample_data_filename(lidar_channel, frame_index, sample_extension)
        sample_data_token = self.sample_data_table.insert_into_table(
            sample_token=sample_token,
            ego_pose_token=ego_pose_token,
            calibrated_sensor_token=calibrated_token,
            filename=filename,
            fileformat=sample_extension,
            timestamp=misc_utils.unix_timestamp_to_nusc_timestamp(timestamp),
            is_key_frame=True,
            info_filename=info_filename,
            next="",
            prev="",
            width=0,
            height=0,
        )
        self._sample_data_tokens_by_channel[lidar_channel].append(sample_data_token)
        num_lidar_feats = int(lidar_sensor.get("num_lidar_feats", 7))
        self._write_lidar_points(
            frame.cloud,
            filename,
            frame_index,
            pcd_output_format,
            lidar_channel,
            num_lidar_feats,
        )
        if info_filename:
            self._save_lidar_info(frame.info, info_filename, num_lidar_feats)

    def _validate_lidar_points(self, frame: ConcatenatedFrame, frame_index: int) -> None:
        point_count = int(frame.cloud.width) * int(frame.cloud.height)
        self._max_points = max(self._max_points, point_count)
        threshold = float(self.conversion.get("lidar_points_ratio_threshold", 0.2))
        accept_drop = bool(self.conversion.get("accept_frame_drop", False))
        if self._max_points and point_count < self._max_points * threshold and not accept_drop:
            raise ValueError(
                f"Concatenated pointcloud has too few points at frame {frame_index}: "
                f"{point_count} < {self._max_points * threshold}"
            )

    def _request_camera_frames(self, frame: ConcatenatedFrame, frame_index: int) -> None:
        lidar_timestamp = stamp_to_seconds(frame.cloud.header.stamp) - float(
            self.conversion.get("lidar_latency_sec", 0.0)
        )
        samples = self.sample_table.to_records()
        sample_token = samples[-1].token
        for camera in self.conversion.get("camera_sensors", []):
            topic = camera["topic"]
            image_index = self._camera_sync[topic].match(lidar_timestamp)
            calibration = self.camera_calibrations[topic]
            filename = misc_utils.get_sample_data_filename(camera["channel"], frame_index, "jpg")

            if image_index is None:
                image_timestamp = self._dummy_camera_timestamp(topic, lidar_timestamp)
                self._insert_camera_sample_data(
                    sample_token=sample_token,
                    calibration=calibration,
                    filename=filename,
                    image_timestamp=image_timestamp,
                    is_key_frame=False,
                    is_valid=False,
                )
                self._write_dummy_camera_file(filename, calibration)
                continue

            image_timestamp = self.camera_timestamps[topic][image_index]
            sample_data_token = self._insert_camera_sample_data(
                sample_token=sample_token,
                calibration=calibration,
                filename=filename,
                image_timestamp=image_timestamp,
                is_key_frame=True,
                is_valid=True,
            )
            request = CameraWriteRequest(
                topic=topic,
                image_index=image_index,
                filename=filename,
                calibration=calibration,
                sample_data_token=sample_data_token,
            )
            buffered = self._find_buffered_camera(topic, image_index)
            if buffered is None:
                self._pending_camera_requests[(topic, image_index)] = request
            else:
                self._write_camera_file(request, buffered)

    def _insert_camera_sample_data(
        self,
        *,
        sample_token: str,
        calibration: CameraCalibration,
        filename: str,
        image_timestamp: float,
        is_key_frame: bool,
        is_valid: bool,
    ) -> str:
        sample_data_token = self.sample_data_table.insert_into_table(
            sample_token=sample_token,
            ego_pose_token=self._ego_pose(image_timestamp),
            calibrated_sensor_token=self._camera_calibrated_sensor(calibration),
            filename=filename,
            fileformat="jpg",
            timestamp=misc_utils.unix_timestamp_to_nusc_timestamp(image_timestamp),
            is_key_frame=is_key_frame,
            next="",
            prev="",
            width=calibration.width,
            height=calibration.height,
            is_valid=is_valid,
            info_filename="",
        )
        self._sample_data_tokens_by_channel[calibration.channel].append(sample_data_token)
        return sample_data_token

    def _dummy_camera_timestamp(self, topic: str, lidar_timestamp: float) -> float:
        state = self._camera_sync[topic]
        system_scan_period = float(self.conversion.get("system_scan_period_sec", 0.1))
        max_camera_jitter = float(self.conversion.get("max_camera_jitter_sec", 0.005))
        if state.current_index < len(state.timestamps):
            adjusted_next_image = state.timestamps[state.current_index] - state.delay_sec
            return adjusted_next_image - system_scan_period
        return lidar_timestamp + system_scan_period + max_camera_jitter

    def _pcd_output_format(self) -> str:
        value = str(self.conversion.get("pcd_output_format", "bin")).lower()
        if value not in {"bin", "pcd", "both"}:
            raise ValueError(f"pcd_output_format must be one of bin, pcd, both; got {value}")
        return value

    def _write_individual_lidar(self) -> bool:
        lidar_sensor = self.conversion.get("lidar_sensor", {})
        return bool(lidar_sensor.get("write_individual_lidar", False))

    def _write_lidar_points(
        self,
        cloud,
        filename: str,
        frame_index: int,
        pcd_output_format: str,
        lidar_channel: str,
        num_lidar_feats: int,
    ) -> None:
        path = self.output_scene_dir / filename
        if pcd_output_format in {"bin", "both"}:
            points = pointcloud_to_lidar_features(cloud, num_lidar_feats=num_lidar_feats)
            points.tofile(path)
        if pcd_output_format in {"pcd", "both"}:
            pcd_path = path
            if pcd_output_format == "both":
                pcd_filename = misc_utils.get_sample_data_filename(
                    f"{lidar_channel}_PCD", frame_index, "pcd"
                )
                pcd_path = self.output_scene_dir / pcd_filename
            save_lidar_pointcloud_pcd(pcd_path, cloud, num_lidar_feats=num_lidar_feats)

    def _find_buffered_camera(self, topic: str, image_index: int) -> CompressedImage | None:
        for index, _timestamp, msg in self._camera_buffers.get(topic, ()):
            if index == image_index:
                return msg
        return None

    def _prune_camera_buffer(self, topic: str, current_timestamp: float) -> None:
        retention = self._camera_retention_seconds()
        buffer = self._camera_buffers[topic]
        while buffer and current_timestamp - buffer[0][1] > retention:
            index = buffer[0][0]
            if (topic, index) in self._pending_camera_requests:
                break
            buffer.popleft()

    def _camera_retention_seconds(self) -> float:
        max_delay = max(
            [abs(float(camera.get("delay_msec", 0.0))) * 1e-3 for camera in self.conversion.get("camera_sensors", [])]
            or [0.0]
        )
        return max(
            0.5,
            float(self.conversion.get("system_scan_period_sec", 0.1))
            + float(self.conversion.get("max_camera_jitter_sec", 0.005))
            + max_delay
            + 0.1,
        )

    def _write_camera_file(self, request: CameraWriteRequest, msg: CompressedImage) -> None:
        self._write_image(msg, self.output_scene_dir / request.filename, request.calibration)

    def _write_dummy_camera_file(self, filename: str, calibration: CameraCalibration) -> None:
        path = self.output_scene_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((calibration.height, calibration.width, 3), dtype=np.uint8)
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def _ensure_sensor(self, channel: str, modality: str) -> str:
        if channel not in self._sensor_tokens:
            self._sensor_tokens[channel] = self.sensor_table.insert_into_table(
                channel=channel,
                modality=modality,
            )
        return self._sensor_tokens[channel]

    def _ensure_calibrated_sensor(self, channel: str, frame_id: str) -> str:
        if channel in self._calibrated_tokens:
            return self._calibrated_tokens[channel]
        sensor_token = self._sensor_tokens[channel]
        if frame_id == self.calibration.base_frame:
            transform = identity(self.calibration.base_frame, frame_id)
        else:
            transform = self.tf_manager.lookup_required(self.calibration.base_frame, frame_id, 0.0)
        token = self.calibrated_sensor_table.insert_into_table(
            sensor_token=sensor_token,
            translation=transform.translation,
            rotation=_rotation_wxyz(transform),
            camera_intrinsic=[],
            camera_distortion=[],
        )
        self._calibrated_tokens[channel] = token
        return token

    def _camera_calibrated_sensor(self, calibration: CameraCalibration) -> str:
        if calibration.channel in self._calibrated_tokens:
            return self._calibrated_tokens[calibration.channel]
        transform = self.tf_manager.lookup_required(
            self.calibration.base_frame,
            calibration.frame_id,
            0.0,
        )
        intrinsic, distortion = self._camera_intrinsics(calibration)
        token = self.calibrated_sensor_table.insert_into_table(
            sensor_token=self._sensor_tokens[calibration.channel],
            translation=transform.translation,
            rotation=_rotation_wxyz(transform),
            camera_intrinsic=intrinsic,
            camera_distortion=distortion,
        )
        self._calibrated_tokens[calibration.channel] = token
        return token

    def _ego_pose(self, timestamp: float) -> str:
        world_frame = self.conversion.get("world_frame_id", "map")
        tolerance_sec = float(self.conversion.get("max_ego_pose_lookup_fallback_sec", 0.5))
        transform, transform_time = self.tf_manager.lookup_nearest(
            world_frame,
            self.calibration.base_frame,
            timestamp,
            tolerance_sec=tolerance_sec,
        )
        if transform is None:
            raise KeyError(
                f"No ego pose transform from {world_frame} to {self.calibration.base_frame} "
                f"at {timestamp:.6f} within {tolerance_sec:.3f}s. "
                "Check TF coverage for the conversion time range."
            )
        if transform_time is not None and abs(transform_time - timestamp) > 1e-9:
            self._ego_pose_fallbacks += 1
            if self._ego_pose_fallbacks <= 10:
                print(
                    "[converter] ego pose fallback "
                    f"requested={timestamp:.6f} used_tf={transform_time:.6f} "
                    f"delta={(transform_time - timestamp) * 1000.0:.3f}ms",
                    flush=True,
                )
        return self.ego_pose_table.insert_into_table(
            reuse_if_duplicate=True,
            translation=transform.translation,
            rotation=_rotation_wxyz(transform),
            timestamp=misc_utils.unix_timestamp_to_nusc_timestamp(timestamp),
            twist=None,
            acceleration=None,
            geocoordinate=None,
        )

    def _save_lidar_info(
        self,
        msg: ConcatenatedPointCloudInfo,
        info_filename: str,
        num_pts_feats: int,
    ) -> None:
        sources = []
        if self._write_individual_lidar():
            topic_to_channel = {
                item["topic"]: item["channel"]
                for item in self.conversion["lidar_sensor"].get("lidar_sources_mapping", [])
            }
            for src in msg.source_info:
                channel = topic_to_channel.get(src.topic)
                if channel is None:
                    continue
                source_data = {
                    "sensor_token": self._sensor_tokens[channel],
                    "idx_begin": src.idx_begin,
                    "length": src.length,
                    "stamp": {
                        "sec": src.header.stamp.sec,
                        "nanosec": src.header.stamp.nanosec,
                    },
                }
                sources.append(source_data)
        data = {
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "num_pts_feats": num_pts_feats,
            "sources": sources,
        }
        with (self.output_scene_dir / info_filename).open("w") as fp:
            json.dump(data, fp, indent=4)

    def _write_image(self, msg: CompressedImage, path: Path, calibration: CameraCalibration) -> tuple[int, int, int]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.conversion.get("undistort_image", False):
            with path.open("wb") as fp:
                fp.write(bytes(msg.data))
            return calibration.height, calibration.width, 3

        image = _compressed_image_to_numpy(msg)
        if image is None:
            raise ValueError(f"Failed to decode image on {calibration.topic}")
        map1, map2 = self._rectify_maps_for(calibration)
        rectified = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(path), rectified, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return rectified.shape

    def _rectify_maps_for(self, calibration: CameraCalibration) -> tuple[np.ndarray, np.ndarray]:
        if calibration.topic not in self._rectify_maps:
            self._rectify_maps[calibration.topic] = cv2.initUndistortRectifyMap(
                calibration.k.reshape(3, 3),
                calibration.d,
                calibration.r.reshape(3, 3),
                calibration.p.reshape(3, 4)[:3, :3],
                (calibration.width, calibration.height),
                cv2.CV_16SC2,
            )
        return self._rectify_maps[calibration.topic]

    def _camera_intrinsics(self, calibration: CameraCalibration):
        if self.conversion.get("undistort_image", False):
            return calibration.rectified_intrinsic, [0.0] * 5
        return calibration.raw_intrinsic, calibration.distortion

    def _connect_samples(self) -> None:
        tokens = [record.token for record in self.sample_table.to_records()]
        for index in range(1, len(tokens)):
            self.sample_table.update_record_from_token(tokens[index - 1], next=tokens[index])
            self.sample_table.update_record_from_token(tokens[index], prev=tokens[index - 1])

    def _connect_sample_data(self) -> None:
        for tokens in self._sample_data_tokens_by_channel.values():
            valid_tokens = [
                token
                for token in tokens
                if self.sample_data_table.get_record_from_token(token).is_valid
            ]
            for index in range(1, len(valid_tokens)):
                self.sample_data_table.update_record_from_token(
                    valid_tokens[index - 1],
                    next=valid_tokens[index],
                )
                self.sample_data_table.update_record_from_token(
                    valid_tokens[index],
                    prev=valid_tokens[index - 1],
                )

    def _finalize_scene(self) -> None:
        samples = self.sample_table.to_records()
        if not samples:
            raise ValueError("No LiDAR samples were generated")
        scene = self.scene_table.to_records()[0]
        self.scene_table.update_record_from_token(
            scene.token,
            nbr_samples=len(samples),
            first_sample_token=samples[0].token,
            last_sample_token=samples[-1].token,
        )

    def _save_tables(self) -> None:
        for table in self.tables:
            table.save_json(str(self.output_anno_dir))

    def _save_status(self) -> None:
        status = {
            "t4dataset_rosbag_converter": {
                "scene_name": self.scene_name,
                "vehicle_id": self.calibration.vehicle_id,
                "sensor_model": self.calibration.sensor_model,
                "conversion": self.conversion,
            }
        }
        with (self.output_scene_dir / "status.json").open("w") as fp:
            json.dump(status, fp, indent=4, default=str)


def _rotation_wxyz(transform: RigidTransform) -> tuple[float, float, float, float]:
    return (
        transform.rotation_xyzw[3],
        transform.rotation_xyzw[0],
        transform.rotation_xyzw[1],
        transform.rotation_xyzw[2],
    )


def _camera_info_topic(image_topic: str) -> str:
    parts = image_topic.strip("/").split("/")
    if len(parts) >= 4:
        return "/" + "/".join(parts[:3] + ["camera_info"])
    return image_topic.rsplit("/", 1)[0] + "/camera_info"


def _compressed_image_to_numpy(msg: CompressedImage) -> np.ndarray | None:
    try:
        image_buffer = np.frombuffer(msg.data, dtype=np.uint8)
    except TypeError:
        image_buffer = np.asarray(msg.data, dtype=np.uint8)
    if image_buffer.size == 0:
        return None
    return cv2.imdecode(image_buffer, cv2.IMREAD_ANYCOLOR)
