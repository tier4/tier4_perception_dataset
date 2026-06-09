from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import os.path as osp
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Tuple

import kognic.io.model as KognicModel
import numpy as np
from scipy.spatial.transform import Rotation

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

_LIDAR_CONCAT_CHANNEL = "LIDAR_CONCAT"
_NUM_POINT_FEATURES = 5  # x, y, z, intensity, auxiliary/ring_idx
_BYTES_PER_POINT = _NUM_POINT_FEATURES * 4  # float32


class NonAnnotatedT4ToKognicConverter(AbstractConverter[None]):
    """Convert non-annotated T4 data to the Kognic IO staging layout.

    This intentionally mirrors ``tier_4_code/tier_iv_t4_extractor.py`` while
    preserving the perception_dataset converter contract:

        <output_base>/<input_item_name>/
            calibration.json
            ego_poses.json
            cameras/<camera_name>/<timestamp_ns>.jpg
            lidar/<lidar_name>/<timestamp_ns>.csv
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        camera_sensors: list,
        annotation_hz: int = 10,
        workers_number: int = 32,
        drop_camera_token_not_found: bool = False,
    ):
        super().__init__(input_base, output_base)
        self._camera_channels: List[str] = [cam["channel"] for cam in camera_sensors]
        self._annotation_hz = annotation_hz
        self._workers_number = workers_number
        self._drop_camera_token_not_found = drop_camera_token_not_found

    def convert(self) -> None:
        start = time.time()

        for seq_path, out_dir in self._iter_input_sequences():
            logger.info(f"[BEGIN] {seq_path} -> {out_dir}")
            self._convert_one_scene(seq_path, out_dir)
            logger.info(f"[DONE]  {seq_path} -> {out_dir}")

        logger.info(f"Elapsed: {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Sequence discovery
    # ------------------------------------------------------------------

    def _iter_input_sequences(self) -> List[Tuple[Path, Path]]:
        input_base = Path(self._input_base)
        output_base = Path(self._output_base)

        if self._is_sequence_root(input_base):
            return [(input_base, output_base / input_base.name)]

        pairs: List[Tuple[Path, Path]] = []
        for item in sorted(Path(path) for path in glob.glob(osp.join(self._input_base, "*"))):
            if not item.is_dir() or item.name == "extracted_data":
                continue

            seq_roots = self._find_sequence_roots(item)
            if not seq_roots:
                logger.warning(f"No T4 sequence root found under {item}; skipping")
                continue

            if len(seq_roots) == 1:
                pairs.append((seq_roots[0], output_base / item.name))
            else:
                for seq_root in seq_roots:
                    pairs.append((seq_root, output_base / item.name / seq_root.name))

        return pairs

    def _find_sequence_roots(self, root: Path) -> List[Path]:
        if self._is_sequence_root(root):
            return [root]

        return sorted(
            path
            for path in root.rglob("*")
            if self._is_sequence_root(path) and "extracted_data" not in path.parts
        )

    @staticmethod
    def _is_sequence_root(path: Path) -> bool:
        annotation_dir = path / "annotation"
        data_dir = path / "data"
        required_annotations = [
            "sensor.json",
            "calibrated_sensor.json",
            "sample.json",
            "sample_data.json",
            "ego_pose.json",
        ]
        return (
            path.is_dir()
            and annotation_dir.is_dir()
            and data_dir.is_dir()
            and all((annotation_dir / name).exists() for name in required_annotations)
        )

    # ------------------------------------------------------------------
    # Scene conversion
    # ------------------------------------------------------------------

    def _convert_one_scene(self, input_dir: Path | str, output_dir: Path | str) -> None:
        seq_path = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._build_lookup_maps(seq_path)
        self._has_lidar_concat_info = (seq_path / "data" / "LIDAR_CONCAT_INFO").is_dir()
        self._lidar_channels = self._discover_lidar_channels()
        self._frame_records = self._build_frame_records()
        logger.info(
            f"Selected {len(self._frame_records)} frames for annotation_hz={self._annotation_hz}"
        )

        if not self._has_lidar_concat_info and _LIDAR_CONCAT_CHANNEL in self._lidar_channels:
            logger.warning(
                "LIDAR_CONCAT_INFO is missing. Exporting fused LIDAR_CONCAT "
                "as a single Kognic LiDAR stream instead of per-source LiDARs."
            )

        calibration = self._extract_calibration(seq_path)
        with open(out_dir / "calibration.json", "w") as f:
            json.dump({k: v.model_dump() for k, v in calibration.items()}, f, indent=2)
        logger.info(f"Calibration saved ({len(calibration)} sensors)")

        ego_poses = self._extract_ego_poses()
        with open(out_dir / "ego_poses.json", "w") as f:
            json.dump({k: v.model_dump() for k, v in ego_poses.items()}, f, indent=2)
        logger.info(f"Ego poses saved ({len(ego_poses)} frames)")

        pending_copies: List[Tuple[Path, Path]] = []
        for camera_channel in self._camera_channels:
            pending_copies.extend(self._collect_image_copies(seq_path, out_dir, camera_channel))

        with ThreadPoolExecutor(max_workers=self._workers_number) as executor:
            list(executor.map(lambda args: _copy_file(*args), pending_copies))

        for lidar_channel in self._lidar_channels:
            self._extract_pointclouds(seq_path, out_dir, lidar_channel)

    def _load_annotation(self, seq_path: Path, name: str) -> list | dict:
        with open(seq_path / "annotation" / name) as f:
            return json.load(f)

    def _build_lookup_maps(self, seq_path: Path) -> None:
        sensors = self._load_annotation(seq_path, "sensor.json")
        self._sensors = sensors
        self._token_to_channel = {s["token"]: s["channel"] for s in sensors}
        self._channel_to_token = {s["channel"]: s["token"] for s in sensors}

        calib_sensors = self._load_annotation(seq_path, "calibrated_sensor.json")
        self._calib_by_token = {c["token"]: c for c in calib_sensors}
        self._calib_by_sensor_token = {c["sensor_token"]: c for c in calib_sensors}

        samples = self._load_annotation(seq_path, "sample.json")
        self._samples = sorted(samples, key=lambda s: s["timestamp"])

        self._sample_data_by_sample: Dict[str, list] = {}
        self._sample_data_by_sample_and_channel: Dict[str, Dict[str, dict]] = {}
        self._sample_data_by_channel: Dict[str, list] = {}
        self._sample_data_by_channel_and_frame_id: Dict[str, Dict[str, dict]] = {}
        sample_data = self._load_annotation(seq_path, "sample_data.json")
        for sd in sample_data:
            self._sample_data_by_sample.setdefault(sd["sample_token"], []).append(sd)
            calib = self._calib_by_token.get(sd["calibrated_sensor_token"])
            if not calib:
                continue
            channel = self._token_to_channel.get(calib["sensor_token"])
            if not channel:
                continue
            self._sample_data_by_sample_and_channel.setdefault(sd["sample_token"], {})[
                channel
            ] = sd
            self._sample_data_by_channel.setdefault(channel, []).append(sd)
            frame_id = Path(sd["filename"]).stem.split(".")[0]
            self._sample_data_by_channel_and_frame_id.setdefault(channel, {})[frame_id] = sd

        for channel in self._sample_data_by_channel:
            self._sample_data_by_channel[channel] = sorted(
                self._sample_data_by_channel[channel],
                key=lambda sample_data_record: sample_data_record["timestamp"],
            )

        ego_poses = self._load_annotation(seq_path, "ego_pose.json")
        self._ego_pose_by_token = {ep["token"]: ep for ep in ego_poses}

    def _discover_lidar_channels(self) -> List[str]:
        if not self._has_lidar_concat_info:
            if _LIDAR_CONCAT_CHANNEL in self._channel_to_token:
                return [_LIDAR_CONCAT_CHANNEL]
            return []

        return [
            sensor["channel"]
            for sensor in self._sensors
            if sensor.get("modality") == "lidar" and sensor.get("channel") != _LIDAR_CONCAT_CHANNEL
        ]

    def _has_existing_channel_file(self, seq_path: Path, channel: str) -> bool:
        return any(
            (seq_path / sample_data["filename"]).exists()
            for sample_data in self._sample_data_by_channel.get(channel, [])
        )

    def _build_frame_records(self) -> List[Dict[str, dict]]:
        """Build selected output frames according to the requested annotation_hz.

        ``sample.json`` is usually 1 Hz, while ``sample_data.json`` may contain
        10 Hz sensor frames. Keep 1 Hz behavior sample-driven to match the
        existing extractor output, and use sample_data-driven selection for
        higher requested frequencies.
        """
        if self._annotation_hz <= 1:
            return self._build_sample_level_frame_records()

        anchor_channel = self._select_high_frequency_anchor_channel()
        if anchor_channel is None:
            return self._build_sample_level_frame_records()

        step = max(1, int(round(10 / self._annotation_hz)))
        anchor_records = self._sample_data_by_channel.get(anchor_channel, [])
        frame_records: List[Dict[str, dict]] = []
        for anchor_record in anchor_records[::step]:
            frame_id = Path(anchor_record["filename"]).stem.split(".")[0]
            frame_record: Dict[str, dict] = {}

            for channel in self._channels_for_frame_records():
                sample_data = self._sample_data_by_channel_and_frame_id.get(channel, {}).get(
                    frame_id
                )
                if sample_data is not None:
                    frame_record[channel] = sample_data

            if frame_record:
                frame_records.append(frame_record)

        return frame_records

    def _build_sample_level_frame_records(self) -> List[Dict[str, dict]]:
        frame_records: List[Dict[str, dict]] = []
        for sample in self._samples:
            sample_by_channel = self._sample_data_by_sample_and_channel.get(
                sample["token"], {}
            )
            frame_record = {
                channel: sample_data
                for channel, sample_data in sample_by_channel.items()
                if channel in self._channels_for_frame_records()
            }
            if frame_record:
                frame_records.append(frame_record)
        return frame_records

    def _select_high_frequency_anchor_channel(self) -> Optional[str]:
        if self._sample_data_by_channel.get(_LIDAR_CONCAT_CHANNEL):
            return _LIDAR_CONCAT_CHANNEL

        for camera_channel in self._camera_channels:
            if self._sample_data_by_channel.get(camera_channel):
                return camera_channel

        return None

    def _channels_for_frame_records(self) -> List[str]:
        return [_LIDAR_CONCAT_CHANNEL, *self._camera_channels]

    def _get_reference_sample_data(self, frame_record: Dict[str, dict]) -> Optional[dict]:
        if _LIDAR_CONCAT_CHANNEL in frame_record:
            return frame_record[_LIDAR_CONCAT_CHANNEL]

        for camera_channel in self._camera_channels:
            if camera_channel in frame_record:
                return frame_record[camera_channel]

        return next(iter(frame_record.values()), None)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _extract_calibration(self, seq_path: Path) -> Dict[str, KognicModel.BaseCalibration]:
        calibration: Dict[str, KognicModel.BaseCalibration] = {}

        for camera_channel in self._camera_channels:
            sensor_token = self._channel_to_token.get(camera_channel)
            if sensor_token is None:
                logger.warning(f"Camera {camera_channel} not found in {seq_path}; skipping")
                continue
            if not self._has_existing_channel_file(seq_path, camera_channel):
                logger.warning(
                    f"Camera {camera_channel} has no files in {seq_path}; skipping"
                )
                continue

            calib = self._calib_by_sensor_token[sensor_token]
            translation = calib["translation"]
            rotation = calib["rotation"]  # [w, x, y, z]
            intrinsic = calib["camera_intrinsic"]
            distortion = calib["camera_distortion"]
            width, height = self._read_image_dims(seq_path, camera_channel)

            calibration[camera_channel] = KognicModel.PinholeCalibration(
                position=KognicModel.Position(
                    x=float(translation[0]),
                    y=float(translation[1]),
                    z=float(translation[2]),
                ),
                rotation_quaternion=KognicModel.RotationQuaternion(
                    w=float(rotation[0]),
                    x=float(rotation[1]),
                    y=float(rotation[2]),
                    z=float(rotation[3]),
                ),
                camera_matrix=KognicModel.CameraMatrix(
                    fx=float(intrinsic[0][0]),
                    fy=float(intrinsic[1][1]),
                    cx=float(intrinsic[0][2]),
                    cy=float(intrinsic[1][2]),
                ),
                distortion_coefficients=KognicModel.DistortionCoefficients(
                    k1=float(distortion[0]) if len(distortion) > 0 else 0.0,
                    k2=float(distortion[1]) if len(distortion) > 1 else 0.0,
                    p1=float(distortion[2]) if len(distortion) > 2 else 0.0,
                    p2=float(distortion[3]) if len(distortion) > 3 else 0.0,
                    k3=float(distortion[4]) if len(distortion) > 4 else 0.0,
                ),
                image_height=height,
                image_width=width,
            )

        for lidar_channel in self._lidar_channels:
            if lidar_channel not in self._channel_to_token:
                logger.warning(f"LiDAR {lidar_channel} not found in {seq_path}; skipping")
                continue
            calibration[lidar_channel] = KognicModel.LidarCalibration(
                position=KognicModel.Position(x=0.0, y=0.0, z=0.0),
                rotation_quaternion=KognicModel.RotationQuaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            )

        return calibration

    def _read_image_dims(self, seq_path: Path, camera_channel: str) -> Tuple[int, int]:
        for sample_data in self._sample_data_by_channel.get(camera_channel, []):
            width = int(sample_data.get("width") or 0)
            height = int(sample_data.get("height") or 0)
            if width > 0 and height > 0:
                return width, height

        # Fallback for malformed T4 records that omit image dimensions.
        try:
            from PIL import Image
        except ImportError as exc:
            raise FileNotFoundError(
                f"No image dimensions recorded for camera {camera_channel}, "
                "and Pillow is not installed to inspect image files."
            ) from exc

        sample_dir = seq_path / "data" / camera_channel
        sample_jpg = next(iter(sorted(sample_dir.glob("*.jpg"))), None)
        if sample_jpg is None:
            raise FileNotFoundError(f"No images found for camera {camera_channel} at {sample_dir}")
        with Image.open(sample_jpg) as image:
            return image.size

    # ------------------------------------------------------------------
    # Ego poses
    # ------------------------------------------------------------------

    def _extract_ego_poses(self) -> Dict[str, KognicModel.EgoVehiclePose]:
        frame_ego_tokens = []
        for frame_record in self._frame_records:
            sample_data = self._get_reference_sample_data(frame_record)
            if sample_data is not None:
                frame_ego_tokens.append(sample_data["ego_pose_token"])

        if not frame_ego_tokens:
            logger.warning("No ego poses found")
            return {}

        t0 = _build_transform(self._ego_pose_by_token[frame_ego_tokens[0]])
        t0_inv = np.linalg.inv(t0)

        ego_poses: Dict[str, KognicModel.EgoVehiclePose] = {}
        for frame_idx, ego_token in enumerate(frame_ego_tokens):
            transform = _build_transform(self._ego_pose_by_token[ego_token])
            relative_transform = t0_inv @ transform
            position = relative_transform[:3, 3]
            quat = _matrix_to_quat_wxyz(relative_transform[:3, :3])

            ego_poses[str(frame_idx)] = KognicModel.EgoVehiclePose(
                position=KognicModel.Position(
                    x=round(float(position[0]), 10),
                    y=round(float(position[1]), 10),
                    z=round(float(position[2]), 10),
                ),
                rotation=KognicModel.RotationQuaternion(
                    w=round(float(quat[0]), 10),
                    x=round(float(quat[1]), 10),
                    y=round(float(quat[2]), 10),
                    z=round(float(quat[3]), 10),
                ),
            )

        return ego_poses

    # ------------------------------------------------------------------
    # Sensor data
    # ------------------------------------------------------------------

    def _collect_image_copies(
        self, seq_path: Path, out_dir: Path, camera_channel: str
    ) -> List[Tuple[Path, Path]]:
        if camera_channel not in self._channel_to_token:
            logger.warning(f"Camera {camera_channel} not found in {seq_path}; skipping")
            return []
        if not self._has_existing_channel_file(seq_path, camera_channel):
            logger.warning(f"Camera {camera_channel} has no files in {seq_path}; skipping")
            return []

        camera_dir = out_dir / "cameras" / camera_channel
        camera_dir.mkdir(parents=True, exist_ok=True)

        copies: List[Tuple[Path, Path]] = []
        for frame_record in self._frame_records:
            sample_data = frame_record.get(camera_channel)
            if sample_data is None:
                if self._drop_camera_token_not_found:
                    logger.warning(
                        f"Camera {camera_channel} missing for selected frame; skipping"
                    )
                continue

            timestamp_ns = int(sample_data["timestamp"]) * 1000
            src = seq_path / sample_data["filename"]
            dst = camera_dir / f"{timestamp_ns}.jpg"
            if src.exists() and not dst.exists():
                copies.append((src, dst))

        logger.info(f"{camera_channel}: {len(copies)} image copies queued")
        return copies

    def _extract_pointclouds(self, seq_path: Path, out_dir: Path, lidar_channel: str) -> None:
        sensor_token = self._channel_to_token.get(lidar_channel)
        if sensor_token is None:
            logger.warning(f"LiDAR {lidar_channel} not found in {seq_path}; skipping")
            return

        lidar_dir = out_dir / "lidar" / lidar_channel
        lidar_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for frame_record in self._frame_records:
            concat_sample_data = frame_record.get(_LIDAR_CONCAT_CHANNEL)
            if concat_sample_data is None:
                continue

            bin_path = seq_path / concat_sample_data["filename"]
            if not bin_path.exists():
                raise FileNotFoundError(
                    f"Required LIDAR_CONCAT point cloud is missing: {bin_path}"
                )

            if lidar_channel == _LIDAR_CONCAT_CHANNEL:
                timestamp_ns = int(concat_sample_data["timestamp"]) * 1000
                points = np.fromfile(bin_path, dtype=np.float32).reshape(
                    -1, _NUM_POINT_FEATURES
                )
                csv_path = lidar_dir / f"{timestamp_ns}.csv"
                _save_pointcloud_csv(csv_path, timestamp_ns, points)
                count += 1
                continue

            info_filename = concat_sample_data.get("info_filename")
            if not info_filename:
                raise FileNotFoundError(
                    f"LIDAR_CONCAT_INFO is required but missing in sample_data for "
                    f"sample_data {concat_sample_data['token']}"
                )

            info_path = seq_path / info_filename
            if not info_path.exists():
                raise FileNotFoundError(f"Required LIDAR_CONCAT_INFO file is missing: {info_path}")

            with open(info_path) as f:
                info = json.load(f)

            source = next(
                (src for src in info["sources"] if src["sensor_token"] == sensor_token),
                None,
            )
            if source is None:
                continue

            idx_begin = int(source["idx_begin"])
            length = int(source["length"])
            if length == 0:
                continue

            with open(bin_path, "rb") as f:
                f.seek(idx_begin * _BYTES_PER_POINT)
                raw = f.read(length * _BYTES_PER_POINT)

            timestamp_ns = _stamp_to_ns(source.get("stamp"))
            if timestamp_ns is None:
                timestamp_ns = int(concat_sample_data["timestamp"]) * 1000

            points = np.frombuffer(raw, dtype=np.float32).reshape(-1, _NUM_POINT_FEATURES)
            csv_path = lidar_dir / f"{timestamp_ns}.csv"
            _save_pointcloud_csv(csv_path, timestamp_ns, points)
            count += 1

        logger.info(f"{lidar_channel}: {count} point clouds extracted")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _save_pointcloud_csv(csv_path: Path, timestamp_ns: int, points: np.ndarray) -> None:
    arr = np.empty((len(points), 5), dtype=np.float64)
    arr[:, 0] = timestamp_ns
    arr[:, 1:5] = points[:, 0:4]

    np.savetxt(
        csv_path,
        arr,
        delimiter=",",
        header="ts_gps,x,y,z,intensity",
        comments="",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"],
    )


def _stamp_to_ns(stamp: dict | None) -> Optional[int]:
    if not stamp:
        return None
    return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])


def _build_transform(ego_pose: dict) -> np.ndarray:
    transform = np.eye(4)
    rotation = ego_pose["rotation"]
    if Rotation is not None:
        transform[:3, :3] = Rotation.from_quat(
            [rotation[1], rotation[2], rotation[3], rotation[0]]
        ).as_matrix()
    else:
        transform[:3, :3] = _quat_wxyz_to_matrix(rotation)
    transform[:3, 3] = ego_pose["translation"]
    return transform


def _quat_wxyz_to_matrix(quat: list) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    norm = w * w + x * x + y * y + z * z
    scale = 2.0 / norm

    wx, wy, wz = scale * w * x, scale * w * y, scale * w * z
    xx, xy, xz = scale * x * x, scale * x * y, scale * x * z
    yy, yz, zz = scale * y * y, scale * y * z, scale * z * z

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    if Rotation is not None:
        quat_xyzw = Rotation.from_matrix(matrix).as_quat()
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    return np.array([w, x, y, z], dtype=np.float64)
