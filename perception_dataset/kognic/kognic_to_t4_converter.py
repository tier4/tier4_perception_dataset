"""Kognic staging format → T4 non-annotated dataset converter."""

import json
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
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

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import (
    LIDAR_CONCAT_BYTES_PER_POINT,
    LIDAR_CONCAT_CHANNEL,
    LIDAR_CONCAT_NUM_POINT_FEATURES,
    SENSOR_MODALITY_ENUM,
)
from perception_dataset.t4_dataset.table_handler import TableHandler
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class KognicToT4Converter(AbstractConverter[None]):
    """Convert Kognic staging format to T4 non-annotated dataset format.

    Expected Kognic input layout::

        <seq_dir>/
            calibration.json
            ego_poses.json
            cameras/<channel>/<timestamp_ns>.jpg
            lidar/<channel>/<timestamp_ns>.csv

    Produced T4 output layout::

        <out_dir>/
            annotation/
                sensor.json  calibrated_sensor.json  ego_pose.json
                sample.json  sample_data.json  scene.json  log.json  map.json
                ... (empty annotation tables)
            data/
                <channel>/<000000.pcd.bin | 000000.jpg>
                LIDAR_CONCAT/<000000.pcd.bin>       (only when multiple lidar sources)
                LIDAR_CONCAT_INFO/<000000.json>     (only when multiple lidar sources)
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        scene_name: str = "",
    ):
        super().__init__(input_base, output_base)
        self._scene_name = scene_name

    # ------------------------------------------------------------------
    # AbstractConverter contract
    # ------------------------------------------------------------------

    def convert(self) -> None:
        start = time.time()
        input_base = Path(self._input_base)
        output_base = Path(self._output_base)

        if self._is_kognic_sequence(input_base):
            name = self._scene_name or input_base.name
            self._convert_one_scene(input_base, output_base / name)
        else:
            for item in sorted(p for p in input_base.iterdir() if p.is_dir()):
                if self._is_kognic_sequence(item):
                    self._convert_one_scene(item, output_base / item.name)

        logger.info(f"Elapsed: {time.time() - start:.1f}s")

    @staticmethod
    def _is_kognic_sequence(path: Path) -> bool:
        return (path / "calibration.json").exists() and (path / "ego_poses.json").exists()

    # ------------------------------------------------------------------
    # Scene conversion
    # ------------------------------------------------------------------

    def _convert_one_scene(self, seq_path: Path, out_dir: Path) -> None:
        logger.info(f"[BEGIN] {seq_path} -> {out_dir}")

        with open(seq_path / "calibration.json") as f:
            calibration: Dict[str, dict] = json.load(f)
        with open(seq_path / "ego_poses.json") as f:
            ego_poses_raw: Dict[str, dict] = json.load(f)

        lidar_channels = sorted(
            ch for ch, cal in calibration.items() if "camera_matrix" not in cal
        )
        camera_channels = sorted(ch for ch, cal in calibration.items() if "camera_matrix" in cal)
        multi_lidar = len(lidar_channels) > 1

        frames = sorted(ego_poses_raw.keys(), key=lambda k: int(k))
        logger.info(f"Frames={len(frames)}, lidars={lidar_channels}, cameras={camera_channels}")

        self._init_tables()

        anno_dir = out_dir / "annotation"
        anno_dir.mkdir(parents=True, exist_ok=True)
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # ---- static records ----
        log_token = self._log_table.insert_into_table(
            logfile="", vehicle="", data_captured="", location=""
        )
        self._map_table.insert_into_table(log_tokens=[log_token], category="", filename="")
        scene_name = self._scene_name or seq_path.name
        scene_token = self._scene_table.insert_into_table(
            name=scene_name,
            description="",
            log_token=log_token,
            nbr_samples=0,
            first_sample_token="tmp_token",
            last_sample_token="tmp_token",
        )

        # ---- sensors + calibrated sensors ----
        channel_to_sensor_token: Dict[str, str] = {}
        channel_to_cal_token: Dict[str, str] = {}

        for channel, cal in calibration.items():
            is_camera = "camera_matrix" in cal
            modality = (
                SENSOR_MODALITY_ENUM.CAMERA.value
                if is_camera
                else SENSOR_MODALITY_ENUM.LIDAR.value
            )
            sensor_token = self._sensor_table.insert_into_table(channel=channel, modality=modality)
            channel_to_sensor_token[channel] = sensor_token

            translation = [cal["position"]["x"], cal["position"]["y"], cal["position"]["z"]]
            rot = cal["rotation_quaternion"]
            rotation = [rot["w"], rot["x"], rot["y"], rot["z"]]

            if is_camera:
                cm = cal["camera_matrix"]
                intrinsic = [[cm["fx"], 0.0, cm["cx"]], [0.0, cm["fy"], cm["cy"]], [0.0, 0.0, 1.0]]
                dc = cal.get("distortion_coefficients", {})
                distortion = [
                    dc.get("k1", 0.0),
                    dc.get("k2", 0.0),
                    dc.get("p1", 0.0),
                    dc.get("p2", 0.0),
                    dc.get("k3", 0.0),
                ]
                cal_token = self._calibrated_sensor_table.insert_into_table(
                    sensor_token=sensor_token,
                    translation=translation,
                    rotation=rotation,
                    camera_intrinsic=intrinsic,
                    camera_distortion=distortion,
                )
            else:
                cal_token = self._calibrated_sensor_table.insert_into_table(
                    sensor_token=sensor_token,
                    translation=translation,
                    rotation=rotation,
                    camera_intrinsic=[],
                    camera_distortion=[],
                )
            channel_to_cal_token[channel] = cal_token

        if multi_lidar:
            concat_sensor_token = self._sensor_table.insert_into_table(
                channel=LIDAR_CONCAT_CHANNEL,
                modality=SENSOR_MODALITY_ENUM.LIDAR.value,
            )
            channel_to_sensor_token[LIDAR_CONCAT_CHANNEL] = concat_sensor_token
            concat_cal_token = self._calibrated_sensor_table.insert_into_table(
                sensor_token=concat_sensor_token,
                translation=[0.0, 0.0, 0.0],
                rotation=[1.0, 0.0, 0.0, 0.0],
                camera_intrinsic=[],
                camera_distortion=[],
            )
            channel_to_cal_token[LIDAR_CONCAT_CHANNEL] = concat_cal_token

        # ---- create data subdirectories ----
        output_lidar_channel = (
            LIDAR_CONCAT_CHANNEL
            if multi_lidar
            else (lidar_channels[0] if lidar_channels else None)
        )
        for channel in camera_channels:
            (data_dir / channel).mkdir(parents=True, exist_ok=True)
        if output_lidar_channel:
            (data_dir / output_lidar_channel).mkdir(parents=True, exist_ok=True)
        if multi_lidar:
            (data_dir / "LIDAR_CONCAT_INFO").mkdir(parents=True, exist_ok=True)

        # ---- per-frame conversion ----
        sample_token_list: List[str] = []
        sd_tokens_by_channel: Dict[str, List[str]] = {}
        for ch in ([output_lidar_channel] if output_lidar_channel else []) + camera_channels:
            sd_tokens_by_channel[ch] = []

        for frame_idx, frame_id in enumerate(frames):
            ep = ego_poses_raw[frame_id]
            pos, rot = ep["position"], ep["rotation"]

            lidar_ts_ns = _frame_timestamp_ns(seq_path, lidar_channels, int(frame_id))
            timestamp_us = lidar_ts_ns // 1000 if lidar_ts_ns is not None else frame_idx * 100_000

            ego_pose_token = self._ego_pose_table.insert_into_table(
                translation=[pos["x"], pos["y"], pos["z"]],
                rotation=[rot["w"], rot["x"], rot["y"], rot["z"]],
                timestamp=timestamp_us,
            )
            sample_token = self._sample_table.insert_into_table(
                timestamp=timestamp_us,
                scene_token=scene_token,
                next="",
                prev="",
            )
            sample_token_list.append(sample_token)

            # --- lidar ---
            if output_lidar_channel is not None:
                if multi_lidar:
                    info_filename_rel = self._write_merged_lidar(
                        seq_path=seq_path,
                        data_dir=data_dir,
                        frame_idx=frame_idx,
                        lidar_channels=lidar_channels,
                        channel_to_sensor_token=channel_to_sensor_token,
                    )
                    lidar_filename = f"data/{LIDAR_CONCAT_CHANNEL}/{frame_idx:06d}.pcd.bin"
                    sd_token = self._sample_data_table.insert_into_table(
                        sample_token=sample_token,
                        ego_pose_token=ego_pose_token,
                        calibrated_sensor_token=channel_to_cal_token[LIDAR_CONCAT_CHANNEL],
                        filename=lidar_filename,
                        fileformat="pcd.bin",
                        width=0,
                        height=0,
                        timestamp=timestamp_us,
                        is_key_frame=True,
                        next="",
                        prev="",
                        is_valid=True,
                        info_filename=info_filename_rel,
                    )
                    sd_tokens_by_channel[LIDAR_CONCAT_CHANNEL].append(sd_token)
                else:
                    channel = lidar_channels[0]
                    csv_path = _indexed_file(seq_path / "lidar" / channel, int(frame_id), "*.csv")
                    if csv_path is not None:
                        points, _ = _load_csv(csv_path)
                        bin_path = data_dir / channel / f"{frame_idx:06d}.pcd.bin"
                        _write_pcd_bin(bin_path, points)
                        sd_token = self._sample_data_table.insert_into_table(
                            sample_token=sample_token,
                            ego_pose_token=ego_pose_token,
                            calibrated_sensor_token=channel_to_cal_token[channel],
                            filename=f"data/{channel}/{frame_idx:06d}.pcd.bin",
                            fileformat="pcd.bin",
                            width=0,
                            height=0,
                            timestamp=timestamp_us,
                            is_key_frame=True,
                            next="",
                            prev="",
                            is_valid=True,
                            info_filename="",
                        )
                        sd_tokens_by_channel[channel].append(sd_token)

            # --- cameras ---
            for camera_channel in camera_channels:
                src = _indexed_file(seq_path / "cameras" / camera_channel, int(frame_id), "*.jpg")
                if src is None:
                    continue
                dst = data_dir / camera_channel / f"{frame_idx:06d}.jpg"
                shutil.copy2(src, dst)
                cal = calibration[camera_channel]
                img_ts_ns = int(src.stem)
                sd_token = self._sample_data_table.insert_into_table(
                    sample_token=sample_token,
                    ego_pose_token=ego_pose_token,
                    calibrated_sensor_token=channel_to_cal_token[camera_channel],
                    filename=f"data/{camera_channel}/{frame_idx:06d}.jpg",
                    fileformat="jpg",
                    width=cal.get("image_width", 0),
                    height=cal.get("image_height", 0),
                    timestamp=img_ts_ns // 1000,
                    is_key_frame=True,
                    next="",
                    prev="",
                    is_valid=True,
                    info_filename="",
                )
                sd_tokens_by_channel[camera_channel].append(sd_token)

        # ---- wire up next/prev chains ----
        _connect_chain(sample_token_list, self._sample_table)
        for token_list in sd_tokens_by_channel.values():
            _connect_chain(token_list, self._sample_data_table)

        # ---- update scene ----
        scene_rec = self._scene_table.to_records()[0]
        scene_rec.nbr_samples = len(sample_token_list)
        if sample_token_list:
            scene_rec.first_sample_token = sample_token_list[0]
            scene_rec.last_sample_token = sample_token_list[-1]

        # ---- save tables ----
        for tbl in self.__dict__.values():
            if isinstance(tbl, TableHandler):
                tbl.save_json(anno_dir)

        logger.info(f"[DONE]  {seq_path} -> {out_dir} ({len(sample_token_list)} frames)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        self._log_table = TableHandler(Log)
        self._map_table = TableHandler(Map)
        self._sensor_table = TableHandler(Sensor)
        self._calibrated_sensor_table = TableHandler(CalibratedSensor)
        self._scene_table = TableHandler(Scene)
        self._sample_table = TableHandler(Sample)
        self._sample_data_table = TableHandler(SampleData)
        self._ego_pose_table = TableHandler(EgoPose)
        self._instance_table = TableHandler(Instance)
        self._sample_annotation_table = TableHandler(SampleAnnotation)
        self._category_table = TableHandler(Category)
        self._attribute_table = TableHandler(Attribute)
        self._visibility_table = TableHandler(Visibility)

    def _write_merged_lidar(
        self,
        seq_path: Path,
        data_dir: Path,
        frame_idx: int,
        lidar_channels: List[str],
        channel_to_sensor_token: Dict[str, str],
    ) -> str:
        """Merge per-source LiDAR CSVs into one LIDAR_CONCAT .pcd.bin and write the INFO sidecar.

        Returns the INFO filename as a relative path string (relative to the T4 root).
        """
        sources = []
        all_points: List[np.ndarray] = []
        idx_cursor = 0

        for channel in lidar_channels:
            csv_path = _indexed_file(seq_path / "lidar" / channel, frame_idx, "*.csv")
            if csv_path is None:
                continue

            points, ts_ns = _load_csv(csv_path)
            length = len(points)
            all_points.append(points)

            sec = ts_ns // 1_000_000_000
            nanosec = ts_ns % 1_000_000_000
            sources.append(
                {
                    "sensor_token": channel_to_sensor_token[channel],
                    "idx_begin": idx_cursor,
                    "length": length,
                    "stamp": {"sec": int(sec), "nanosec": int(nanosec)},
                }
            )
            idx_cursor += length

        merged = (
            np.concatenate(all_points, axis=0)
            if all_points
            else np.empty((0, LIDAR_CONCAT_NUM_POINT_FEATURES), dtype=np.float32)
        )
        bin_path = data_dir / LIDAR_CONCAT_CHANNEL / f"{frame_idx:06d}.pcd.bin"
        _write_pcd_bin(bin_path, merged)

        info_rel = f"data/LIDAR_CONCAT_INFO/{frame_idx:06d}.json"
        info_path = data_dir / "LIDAR_CONCAT_INFO" / f"{frame_idx:06d}.json"
        with open(info_path, "w") as f:
            json.dump({"sources": sources}, f)

        return info_rel


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _indexed_file(directory: Path, index: int, pattern: str) -> Optional[Path]:
    """Return the *index*-th file (sorted) matching *pattern* under *directory*, or None."""
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern))
    return files[index] if 0 <= index < len(files) else None


def _frame_timestamp_ns(
    seq_path: Path, lidar_channels: List[str], frame_index: int
) -> Optional[int]:
    """Read the nanosecond timestamp from the lidar CSV filename for a given frame index."""
    for channel in lidar_channels:
        csv_path = _indexed_file(seq_path / "lidar" / channel, frame_index, "*.csv")
        if csv_path is not None:
            try:
                return int(csv_path.stem)
            except ValueError:
                pass
    return None


def _load_csv(csv_path: Path) -> Tuple[np.ndarray, int]:
    """Load a Kognic point-cloud CSV and return (points, timestamp_ns).

    CSV columns: ts_gps, x, y, z, intensity.
    Returned *points* is float32 with columns [x, y, z, intensity, ring_idx=0],
    matching the LIDAR_CONCAT .pcd.bin layout (LIDAR_CONCAT_NUM_POINT_FEATURES=5).
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ts_ns = int(data[0, 0]) if len(data) > 0 else 0

    n = len(data)
    points = np.zeros((n, LIDAR_CONCAT_NUM_POINT_FEATURES), dtype=np.float32)
    if n > 0:
        points[:, 0] = data[:, 1]  # x
        points[:, 1] = data[:, 2]  # y
        points[:, 2] = data[:, 3]  # z
        points[:, 3] = data[:, 4]  # intensity
        # column 4 (ring_idx) stays 0

    return points, ts_ns


def _write_pcd_bin(path: Path, points: np.ndarray) -> None:
    """Write *points* (float32, shape [N, LIDAR_CONCAT_NUM_POINT_FEATURES]) as a raw .pcd.bin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32).tofile(path)


def _connect_chain(token_list: List[str], table: TableHandler) -> None:
    """Set next/prev on consecutive tokens in *table*."""
    for i in range(1, len(token_list)):
        table.update_record_from_token(token_list[i - 1], next=token_list[i])
        table.update_record_from_token(token_list[i], prev=token_list[i - 1])
