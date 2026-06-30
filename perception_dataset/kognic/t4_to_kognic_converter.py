from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

from t4_devkit import Tier4

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import LIDAR_CONCAT_CHANNEL
from perception_dataset.kognic.utils import (
    extract_calibration,
    extract_ego_poses,
    iter_scene_pairs,
)
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.pointcloud import (
    copy_file,
    extract_pointclouds,
)

logger = configure_logger(modname=__name__)


class T4ToKognicConverter(AbstractConverter[None]):
    """Convert T4 data (annotated or non-annotated) to the Kognic IO staging layout.

    Only sensor data, calibration, and ego poses are exported;
    annotation tables,  if present, are ignored.

    output layout:

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
        workers_number: int = 32,
        drop_camera_token_not_found: bool = False,
    ):
        super().__init__(input_base, output_base)
        self._camera_channels: List[str] = [cam["channel"] for cam in camera_sensors]
        self._workers_number = workers_number
        self._drop_camera_token_not_found = drop_camera_token_not_found

    def convert(self) -> None:
        start = time.time()

        for seq_path, out_dir in iter_scene_pairs(Path(self._input_base), Path(self._output_base)):
            logger.info(f"[BEGIN] {seq_path} -> {out_dir}")
            self._convert_one_scene(seq_path, out_dir)
            logger.info(f"[DONE]  {seq_path} -> {out_dir}")

        logger.info(f"Elapsed: {time.time() - start:.1f}s")

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
        logger.info(f"Selected {len(self._frame_records)} frames")

        if not self._has_lidar_concat_info and LIDAR_CONCAT_CHANNEL in self._lidar_channels:
            logger.warning(
                "LIDAR_CONCAT_INFO is missing. Exporting fused LIDAR_CONCAT "
                "as a single Kognic LiDAR stream instead of per-source LiDARs."
            )

        calibration = extract_calibration(
            channel_to_token=self._channel_to_token,
            calib_by_sensor_token=self._calib_by_sensor_token,
            camera_channels=self._camera_channels,
            lidar_channels=self._lidar_channels,
            sample_data_by_channel=self._sample_data_by_channel,
            seq_path=seq_path,
        )
        with open(out_dir / "calibration.json", "w") as f:
            json.dump({k: v.model_dump() for k, v in calibration.items()}, f, indent=2)
        logger.info(f"Calibration saved ({len(calibration)} sensors)")

        ego_poses = extract_ego_poses(
            frame_records=self._frame_records,
            ego_pose_by_token=self._ego_pose_by_token,
            camera_channels=self._camera_channels,
        )
        with open(out_dir / "ego_poses.json", "w") as f:
            json.dump({k: v.model_dump() for k, v in ego_poses.items()}, f, indent=2)
        logger.info(f"Ego poses saved ({len(ego_poses)} frames)")

        pending_copies: List[Tuple[Path, Path]] = []
        for camera_channel in self._camera_channels:
            pending_copies.extend(self._collect_image_copies(seq_path, out_dir, camera_channel))

        with ThreadPoolExecutor(max_workers=self._workers_number) as executor:
            list(executor.map(lambda args: copy_file(*args), pending_copies))

        for lidar_channel in self._lidar_channels:
            extract_pointclouds(
                seq_path=seq_path,
                out_dir=out_dir,
                lidar_channel=lidar_channel,
                frame_records=self._frame_records,
                channel_to_token=self._channel_to_token,
            )

    def _build_lookup_maps(self, seq_path: Path) -> None:
        t4 = Tier4(data_root=str(seq_path), verbose=False)

        sensors = t4.get_table("sensor")
        self._sensors = sensors
        self._token_to_channel = {s.token: s.channel for s in sensors}
        self._channel_to_token = {s.channel: s.token for s in sensors}

        calib_sensors = t4.get_table("calibrated_sensor")
        self._calib_by_token = {c.token: c for c in calib_sensors}
        self._calib_by_sensor_token = {c.sensor_token: c for c in calib_sensors}

        samples = t4.get_table("sample")
        self._samples = sorted(samples, key=lambda s: s.timestamp)

        self._sample_data_by_channel: Dict[str, list] = {}
        self._sample_data_by_channel_and_frame_id: Dict[str, Dict[str, object]] = {}
        for sd in t4.get_table("sample_data"):
            # Tier4 resolves the channel (sample_data -> calibrated_sensor ->
            # sensor) for us, so use it directly instead of re-deriving it.
            channel = sd.channel
            if not channel:
                continue
            self._sample_data_by_channel.setdefault(channel, []).append(sd)
            frame_id = Path(sd.filename).stem.split(".")[0]
            self._sample_data_by_channel_and_frame_id.setdefault(channel, {})[frame_id] = sd

        for channel in self._sample_data_by_channel:
            self._sample_data_by_channel[channel] = sorted(
                self._sample_data_by_channel[channel],
                key=lambda sample_data_record: sample_data_record.timestamp,
            )

        self._ego_pose_by_token = {ep.token: ep for ep in t4.get_table("ego_pose")}

    def _discover_lidar_channels(self) -> List[str]:
        if not self._has_lidar_concat_info:
            if LIDAR_CONCAT_CHANNEL in self._channel_to_token:
                return [LIDAR_CONCAT_CHANNEL]
            return []

        return [
            sensor.channel
            for sensor in self._sensors
            if sensor.modality.value == "lidar" and sensor.channel != LIDAR_CONCAT_CHANNEL
        ]

    def _has_existing_channel_file(self, seq_path: Path, channel: str) -> bool:
        return any(
            (seq_path / sample_data.filename).exists()
            for sample_data in self._sample_data_by_channel.get(channel, [])
        )

    def _build_frame_records(self) -> List[Dict[str, object]]:
        """Build one output frame per record of the high-frequency anchor stream.

        Every ``sample_data.json`` record of the anchor channel is exported —
        key frames and intermediate sweeps alike. Annotation frequency is
        controlled at upload time via ``target_hz``.
        """
        anchor_channel = self._select_anchor_channel()
        anchor_records = self._sample_data_by_channel.get(anchor_channel, [])
        frame_records: List[Dict[str, object]] = []
        for anchor_record in anchor_records:
            frame_id = Path(anchor_record.filename).stem.split(".")[0]
            frame_record: Dict[str, object] = {}

            for channel in self._channels_for_frame_records():
                sample_data = self._sample_data_by_channel_and_frame_id.get(channel, {}).get(
                    frame_id
                )
                if sample_data is not None:
                    frame_record[channel] = sample_data

            if frame_record:
                frame_records.append(frame_record)

        return frame_records

    def _select_anchor_channel(self) -> str:
        if self._sample_data_by_channel.get(LIDAR_CONCAT_CHANNEL):
            return LIDAR_CONCAT_CHANNEL

        for camera_channel in self._camera_channels:
            if self._sample_data_by_channel.get(camera_channel):
                return camera_channel

        raise ValueError(
            f"No anchor channel with sample_data found "
            f"({LIDAR_CONCAT_CHANNEL} or any of {self._camera_channels})"
        )

    def _channels_for_frame_records(self) -> List[str]:
        return [LIDAR_CONCAT_CHANNEL, *self._camera_channels]

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
                    logger.warning(f"Camera {camera_channel} missing for selected frame; skipping")
                continue

            timestamp_ns = int(sample_data.timestamp) * 1000
            src = seq_path / sample_data.filename
            dst = camera_dir / f"{timestamp_ns}.jpg"
            if src.exists() and not dst.exists():
                copies.append((src, dst))

        logger.info(f"{camera_channel}: {len(copies)} image copies queued")
        return copies
