"""Convert T4 sensor data to the Kognic staging layout."""

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import time
from typing import Dict, Generator, List, Optional, Set, Tuple
import uuid

import kognic.io.model as KognicModel
from kognic.io.model.ego.imu_data import IMUData
from kognic.io.model.scene.lidars_and_cameras_sequence.frame import (
    Frame as LidarsAndCamerasSequenceFrame,
)
from kognic.io.model.scene.metadata.metadata import FrameMetaData, MetaData
from kognic.io.model.scene.resources.image import ImageMetadata
import numpy as np
from t4_devkit import Tier4

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import (
    IMU_EXTRAPOLATE_S,
    IMU_TARGET_HZ,
    LIDAR_CONCAT_CHANNEL,
    PREFERRED_CAMERA_SENSORS,
    PREFERRED_LIDAR_SENSORS,
)
from perception_dataset.kognic.sequence_artifact import (
    PENDING_CALIBRATION_ID,
    SEQUENCE_ARTIFACT_FILENAME,
    save_sequence_artifact,
)
from perception_dataset.kognic.utils import (
    extract_calibration,
    extract_ego_poses,
    iter_scene_pairs,
    read_image_dims,
)
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.misc import MAX_ANNOTATION_HZ, validate_annotation_hz
from perception_dataset.utils.pointcloud import (
    copy_file,
    extract_pointclouds,
)

logger = configure_logger(modname=__name__)

_REPORT_FILENAME = "conversion_report.tsv"
_REPORT_FIELDS = (
    "scene",
    "status",
    "sensor_type",
    "sensor",
    "frame",
    "timestamp_ns",
    "details",
)


def _sort_key(path: Path) -> Tuple[int, str]:
    """Build a deterministic key for ordering converted sensor files.

    Numeric filename stems are zero-padded and sorted before non-numeric
    stems, preserving chronological order for timestamp-named resources while
    retaining deterministic behavior for unexpected filenames.

    Args:
        path (Path): Sensor resource path to order.

    Returns:
        Tuple[int, str]: Numeric-first sort group and normalized filename stem.
    """
    try:
        return (0, f"{int(path.stem):030d}")
    except ValueError:
        return (1, path.stem)


def _sensor_sort_key(sensor_name: str, preferred_order: List[str]) -> Tuple[int, str]:
    """Build a deterministic key that prioritizes known sensor channels.

    Args:
        sensor_name (str): Sensor channel to rank.
        preferred_order (List[str]): Channels ordered from highest to lowest
            anchor preference.

    Returns:
        Tuple[int, str]: Preference index and sensor name. Unknown sensors sort
            after preferred sensors in lexical order.
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
    """Require every converted sensor to contribute one file per frame.

    Args:
        sequence_path (Path): Scene staging directory used in error messages.
        sensor_files (Dict[str, List[Path]]): Ordered resource files keyed by
            sensor channel.
        anchor_sensor (str): Channel defining the expected frame count.
        expected_count (int): Number of resources in the anchor channel.

    Returns:
        None

    Raises:
        ValueError: If any sensor has a different number of resources than the
            anchor sensor.
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
    """Require converted sensor timestamps to track the anchor frame sequence.

    Each sensor resource is compared with the anchor resource at the same
    frame index. A deviation greater than 90 percent of the median anchor
    interval indicates stale or shifted staging data.

    Args:
        sequence_path (Path): Scene staging directory used in error messages.
        sensor_files (Dict[str, List[Path]]): Ordered resource files keyed by
            sensor channel.
        anchor_files (List[Path]): Ordered resources for the channel defining
            the frame timeline.

    Returns:
        None

    Raises:
        ValueError: If anchor timestamps are not strictly increasing or a
            sensor timestamp is too far from its corresponding anchor frame.
    """
    try:
        anchor_ts = [int(path.stem) for path in anchor_files]
    except ValueError:
        return

    intervals = sorted(b - a for a, b in zip(anchor_ts, anchor_ts[1:]))
    if not intervals:
        return

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


class T4ToKognicConverter(AbstractConverter[None]):
    """Convert T4 data (annotated or non-annotated) to the Kognic IO staging layout.

    Only sensor data, calibration, and ego poses are exported;
    annotation tables,  if present, are ignored.

    output layout:

        <output_base>/<input_item_name>/
            calibration.json
            ego_poses.json
            keyframes.json
            lidars_and_cameras_sequence.json
            cameras/<camera_name>/<timestamp_ns>.jpg
            lidar/<lidar_name>/<timestamp_ns>.csv

    ``keyframes.json`` holds the staging frame indices of the keyframes; the
    uploader requires this file and marks exactly those frames ``annotate=True``.
    For annotated datasets (``annotated=True``), source T4 keyframes remain
    annotatable even when they contain no objects. Non-annotated datasets have
    no source keyframes, so their keyframes are selected by sample index at
    ``annotation_hz``, matching the non-annotated T4 -> Deepen converter.
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        camera_sensors: list,
        workers_number: int = 32,
        annotated: bool = True,
        annotation_hz: int = 10,
        include_imu_data: bool = True,
        generate_tsv_report: bool = False,
    ):
        """Initialize the converter.

        Args:
            input_base (str): Input T4 dataset directory.
            output_base (str): Destination staging directory.
            camera_sensors (list): Camera configuration records.
            workers_number (int): Number of image-copy worker threads.
            annotated (bool): Whether the source carries T4 annotations.
            annotation_hz (int): Keyframe frequency for non-annotated data, in
                ``1..10``.
            include_imu_data (bool): Whether to include interpolated IMU data
                in the validated sequence artifact.
            generate_tsv_report (bool): Write ``conversion_report.tsv`` in
                ``output_base`` with scene outcomes and missing sensor frames.

        Raises:
            ValueError: If ``annotation_hz`` is outside ``1..10``.
        """
        super().__init__(input_base, output_base)
        self._camera_channels: List[str] = [cam["channel"] for cam in camera_sensors]
        self._workers_number = workers_number
        self._annotated = annotated
        self._annotation_hz = validate_annotation_hz(annotation_hz)
        self._include_imu_data = include_imu_data
        self._generate_tsv_report = generate_tsv_report
        self._report_rows: List[Dict[str, str]] = []
        # Cache one blank black image per camera, sized to that camera's frames,
        # reused for every frame that is missing an image (see
        # ``_write_blank_image``).
        self._blank_image_cache: Dict[str, object] = {}

    def convert(self) -> None:
        """Convert every discovered T4 sequence.

        Returns:
            None
        """
        start = time.time()

        self._report_rows = []
        failed_scenes: List[Tuple[Path, Exception]] = []
        try:
            for seq_path, out_dir in iter_scene_pairs(
                Path(self._input_base), Path(self._output_base)
            ):
                report_scene = self._report_scene_path(seq_path)
                logger.info(f"[BEGIN] {seq_path} -> {out_dir}")
                try:
                    self._convert_one_scene(seq_path, out_dir)
                except Exception as exc:
                    self._append_report_row(
                        scene=report_scene,
                        status="failed",
                        details=f"{type(exc).__name__}: {exc}",
                    )
                    if not self._generate_tsv_report:
                        raise
                    failed_scenes.append((seq_path, exc))
                    logger.exception(f"[FAILED] {seq_path} -> {out_dir}")
                else:
                    self._append_report_row(scene=report_scene, status="successful")
                    logger.info(f"[DONE]  {seq_path} -> {out_dir}")

            if failed_scenes:
                failed_names = ", ".join(
                    self._report_scene_path(path) for path, _ in failed_scenes
                )
                raise RuntimeError(
                    f"{len(failed_scenes)} scene conversion(s) failed: {failed_names}. "
                    f"See {Path(self._output_base) / _REPORT_FILENAME} for details."
                ) from failed_scenes[0][1]
        finally:
            if self._generate_tsv_report:
                self._write_tsv_report()
            logger.info(f"Elapsed: {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Scene conversion
    # ------------------------------------------------------------------

    def _convert_one_scene(self, input_dir: Path | str, output_dir: Path | str) -> None:
        """Convert one T4 sequence to a Kognic staging directory.

        Args:
            input_dir (Path | str): Source T4 sequence root.
            output_dir (Path | str): Destination staging directory.

        Returns:
            None
        """
        seq_path = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sequence construction pairs sensor files by sorted position, so any
        # file left over from a previous generation would shift that pairing.
        for stale_dir in (out_dir / "cameras", out_dir / "lidar"):
            shutil.rmtree(stale_dir, ignore_errors=True)
        (out_dir / SEQUENCE_ARTIFACT_FILENAME).unlink(missing_ok=True)

        self._build_lookup_maps(seq_path)
        self._has_lidar_concat_info = any(
            sample_data.info_filename
            for sample_data in self._sample_data_by_channel.get(LIDAR_CONCAT_CHANNEL, [])
        )
        self._lidar_channels = self._discover_lidar_channels()
        self._frame_records = self._build_frame_records()
        self._record_missing_sensor_frames(seq_path)
        logger.info(f"Selected {len(self._frame_records)} frames")
        self._write_keyframes(out_dir)

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

        self._write_sequence_artifact(out_dir, ego_poses)

    # ------------------------------------------------------------------
    # Conversion report
    # ------------------------------------------------------------------

    def _append_report_row(
        self,
        *,
        scene: str,
        status: str,
        sensor_type: str = "",
        sensor: str = "",
        frame: str = "",
        timestamp_ns: int | str = "",
        details: str = "",
    ) -> None:
        """Append one normalized conversion-report row when reporting is enabled."""
        if not self._generate_tsv_report:
            return
        self._report_rows.append(
            {
                "scene": scene,
                "status": status,
                "sensor_type": sensor_type,
                "sensor": sensor,
                "frame": frame,
                "timestamp_ns": str(timestamp_ns),
                "details": details,
            }
        )

    def _write_tsv_report(self) -> None:
        """Write the accumulated report to ``output_base/conversion_report.tsv``."""
        output_base = Path(self._output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        report_path = output_base / _REPORT_FILENAME
        with open(report_path, "w", newline="", encoding="utf-8") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=_REPORT_FIELDS, dialect="excel-tab")
            writer.writeheader()
            writer.writerows(self._report_rows)
        logger.info(f"Conversion report saved to {report_path}")

    def _report_scene_path(self, seq_path: Path) -> str:
        """Return the full nested scene path relative to ``input_base``."""
        input_base = Path(self._input_base).resolve()
        resolved_scene = Path(seq_path).resolve()
        try:
            relative_scene = resolved_scene.relative_to(input_base)
        except ValueError:
            return str(resolved_scene)

        # When input_base points directly at a sequence root, retain its name
        # instead of reporting the unhelpful relative path ".".
        if relative_scene == Path("."):
            return resolved_scene.name
        return relative_scene.as_posix()

    def _record_missing_sensor_frames(self, seq_path: Path) -> None:
        """Record camera and LiDAR gaps, including gaps replaced by blank output files."""
        if not self._generate_tsv_report:
            return

        scene = self._report_scene_path(seq_path)
        camera_actions = {
            channel: self._missing_camera_action(seq_path, channel)
            for channel in self._camera_channels
        }
        for frame_index, frame_record in enumerate(self._frame_records):
            frame, timestamp_ns = self._frame_identity(frame_index, frame_record)

            for camera_channel in self._camera_channels:
                sample_data = frame_record.get(camera_channel)
                if sample_data is None:
                    self._append_report_row(
                        scene=scene,
                        status="missing_camera_frame",
                        sensor_type="camera",
                        sensor=camera_channel,
                        frame=frame,
                        timestamp_ns=timestamp_ns,
                        details=f"sample_data record is missing; "
                        f"{camera_actions[camera_channel]}",
                    )
                    continue

                image_path = seq_path / sample_data.filename
                if not image_path.exists():
                    self._append_report_row(
                        scene=scene,
                        status="missing_camera_frame",
                        sensor_type="camera",
                        sensor=camera_channel,
                        frame=frame,
                        timestamp_ns=timestamp_ns,
                        details=f"source file is missing: {sample_data.filename}; "
                        f"{camera_actions[camera_channel]}",
                    )

            for lidar_channel in self._lidar_channels:
                reason = self._missing_lidar_reason(seq_path, frame_record, lidar_channel)
                if reason:
                    self._append_report_row(
                        scene=scene,
                        status="missing_lidar_frame",
                        sensor_type="lidar",
                        sensor=lidar_channel,
                        frame=frame,
                        timestamp_ns=timestamp_ns,
                        details=reason,
                    )

    def _missing_camera_action(self, seq_path: Path, camera_channel: str) -> str:
        """Describe what conversion does when this camera has a missing frame."""
        if camera_channel not in self._channel_to_token:
            return "configured camera is absent from the dataset; channel skipped"
        if not self._has_existing_channel_file(seq_path, camera_channel):
            return "camera has no existing source files; channel skipped"
        return "blank image written"

    def _frame_identity(
        self, frame_index: int, frame_record: Dict[str, object]
    ) -> Tuple[str, int]:
        """Return a stable source frame identifier and representative timestamp."""
        anchor = frame_record.get(self._anchor_channel)
        if anchor is not None:
            frame = Path(anchor.filename).stem.split(".")[0]
        else:
            frame = str(frame_index)
        return frame, self._frame_timestamp_ns(frame_record)

    def _missing_lidar_reason(
        self, seq_path: Path, frame_record: Dict[str, object], lidar_channel: str
    ) -> str | None:
        """Describe a missing LiDAR contribution for one output frame, if any."""
        concat_sample_data = frame_record.get(LIDAR_CONCAT_CHANNEL)
        if concat_sample_data is None:
            return f"{LIDAR_CONCAT_CHANNEL} sample_data record is missing"

        bin_path = seq_path / concat_sample_data.filename
        if not bin_path.exists():
            return f"source point cloud is missing: {concat_sample_data.filename}"
        if bin_path.stat().st_size == 0:
            return "source point cloud contains zero points"

        if lidar_channel == LIDAR_CONCAT_CHANNEL:
            return None

        info_filename = concat_sample_data.info_filename
        if not info_filename:
            return "LIDAR_CONCAT_INFO filename is missing"
        info_path = seq_path / info_filename
        if not info_path.exists():
            return f"LIDAR_CONCAT_INFO file is missing: {info_filename}"

        try:
            with open(info_path, encoding="utf-8") as info_file:
                info = json.load(info_file)
        except (OSError, ValueError) as exc:
            return f"LIDAR_CONCAT_INFO cannot be read: {exc}"

        sensor_token = self._channel_to_token.get(lidar_channel)
        source = next(
            (
                source
                for source in info.get("sources", [])
                if source.get("sensor_token") == sensor_token
            ),
            None,
        )
        if source is None:
            return "sensor contribution is absent from LIDAR_CONCAT_INFO; blank frame written"
        if int(source.get("length", 0)) == 0:
            return "sensor contributed zero points; header-only point cloud written"
        return None

    def _build_lookup_maps(self, seq_path: Path) -> None:
        """Load T4 tables and build lookup mappings used during conversion.

        Args:
            seq_path (Path): Source T4 sequence root.

        Returns:
            None
        """
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
        """Select lidar channels that can be exported.

        Returns:
            List[str]: Per-sensor lidar channels when concat metadata exists,
                otherwise the fused concat channel when available.
        """
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
        """Check whether a channel references an existing data file.

        Args:
            seq_path (Path): Source T4 sequence root.
            channel (str): Sensor channel to inspect.

        Returns:
            bool: ``True`` when at least one referenced file exists.
        """
        return any(
            (seq_path / sample_data.filename).exists()
            for sample_data in self._sample_data_by_channel.get(channel, [])
        )

    def _write_keyframes(self, out_dir: Path) -> None:
        """Write the staging frame indices of the keyframes to ``keyframes.json``.

        The uploader marks exactly those frames ``annotate=True``;
        ``frame_count`` lets it detect a stale file after the staging data
        changed.

        Annotated datasets: source T4 keyframes remain annotatable, including
        keyframes that contain no objects.

        Non-annotated datasets: there are no annotations to key off, so
        keyframes are selected by sample index at ``annotation_hz``, with the
        same logic as the non-annotated T4 -> Deepen converter (every
        ``int(10 / annotation_hz)``-th sample).

        Args:
            out_dir (Path): Destination staging directory.

        Returns:
            None
        """
        if self._annotated:
            keyframe_indices = [
                idx
                for idx, frame_record in enumerate(self._frame_records)
                if getattr(frame_record.get(self._anchor_channel), "is_key_frame", False)
            ]
        else:
            step = int(MAX_ANNOTATION_HZ / self._annotation_hz)
            selected_samples = {
                sample.token
                for sample_index, sample in enumerate(self._samples)
                if sample_index % step == 0
            }
            keyframe_indices = [
                idx
                for idx, frame_record in enumerate(self._frame_records)
                if getattr(frame_record.get(self._anchor_channel), "sample_token", None)
                in selected_samples
            ]
        with open(out_dir / "keyframes.json", "w") as f:
            json.dump(
                {"frame_count": len(self._frame_records), "keyframe_indices": keyframe_indices},
                f,
            )
        logger.info(
            f"keyframes.json: {len(keyframe_indices)} keyframes over "
            f"{len(self._frame_records)} frames"
        )

    @staticmethod
    def _collect_sensor_files(
        sequence_path: Path, root_name: str, suffix: str
    ) -> Dict[str, List[Path]]:
        """Collect and order converted files by sensor channel.

        Args:
            sequence_path (Path): Scene staging directory.
            root_name (str): Sensor directory name, such as ``lidar`` or
                ``cameras``.
            suffix (str): File extension to include, such as ``.csv`` or
                ``.jpg``.

        Returns:
            Dict[str, List[Path]]: Timestamp-ordered files keyed by sensor
                channel. Returns an empty dictionary when the sensor root is
                absent or contains no matching files.
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

    def _iterate_frames(
        self, sequence_path: Path
    ) -> Generator[Tuple[str, int, Dict[str, Path]], None, None]:
        """Yield synchronized converted resources in anchor-frame order.

        A preferred LiDAR channel defines the frame timeline when available;
        otherwise a preferred camera channel is used. Before yielding, the
        method verifies that every sensor has the same frame count and that its
        timestamps remain aligned with the anchor.

        Args:
            sequence_path (Path): Scene staging directory containing converted
                ``lidar`` and/or ``cameras`` subdirectories.

        Yields:
            Generator[Tuple[str, int, Dict[str, Path]], None, None]: Frame ID,
                anchor timestamp in nanoseconds, and one resource path per
                sensor channel.

        Raises:
            FileNotFoundError: If no converted LiDAR CSV or camera JPEG exists.
            ValueError: If sensor counts differ, timestamps are misaligned, or
                an anchor filename is not a numeric timestamp.
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

        combined_files = {**lidar_files, **camera_files}
        all_files = {
            name: combined_files[name]
            for name in [
                anchor_sensors[0],
                *sorted(set(combined_files) - {anchor_sensors[0]}),
            ]
        }
        _validate_sensor_file_counts(
            sequence_path,
            all_files,
            anchor_sensors[0],
            len(anchor_files),
        )
        _validate_sensor_timestamps(sequence_path, all_files, anchor_files)

        for frame_idx, anchor_file in enumerate(anchor_files):
            yield (
                str(frame_idx),
                int(anchor_file.stem),
                {name: files[frame_idx] for name, files in all_files.items()},
            )

    @staticmethod
    def _load_keyframe_indices(sequence_path: Path, frame_count: int) -> List[int]:
        """Load and validate converted T4 keyframe positions.

        Args:
            sequence_path (Path): Scene staging directory containing
                ``keyframes.json``.
            frame_count (int): Number of frames in the converted sensor data.

        Returns:
            List[int]: Frame indices that must be marked ``annotate=True``.

        Raises:
            FileNotFoundError: If ``keyframes.json`` does not exist.
            ValueError: If its recorded frame count differs from
                ``frame_count`` or any keyframe index is invalid.
        """
        keyframes_path = sequence_path / "keyframes.json"
        if not keyframes_path.exists():
            raise FileNotFoundError(f"Required keyframe metadata is missing: {keyframes_path}")

        with open(keyframes_path) as f:
            data = json.load(f)
        if data.get("frame_count") != frame_count:
            raise ValueError(
                f"{keyframes_path} was generated for {data.get('frame_count')} frames but "
                f"the current staging data has {frame_count}."
            )

        keyframe_indices = data.get("keyframe_indices")
        if not isinstance(keyframe_indices, list) or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < frame_count
            for index in keyframe_indices
        ):
            raise ValueError(
                f"{keyframes_path} contains invalid keyframe_indices for {frame_count} frames."
            )
        return keyframe_indices

    def _build_sequence_frames(
        self,
        sequence_path: Path,
        ego_poses: Dict[str, KognicModel.EgoVehiclePose],
    ) -> List[LidarsAndCamerasSequenceFrame]:
        """Build validated Kognic frames from converted sensor resources.

        Args:
            sequence_path (Path): Scene staging directory containing sensor
                files and ``keyframes.json``.
            ego_poses (Dict[str, KognicModel.EgoVehiclePose]): Relative ego
                poses keyed by converted frame ID.

        Returns:
            List[LidarsAndCamerasSequenceFrame]: Ordered frames containing
                sensor resources, timestamps, poses, and annotation flags.

        Raises:
            FileNotFoundError: If required sensor or keyframe data is missing.
            ValueError: If sensor synchronization or keyframe metadata is
                invalid.
        """
        frames = []
        frame_records = list(self._iterate_frames(sequence_path))
        annotate_indices = set(
            self._load_keyframe_indices(sequence_path, len(frame_records))
        )
        reference_timestamp: Optional[int] = None

        for frame_idx, (frame_id, timestamp_ns, sensor_files) in enumerate(frame_records):
            if reference_timestamp is None:
                reference_timestamp = timestamp_ns
            frames.append(
                LidarsAndCamerasSequenceFrame(
                    frame_id=frame_id,
                    relative_timestamp=int((timestamp_ns - reference_timestamp) / 1e6),
                    unix_timestamp=timestamp_ns,
                    ego_vehicle_pose=ego_poses.get(frame_id),
                    point_clouds=[
                        KognicModel.PointCloud(sensor_name=name, filename=str(path))
                        for name, path in sensor_files.items()
                        if path.suffix == ".csv"
                    ],
                    images=[
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
                    ],
                    metadata=FrameMetaData(annotate=frame_idx in annotate_indices),
                )
            )
        return frames

    def _build_imu_data(
        self,
        sequence_path: Path,
        ego_poses: Dict[str, KognicModel.EgoVehiclePose],
    ) -> List[IMUData]:
        """Interpolate converted ego poses into dense Kognic IMU samples.

        Position is linearly interpolated and rotation uses spherical linear
        interpolation. Samples outside the sparse pose interval use the same
        linear and rotational trends for bounded extrapolation.

        Args:
            sequence_path (Path): Scene staging directory whose sensor frames
                define the timestamps to interpolate.
            ego_poses (Dict[str, KognicModel.EgoVehiclePose]): Sparse relative
                ego poses keyed by converted frame ID.

        Returns:
            List[IMUData]: Dense IMU samples, or an empty list when IMU output
                is disabled, fewer than two poses are available, or SciPy is
                unavailable.

        Raises:
            ValueError: If converted sensor frames fail synchronization
                validation.
        """
        if not self._include_imu_data or not ego_poses:
            return []

        try:
            from scipy.interpolate import interp1d
            from scipy.spatial.transform import Rotation, Slerp
        except ModuleNotFoundError:
            logger.warning("scipy is not installed; skipping optional IMU data generation")
            return []

        sparse = [
            (timestamp_ns, ego_poses[frame_id])
            for frame_id, timestamp_ns, _ in self._iterate_frames(sequence_path)
            if frame_id in ego_poses
        ]
        if len(sparse) < 2:
            return []

        sparse.sort(key=lambda item: item[0])
        ts_sparse = np.array([item[0] for item in sparse], dtype=np.float64)
        pos_sparse = np.array(
            [
                [item[1].position.x, item[1].position.y, item[1].position.z]
                for item in sparse
            ],
            dtype=np.float64,
        )
        rot_sparse = Rotation.concatenate(
            [
                Rotation.from_quat(
                    [
                        item[1].rotation.x,
                        item[1].rotation.y,
                        item[1].rotation.z,
                        item[1].rotation.w,
                    ]
                )
                for item in sparse
            ]
        )

        dense_dt_ns = int(1e9 / IMU_TARGET_HZ)
        extrap_ns = IMU_EXTRAPOLATE_S * 1e9
        ts_dense = np.arange(
            ts_sparse[0] - extrap_ns,
            ts_sparse[-1] + extrap_ns + dense_dt_ns,
            dense_dt_ns,
        )
        pos_dense = interp1d(
            ts_sparse,
            pos_sparse,
            axis=0,
            fill_value="extrapolate",
            bounds_error=False,
        )(ts_dense)
        slerp = Slerp(ts_sparse, rot_sparse)
        rot_dense_list = []
        for timestamp in ts_dense:
            if timestamp <= ts_sparse[0]:
                delta = ts_sparse[1] - ts_sparse[0]
                difference = rot_sparse[0].inv() * rot_sparse[1]
                rot_dense_list.append(
                    rot_sparse[0]
                    * (difference.inv() ** ((ts_sparse[0] - timestamp) / delta))
                )
            elif timestamp >= ts_sparse[-1]:
                delta = ts_sparse[-1] - ts_sparse[-2]
                difference = rot_sparse[-2].inv() * rot_sparse[-1]
                rot_dense_list.append(
                    rot_sparse[-1]
                    * (difference ** ((timestamp - ts_sparse[-1]) / delta))
                )
            else:
                rot_dense_list.append(slerp(timestamp))

        quaternions = Rotation.concatenate(rot_dense_list).as_quat()
        return [
            IMUData(
                timestamp=float(ts_dense[index]),
                position=KognicModel.Position(
                    x=float(pos_dense[index, 0]),
                    y=float(pos_dense[index, 1]),
                    z=float(pos_dense[index, 2]),
                ),
                rotation_quaternion=KognicModel.RotationQuaternion(
                    w=float(quaternions[index, 3]),
                    x=float(quaternions[index, 0]),
                    y=float(quaternions[index, 1]),
                    z=float(quaternions[index, 2]),
                ),
            )
            for index in range(len(ts_dense))
        ]

    def _write_sequence_artifact(
        self,
        sequence_path: Path,
        ego_poses: Dict[str, KognicModel.EgoVehiclePose],
    ) -> None:
        """Build, validate, and persist the complete Kognic sequence artifact.

        Args:
            sequence_path (Path): Scene staging directory that receives
                ``lidars_and_cameras_sequence.json``.
            ego_poses (Dict[str, KognicModel.EgoVehiclePose]): Relative ego
                poses keyed by converted frame ID.

        Returns:
            None

        Raises:
            FileNotFoundError: If required converted resources or keyframe
                metadata are missing.
            ValueError: If sensor synchronization, keyframe metadata, or
                artifact resource paths are invalid.
        """
        frames = self._build_sequence_frames(sequence_path, ego_poses)
        imu_data = self._build_imu_data(sequence_path, ego_poses)
        sequence = KognicModel.LidarsAndCamerasSequence(
            external_id=sequence_path.name,
            frames=frames,
            calibration_id=PENDING_CALIBRATION_ID,
            imu_data=imu_data,
            metadata=MetaData(
                source_filename=sequence_path.name,
                dataset_id=sequence_path.name,
                inner_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, sequence_path.name)),
            ),
        )
        save_sequence_artifact(sequence_path, sequence)
        logger.info(
            f"{SEQUENCE_ARTIFACT_FILENAME}: {len(frames)} frames, "
            f"{len(imu_data)} IMU samples"
        )

    def _build_frame_records(self) -> List[Dict[str, object]]:
        """Build one output frame per record of the high-frequency anchor stream.

        Every ``sample_data.json`` record of the anchor channel is exported —
        key frames and intermediate sweeps alike. The keyframes among them are
        recorded in ``keyframes.json`` and become the annotatable frames at
        upload time.

        Returns:
            List[Dict[str, object]]: Ordered channel-to-sample-data mappings.
        """
        anchor_channel = self._anchor_channel = self._select_anchor_channel()
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
        """Select the high-frequency stream that defines output frames.

        Returns:
            str: Fused lidar channel when available, otherwise the first
                configured camera containing sample data.

        Raises:
            ValueError: If no usable anchor channel exists.
        """
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
        """Get channels included in synchronized frame records.

        Returns:
            List[str]: Fused lidar followed by configured camera channels.
        """
        return [LIDAR_CONCAT_CHANNEL, *self._camera_channels]

    # ------------------------------------------------------------------
    # Sensor data
    # ------------------------------------------------------------------

    def _collect_image_copies(
        self, seq_path: Path, out_dir: Path, camera_channel: str
    ) -> List[Tuple[Path, Path]]:
        """Prepare image-copy operations for a camera channel.

        Args:
            seq_path (Path): Source T4 sequence root.
            out_dir (Path): Destination staging directory.
            camera_channel (str): Camera channel to export.

        Returns:
            List[Tuple[Path, Path]]: Source and destination paths for images
                that should be copied.
        """
        if camera_channel not in self._channel_to_token:
            logger.warning(f"Camera {camera_channel} not found in {seq_path}; skipping")
            return []
        if not self._has_existing_channel_file(seq_path, camera_channel):
            logger.warning(f"Camera {camera_channel} has no files in {seq_path}; skipping")
            return []

        camera_dir = out_dir / "cameras" / camera_channel
        camera_dir.mkdir(parents=True, exist_ok=True)

        copies: List[Tuple[Path, Path]] = []
        blanks_written = 0
        used_timestamps_ns: Set[int] = set()
        for frame_record in self._frame_records:
            sample_data = frame_record.get(camera_channel)

            src: Path | None = None
            if sample_data is not None:
                timestamp_ns = int(sample_data.timestamp) * 1000
                candidate = seq_path / sample_data.filename
                if candidate.exists():
                    src = candidate
            else:
                # No sample_data for this camera in this frame; use a neighbouring
                # synchronised sensor's timestamp so the blank image sorts into the
                # correct frame position at upload time.
                timestamp_ns = self._frame_timestamp_ns(frame_record)

            # Frames must map 1:1 onto files. Sequence construction builds frames by
            # counting files per sensor directory and pairing them by index, so a
            # dropped file shifts every later frame onto the wrong image and
            # leaves the last frame without one, failing Kognic scene validation
            # with "Sensors: [...] not present in frame: N".
            #
            if timestamp_ns in used_timestamps_ns:
                raise ValueError(
                    f"Camera {camera_channel} has duplicate timestamp {timestamp_ns} ns; "
                    "cannot create unique Kognic staging filenames"
                )
            used_timestamps_ns.add(timestamp_ns)

            dst = camera_dir / f"{timestamp_ns}.jpg"

            if src is not None:
                copies.append((src, dst))
                continue

            # Kognic requires every calibrated camera to be present in every
            # frame; a gap fails scene validation ("Sensors: [...] not present in
            # frame: N"). Fill it with a blank black image so the frame validates.
            logger.warning(
                f"Camera {camera_channel} missing for frame at {timestamp_ns}; "
                "writing a blank black image so the frame stays valid for Kognic"
            )
            self._write_blank_image(seq_path, camera_channel, dst)
            blanks_written += 1

        logger.info(
            f"{camera_channel}: {len(copies)} image copies queued, "
            f"{blanks_written} blank images written"
        )
        return copies

    def _frame_timestamp_ns(self, frame_record: Dict[str, object]) -> int:
        """Representative frame timestamp (ns) from any sensor present in it.

        Used to name a blank filler image when a camera has no sample_data for a
        frame. Sensors in one frame are time-synchronised, so a neighbour's
        timestamp places the blank at the right sort position for the missing
        camera. T4 timestamps are microseconds, hence ``* 1000``.

        Args:
            frame_record (Dict[str, object]): Channel-to-sample-data mapping.

        Returns:
            int: Representative frame timestamp in nanoseconds, or zero for an
                empty record.
        """
        for channel in self._channels_for_frame_records():
            sample_data = frame_record.get(channel)
            if sample_data is not None:
                return int(sample_data.timestamp) * 1000
        return 0

    def _write_blank_image(self, seq_path: Path, camera_channel: str, dst: Path) -> None:
        """Write a black JPEG matching a camera's resolution.

        Args:
            seq_path (Path): Source T4 sequence root.
            camera_channel (str): Camera channel defining image dimensions.
            dst (Path): Destination JPEG path.

        Returns:
            None

        Raises:
            RuntimeError: If Pillow is unavailable.
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                f"Pillow is required to write a blank filler image for camera "
                f"{camera_channel}; install it to continue."
            ) from exc

        image = self._blank_image_cache.get(camera_channel)
        if image is None:
            width, height = read_image_dims(self._sample_data_by_channel, seq_path, camera_channel)
            image = Image.new("RGB", (width, height), (0, 0, 0))
            self._blank_image_cache[camera_channel] = image

        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, format="JPEG")
