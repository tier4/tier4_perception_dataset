"""Convert T4 sensor data to the Kognic staging layout."""

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import time
from typing import Dict, List, Set, Tuple

from t4_devkit import Tier4

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import LIDAR_CONCAT_CHANNEL, LIDAR_CONCAT_NUM_POINT_FEATURES
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


class T4ToKognicConverter(AbstractConverter[None]):
    """Convert T4 data (annotated or non-annotated) to the Kognic IO staging layout.

    Only sensor data, calibration, and ego poses are exported;
    annotation tables,  if present, are ignored.

    output layout:

        <output_base>/<input_item_name>/
            calibration.json
            ego_poses.json
            keyframes.json
            cameras/<camera_name>/<timestamp_ns>.jpg
            lidar/<lidar_name>/<timestamp_ns>.csv

    ``keyframes.json`` holds the staging frame indices of the keyframes; the
    uploader marks exactly those frames ``annotate=True`` instead of walking a
    fixed ``target_hz`` grid. For annotated datasets (``annotated=True``) the
    keyframes are the frames whose sample carries at least one
    ``sample_annotation``. Non-annotated datasets have no annotations, so their
    keyframes are instead selected by sample index at ``annotation_hz``,
    matching the non-annotated T4 -> Deepen converter.
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        camera_sensors: list,
        workers_number: int = 32,
        drop_camera_token_not_found: bool = False,
        annotated: bool = True,
        annotation_hz: int = 10,
        lidar_point_stride: int | None = LIDAR_CONCAT_NUM_POINT_FEATURES,
        generate_tsv_report: bool = False,
    ):
        """Initialize the converter.

        Args:
            input_base (str): Input T4 dataset directory.
            output_base (str): Destination staging directory.
            camera_sensors (list): Camera configuration records.
            workers_number (int): Number of image-copy worker threads.
            drop_camera_token_not_found (bool): Whether to omit missing camera
                frames instead of writing blank images.
            annotated (bool): Whether the source carries T4 annotations.
            annotation_hz (int): Keyframe frequency for non-annotated data, in
                ``1..10``.
            lidar_point_stride (int | None): Explicit floats per point for
                fused clouds without ``LIDAR_CONCAT_INFO``. Set to ``None`` to
                require unambiguous automatic detection.
            generate_tsv_report (bool): Write ``conversion_report.tsv`` in
                ``output_base`` with scene outcomes and missing sensor frames.

        Raises:
            ValueError: If ``annotation_hz`` is outside ``1..10``.
        """
        super().__init__(input_base, output_base)
        self._camera_channels: List[str] = [cam["channel"] for cam in camera_sensors]
        self._workers_number = workers_number
        self._drop_camera_token_not_found = drop_camera_token_not_found
        self._annotated = annotated
        self._annotation_hz = validate_annotation_hz(annotation_hz)
        self._lidar_point_stride = lidar_point_stride
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

        # The uploader pairs sensor files with frames by sorted position, so any
        # file left over from a previous generation would shift that pairing.
        for stale_dir in (out_dir / "cameras", out_dir / "lidar"):
            shutil.rmtree(stale_dir, ignore_errors=True)

        self._build_lookup_maps(seq_path)
        self._has_lidar_concat_info = (seq_path / "data" / "LIDAR_CONCAT_INFO").is_dir()
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
                point_stride=self._lidar_point_stride,
            )

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
        if self._drop_camera_token_not_found:
            return "camera frame dropped"
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

        self._annotated_sample_tokens: Set[str] = set()
        if self._annotated:
            self._annotated_sample_tokens = {
                annotation.sample_token for annotation in t4.get_table("sample_annotation")
            }

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

        Annotated datasets: a frame is a keyframe when its sample carries at
        least one ``sample_annotation``.

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
                if getattr(frame_record.get(self._anchor_channel), "sample_token", None)
                in self._annotated_sample_tokens
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
        collisions = 0
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

            # Frames must map 1:1 onto files. The uploader rebuilds frames by
            # counting files per sensor directory and pairing them by index, so a
            # dropped file shifts every later frame onto the wrong image and
            # leaves the last frame without one, failing Kognic scene validation
            # with "Sensors: [...] not present in frame: N".
            #
            # T4 can repeat a timestamp across frames: when a camera drops out,
            # upstream extraction writes a black image but reuses the previous
            # capture's timestamp, so a run of dropped frames shares one
            # timestamp. Naming files by timestamp alone would collapse the run
            # into a single file, so nudge each duplicate forward by 1ns. Source
            # frame intervals are ~1e8 ns, hence the nudged file still sorts into
            # its own frame position.
            if timestamp_ns in used_timestamps_ns:
                collisions += 1
                while timestamp_ns in used_timestamps_ns:
                    timestamp_ns += 1
            used_timestamps_ns.add(timestamp_ns)

            dst = camera_dir / f"{timestamp_ns}.jpg"

            if src is not None:
                copies.append((src, dst))
                continue

            if self._drop_camera_token_not_found:
                logger.warning(f"Camera {camera_channel} missing for selected frame; dropping")
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

        if collisions:
            logger.warning(
                f"{camera_channel}: {collisions} frame(s) shared a timestamp with an "
                "earlier frame and were renamed with a 1ns offset to keep one file "
                "per frame; upstream T4 likely reused a timestamp for dropped frames"
            )
        logger.info(
            f"{camera_channel}: {len(copies)} image copies queued, "
            f"{blanks_written} blank images written, {collisions} timestamp collisions"
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
                f"{camera_channel}; install it or set drop_camera_token_not_found."
            ) from exc

        image = self._blank_image_cache.get(camera_channel)
        if image is None:
            width, height = read_image_dims(self._sample_data_by_channel, seq_path, camera_channel)
            image = Image.new("RGB", (width, height), (0, 0, 0))
            self._blank_image_cache[camera_channel] = image

        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, format="JPEG")
