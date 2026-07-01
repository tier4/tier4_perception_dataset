from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

from .camera_calibration import CameraCalibration
from .camera_calibration import load_camera_calibrations
from .calibration import resolve_calibration
from .config import ConverterConfig
from .config import load_config
from .geometry import RigidTransform
from .geometry import compose
from .models import LidarSource
from .pointcloud import stamp_to_seconds
from .tf_manager import TfManager


@dataclass(frozen=True)
class RuntimeOptions:
    config_path: Path
    input_bags: list[Path]
    output_base: Path
    individual_params_root: Path
    vehicle_id: str
    sensor_model: str
    vehicle_model: str = "j6_gen2"
    start_time: float | None = None
    end_time: float | None = None
    dry_run: bool = False
    progress_interval: int = 100
    debug_timing: bool = False
    verbose: bool = False
    pcd_output_format: str | None = None
    output_rosbag_storage_id: str | None = None
    lidar_undistort_mode: str = "config"


@dataclass
class ConversionMetadata:
    tf_manager: TfManager
    camera_timestamps: dict[str, list[float]]
    tf_messages: int = 0
    camera_messages: int = 0
    index_events: int = 0
    index_seconds: float = 0.0
    index_tf_seconds: float = 0.0
    index_camera_seconds: float = 0.0


def convert(options: RuntimeOptions) -> Path | None:
    config = load_config(options.config_path)
    if options.pcd_output_format is not None:
        config.conversion["pcd_output_format"] = options.pcd_output_format
    if options.lidar_undistort_mode != "config":
        config.distortion_options["enabled"] = options.lidar_undistort_mode != "none"
        if options.lidar_undistort_mode in ("2d", "3d"):
            config.distortion_options["use_3d_distortion_correction"] = (
                options.lidar_undistort_mode == "3d"
            )
    calibration = resolve_calibration(
        options.individual_params_root,
        options.vehicle_id,
        options.sensor_model,
    )
    sources = _build_lidar_sources(config, calibration)
    camera_calibrations = load_camera_calibrations(
        calibration.root,
        config.conversion.get("camera_sensors", []),
    )
    if options.dry_run:
        _print_dry_run(config, calibration, sources, options, camera_calibrations)
        return None

    from geometry_msgs.msg import TwistWithCovarianceStamped
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import Imu

    from .concat import PointCloudConcatenator
    from .decode import LidarDecoder
    from .motion import DistortionBank
    from .rosbag import SequentialBagReader
    from .writer import T4DatasetWriter

    decoders = {
        source.packet_topic: LidarDecoder(
            lidar_name=source.lidar_name,
            packet_topic=source.packet_topic,
            frame_id=source.frame_id,
            calibration_csv=calibration.pandar_csv(source.lidar_name),
            base_to_lidar=source.base_to_sensor,
            sensor_model=source.sensor_model,
            return_mode=source.return_mode,
            min_range=source.min_range,
            max_range=source.max_range,
            cloud_min_angle=source.cloud_min_angle,
            cloud_max_angle=source.cloud_max_angle,
            cut_angle=source.cut_angle,
            dual_return_distance_threshold=source.dual_return_distance_threshold,
        )
        for source in sources
    }
    distortion = DistortionBank(
        {source.packet_topic: source.base_to_sensor for source in sources},
        _motion_imu_transforms(calibration),
        base_frame=calibration.base_frame,
        enabled=bool(config.distortion_options.get("enabled", True)),
        use_3d=bool(config.distortion_options.get("use_3d_distortion_correction", False)),
        use_imu=bool(config.distortion_options.get("use_imu", True)),
        update_azimuth_and_distance=bool(
            config.distortion_options.get("update_azimuth_and_distance", False)
        ),
    )
    concat = PointCloudConcatenator(
        sources=sources,
        output_frame=calibration.base_frame,
        allow_drop=config.allow_concat_drop,
        options=config.concat_options,
    )

    progress = _Progress(options.progress_interval)
    metadata = _collect_metadata(options, config, calibration, camera_calibrations)
    progress.index_events = metadata.index_events
    progress.add_time("index_total", metadata.index_seconds)
    progress.add_time("index_tf", metadata.index_tf_seconds)
    progress.add_time("index_camera", metadata.index_camera_seconds)
    progress.tf_messages += metadata.tf_messages
    progress.camera_messages += metadata.camera_messages

    scene_name = _scene_name(options.input_bags)
    start = perf_counter()
    rosbag_writer = _create_output_rosbag_writer(options, config, scene_name)
    progress.add_time("rosbag_setup", perf_counter() - start)
    writer = T4DatasetWriter(
        output_base=options.output_base,
        scene_name=scene_name,
        conversion=config.conversion,
        scene_description=config.scene_description,
        calibration=calibration,
        tf_manager=metadata.tf_manager,
        camera_calibrations=camera_calibrations,
        camera_timestamps=metadata.camera_timestamps,
    )
    start = perf_counter()
    writer.start()
    progress.add_time("t4_setup", perf_counter() - start)

    topics = _topics_to_read(config)
    reader = SequentialBagReader(options.input_bags)
    camera_topics = {camera["topic"] for camera in config.conversion.get("camera_sensors", [])}
    camera_indices: dict[str, int] = defaultdict(int)
    pass_through_topics = _pass_through_topics(config)
    generated_topics = _generated_rosbag_topics(config)
    distortion_imu_topic = str(config.distortion_options.get("input_imu_topic"))
    motion_twist_topic = str(config.distortion_options.get("input_twist_topic"))
    _debug(
        "topic filters "
        f"index={len(_metadata_topics(config))} read={len(topics)} "
        f"pass_through={len(pass_through_topics)} generated={len(generated_topics)}"
    )
    _print_decoder_config(sources)
    for event in reader.iter_events(topics=topics):
        progress.events += 1
        if _is_after_end(event.storage_time_ns, options.end_time):
            break
        if not _event_in_range(event.storage_time_ns, options.start_time, options.end_time):
            continue
        if event.topic in pass_through_topics:
            start = perf_counter()
            rosbag_writer.write_serialized(event.topic, event.serialized, event.storage_time_ns)
            if event.msg_type == "sensor_msgs/msg/CameraInfo":
                progress.camera_info_messages += 1
            progress.add_time("rosbag_passthrough", perf_counter() - start)
        if event.topic in camera_topics:
            start = perf_counter()
            msg = event.deserialize()
            if isinstance(msg, CompressedImage):
                writer.write_camera_message(event.topic, camera_indices[event.topic], msg)
                camera_indices[event.topic] += 1
            progress.add_time("camera_t4", perf_counter() - start)
            continue
        if event.topic in decoders:
            progress.lidar_messages += 1
            if options.verbose:
                _debug(
                    f"decode start topic={event.topic} "
                    f"storage_time={event.storage_time_ns * 1e-9:.6f}"
                )
            start = perf_counter()
            cloud = decoders[event.topic].decode_scan_cdr(event.serialized)
            progress.add_time("decode", perf_counter() - start)
            if cloud is None:
                progress.empty_decodes += 1
                progress.maybe_report("empty decode")
                continue
            progress.decoded_clouds += 1
            progress.topic_clouds[event.topic] += 1

            point_count = _point_count(cloud)
            progress.decoded_points += point_count
            if options.verbose:
                _debug(f"undistort start topic={event.topic} points={point_count}")
            start = perf_counter()
            if options.debug_timing:
                undistorted = distortion.undistort_with_status(event.topic, cloud)
                cloud = undistorted.cloud
                undistort_timings = undistorted.timings
                undistort_status = undistorted.status
            else:
                cloud = distortion.undistort(event.topic, cloud)
                undistort_timings = None
                undistort_status = None
            distortion_seconds = perf_counter() - start
            progress.add_time("undistort", distortion_seconds)
            progress.undistorted_clouds += 1
            progress.undistorted_points += _point_count(cloud)
            if undistort_timings is not None:
                progress.add_time("undistort_serialize", undistort_timings.serialize_seconds)
                progress.add_time("undistort_pybind", undistort_timings.pybind_seconds)
                progress.add_time("undistort_deserialize", undistort_timings.deserialize_seconds)
            if undistort_status is not None:
                progress.add_undistort_status(undistort_status)
            if options.verbose:
                timing_text = ""
                if undistort_timings is not None:
                    timing_text = (
                        f" serialize={undistort_timings.serialize_seconds:.3f}s"
                        f" pybind={undistort_timings.pybind_seconds:.3f}s"
                        f" deserialize={undistort_timings.deserialize_seconds:.3f}s"
                    )
                status_text = ""
                if undistort_status is not None:
                    status_text = (
                        f" mismatch={undistort_status.timestamp_mismatch_count}"
                        f" mismatch_fraction={undistort_status.timestamp_mismatch_fraction:.6f}"
                        f" twist_empty={undistort_status.twist_queue_empty}"
                        f" twist_late={undistort_status.twist_timestamp_too_late}"
                        f" imu_late={undistort_status.imu_timestamp_too_late}"
                    )
                _debug(
                    f"undistort done topic={event.topic} "
                    f"stamp={stamp_to_seconds(cloud.header.stamp):.6f} "
                    f"points={_point_count(cloud)} seconds={distortion_seconds:.3f}"
                    f"{timing_text}{status_text}"
                )

            start = perf_counter()
            new_frames = concat.add_cloud(
                event.topic,
                cloud,
                arrival_time=event.storage_time_ns * 1e-9,
            )
            progress.add_time("concat", perf_counter() - start)
            progress.concatenated_frames += len(new_frames)
            for frame in new_frames:
                progress.add_concat_frame(frame)
                if options.verbose:
                    _debug(
                        "concat frame "
                        f"status={frame.status} sources={frame.source_count}/{len(sources)} "
                        f"source_span={frame.source_span_sec * 1000.0:.3f}ms "
                        f"stamp={stamp_to_seconds(frame.cloud.header.stamp):.6f}"
                    )
                if options.verbose and frame.dropped_frames_missing_transform:
                    _debug(
                        "concat dropped missing transforms "
                        f"frames={list(frame.dropped_frames_missing_transform)}"
                    )
                start = perf_counter()
                should_write_rosbag = writer.write_lidar_frame(frame)
                progress.add_time("t4_lidar_write", perf_counter() - start)
                if should_write_rosbag:
                    start = perf_counter()
                    _write_concat_to_rosbag(rosbag_writer, config, frame)
                    progress.add_time("rosbag_generated", perf_counter() - start)
            if options.verbose and new_frames:
                _debug(f"concat emitted frames={len(new_frames)} total_frames={writer.frame_count}")
            progress.maybe_report("lidar")
            continue
        if event.msg_type == "sensor_msgs/msg/Imu" and event.topic == distortion_imu_topic:
            start = perf_counter()
            msg = event.deserialize()
            if isinstance(msg, Imu):
                distortion.process_imu(msg)
                progress.imu_messages += 1
            progress.add_time("imu", perf_counter() - start)
            continue
        if (
            event.msg_type == "geometry_msgs/msg/TwistWithCovarianceStamped"
            and event.topic == motion_twist_topic
        ):
            start = perf_counter()
            msg = event.deserialize()
            if isinstance(msg, TwistWithCovarianceStamped):
                distortion.process_twist(msg)
                concat.process_twist(msg)
                progress.twist_messages += 1
            progress.add_time("twist", perf_counter() - start)
            continue

    progress.report("bag read complete", force=True)
    start = perf_counter()
    finish_frames = concat.finish()
    progress.add_time("concat_finish", perf_counter() - start)
    progress.concatenated_frames += len(finish_frames)
    for frame in finish_frames:
        progress.add_concat_frame(frame)
        if options.verbose:
            _debug(
                "concat frame "
                f"status={frame.status} sources={frame.source_count}/{len(sources)} "
                f"source_span={frame.source_span_sec * 1000.0:.3f}ms "
                f"stamp={stamp_to_seconds(frame.cloud.header.stamp):.6f}"
            )
        if options.verbose and frame.dropped_frames_missing_transform:
            _debug(
                "concat dropped missing transforms "
                f"frames={list(frame.dropped_frames_missing_transform)}"
            )
        start = perf_counter()
        should_write_rosbag = writer.write_lidar_frame(frame)
        progress.add_time("t4_lidar_write", perf_counter() - start)
        if should_write_rosbag:
            start = perf_counter()
            _write_concat_to_rosbag(rosbag_writer, config, frame)
            progress.add_time("rosbag_generated", perf_counter() - start)
    progress.report("concat finish", force=True)

    if writer.frame_count == 0:
        progress.report("no frames generated", force=True)
        raise RuntimeError("No concatenated LiDAR frames were generated")

    _debug(f"writer finalize frames={writer.frame_count} output={options.output_base}")
    start = perf_counter()
    output = writer.finalize()
    progress.add_time("t4_finalize", perf_counter() - start)
    start = perf_counter()
    rosbag_writer.close()
    progress.add_time("rosbag_close", perf_counter() - start)
    progress.report("writer complete", force=True)
    if options.debug_timing:
        progress.final_timing_report()
    print(f"Wrote T4 dataset: {output}")
    return output


class _Progress:
    def __init__(self, interval: int) -> None:
        self.interval = max(1, interval)
        self.started_at = perf_counter()
        self.last_report_at = self.started_at
        self.events = 0
        self.index_events = 0
        self.lidar_messages = 0
        self.decoded_clouds = 0
        self.empty_decodes = 0
        self.undistorted_clouds = 0
        self.decoded_points = 0
        self.undistorted_points = 0
        self.concatenated_frames = 0
        self.tf_messages = 0
        self.camera_info_messages = 0
        self.camera_messages = 0
        self.imu_messages = 0
        self.twist_messages = 0
        self.stage_seconds: dict[str, float] = defaultdict(float)
        self.topic_clouds: dict[str, int] = defaultdict(int)
        self.concat_statuses: dict[str, int] = defaultdict(int)
        self.concat_source_count_sum = 0
        self.concat_source_span_sum_sec = 0.0
        self.concat_source_span_max_sec = 0.0
        self.concat_missing_transform_frames = 0
        self.undistort_status_counts: dict[str, int] = defaultdict(int)
        self.undistort_timestamp_mismatches = 0

    def add_time(self, stage: str, seconds: float) -> None:
        self.stage_seconds[stage] += seconds

    def add_concat_frame(self, frame: Any) -> None:
        status = str(getattr(frame, "status", "") or "unknown")
        self.concat_statuses[status] += 1
        self.concat_source_count_sum += int(getattr(frame, "source_count", 0))
        span = float(getattr(frame, "source_span_sec", 0.0) or 0.0)
        self.concat_source_span_sum_sec += span
        self.concat_source_span_max_sec = max(self.concat_source_span_max_sec, span)
        if getattr(frame, "dropped_frames_missing_transform", ()):
            self.concat_missing_transform_frames += 1

    def add_undistort_status(self, status: Any) -> None:
        mismatch_count = int(getattr(status, "timestamp_mismatch_count", 0) or 0)
        if mismatch_count:
            self.undistort_status_counts["timestamp_mismatch_clouds"] += 1
            self.undistort_timestamp_mismatches += mismatch_count
        for name in (
            "twist_queue_empty",
            "twist_timestamp_too_late",
            "imu_timestamp_too_late",
        ):
            if bool(getattr(status, name, False)):
                self.undistort_status_counts[name] += 1

    def maybe_report(self, reason: str) -> None:
        if self.lidar_messages and self.lidar_messages % self.interval == 0:
            self.report(reason)

    def report(self, reason: str, *, force: bool = False) -> None:
        now = perf_counter()
        if not force and now - self.last_report_at < 1.0:
            return
        elapsed = now - self.started_at
        rate = self.lidar_messages / elapsed if elapsed else 0.0
        index_summary = (
            f"total={self.stage_seconds.get('index_total', 0.0):.2f}s "
            f"tf={self.stage_seconds.get('index_tf', 0.0):.2f}s "
            f"camera={self.stage_seconds.get('index_camera', 0.0):.2f}s"
        )
        stage_summary = " ".join(
            f"{name}={seconds:.2f}s"
            for name, seconds in sorted(self.stage_seconds.items())
            if not name.startswith("index_") and not name.startswith("undistort_")
        )
        topic_summary = " ".join(
            f"{topic.rsplit('/', 2)[-2]}={count}" for topic, count in sorted(self.topic_clouds.items())
        )
        _debug(
            f"progress reason={reason} elapsed={elapsed:.1f}s events={self.events} "
            f"index_events={self.index_events} "
            f"lidar_msgs={self.lidar_messages} decoded={self.decoded_clouds} "
            f"empty_decodes={self.empty_decodes} undistorted={self.undistorted_clouds} "
            f"frames={self.concatenated_frames} camera={self.camera_messages} "
            f"camera_info={self.camera_info_messages} imu={self.imu_messages} "
            f"twist={self.twist_messages} tf={self.tf_messages} "
            f"lidar_rate={rate:.2f}/s index=[{index_summary}] "
            f"stages=[{stage_summary}] topics=[{topic_summary}]"
        )
        self.last_report_at = now

    def final_timing_report(self) -> None:
        elapsed = perf_counter() - self.started_at
        accounted_stages = [
            name
            for name in self.stage_seconds
            if name not in {"index_tf", "index_camera"}
            and not name.startswith("undistort_")
        ]
        accounted = sum(self.stage_seconds[name] for name in accounted_stages)
        unaccounted = max(0.0, elapsed - accounted)
        rows = [
            ("index_total", "metadata/index pass: TF and camera timestamp cache"),
            ("decode", "Nebula packet decode to per-LiDAR PointCloud2"),
            ("undistort", "Autoware distortion correction per LiDAR scan"),
            ("undistort_pybind", "distortion correction C++ binding body"),
            ("undistort_serialize", "PointCloud2 serialization before pybind call"),
            ("undistort_deserialize", "PointCloud2 deserialization after pybind call"),
            ("concat", "Autoware pointcloud time sync and concatenation"),
            ("concat_finish", "flush incomplete concatenator frames"),
            ("t4_lidar_write", "T4 lidar sample data rows and pointcloud file writes"),
            ("camera_t4", "T4 camera sample selection and sample data rows"),
            ("imu", "IMU deserialize and distortion queue update"),
            ("twist", "twist deserialize and motion queue update"),
            ("rosbag_passthrough", "copy configured input topics to output rosbag"),
            ("rosbag_generated", "write generated concat cloud/info to output rosbag"),
            ("rosbag_setup", "create output rosbag topics"),
            ("rosbag_close", "close output rosbag writer"),
            ("t4_setup", "initialize T4 output tables"),
            ("t4_finalize", "finalize T4 metadata tables"),
        ]
        _debug("timing summary")
        _debug(
            f"  elapsed={elapsed:.2f}s lidar_msgs={self.lidar_messages} "
            f"frames={self.concatenated_frames} lidar_rate={self.lidar_messages / elapsed if elapsed else 0.0:.2f}/s "
            f"frame_rate={self.concatenated_frames / elapsed if elapsed else 0.0:.2f}/s"
        )
        if self.decoded_points:
            _debug(
                f"  points decoded={self.decoded_points} undistorted={self.undistorted_points} "
                f"decode_rate={self.decoded_points / max(self.stage_seconds.get('decode', 0.0), 1e-9) / 1e6:.2f}Mpts/s "
                f"undistort_rate={self.undistorted_points / max(self.stage_seconds.get('undistort', 0.0), 1e-9) / 1e6:.2f}Mpts/s"
            )
        _debug("  operation              time(s)  elapsed%  description")
        for name, description in rows:
            seconds = self.stage_seconds.get(name, 0.0)
            if seconds <= 0.0:
                continue
            _debug(f"  {name:<22} {seconds:7.2f}  {seconds / elapsed * 100.0:7.2f}  {description}")
        if unaccounted > 0.01:
            _debug(
                f"  {'unaccounted':<22} {unaccounted:7.2f}  {unaccounted / elapsed * 100.0:7.2f}  loop overhead and uncategorized work"
            )

        status_summary = " ".join(
            f"{status}={count}" for status, count in sorted(self.concat_statuses.items())
        )
        if self.concatenated_frames:
            _debug(
                "  concat summary "
                f"statuses=[{status_summary}] "
                f"avg_sources={self.concat_source_count_sum / self.concatenated_frames:.2f} "
                f"avg_source_span={self.concat_source_span_sum_sec / self.concatenated_frames * 1000.0:.3f}ms "
                f"max_source_span={self.concat_source_span_max_sec * 1000.0:.3f}ms "
                f"missing_transform_frames={self.concat_missing_transform_frames}"
            )
        if self.undistort_status_counts:
            status_text = " ".join(
                f"{name}={count}" for name, count in sorted(self.undistort_status_counts.items())
            )
            _debug(
                "  undistort status "
                f"{status_text} timestamp_mismatches={self.undistort_timestamp_mismatches}"
            )


def _debug(message: str) -> None:
    print(f"[converter] {message}", flush=True)


def _point_count(cloud: Any) -> int:
    return int(getattr(cloud, "width", 0)) * int(getattr(cloud, "height", 0))


def _print_decoder_config(sources: list[LidarSource]) -> None:
    _debug("decoder configuration:")
    for source in sources:
        _debug(
            f"  topic={source.packet_topic} lidar={source.lidar_name} "
            f"model={source.sensor_model} return_mode={source.return_mode} "
            f"frame={source.frame_id}"
        )


def _collect_metadata(
    options: RuntimeOptions,
    config: ConverterConfig,
    calibration,
    camera_calibrations: dict[str, CameraCalibration],
) -> ConversionMetadata:
    from sensor_msgs.msg import CompressedImage

    from .rosbag import SequentialBagReader

    tf_manager = TfManager(calibration, camera_calibrations.values())
    camera_topics = _camera_topics(config)
    topics = _metadata_topics(config)
    metadata = ConversionMetadata(
        tf_manager=tf_manager,
        camera_timestamps={topic: [] for topic in camera_topics},
    )
    index_start = perf_counter()
    reader = SequentialBagReader(options.input_bags)
    for event in reader.iter_events(topics=topics):
        metadata.index_events += 1
        if _is_after_end(event.storage_time_ns, options.end_time):
            break
        if event.msg_type == "tf2_msgs/msg/TFMessage":
            start = perf_counter()
            msg = event.deserialize()
            tf_manager.add_tf_message(msg, is_static=(event.topic == "/tf_static"))
            metadata.index_tf_seconds += perf_counter() - start
            metadata.tf_messages += 1
            continue
        if not _event_in_range(event.storage_time_ns, options.start_time, options.end_time):
            continue
        if event.topic in camera_topics:
            start = perf_counter()
            msg = event.deserialize()
            if isinstance(msg, CompressedImage):
                metadata.camera_timestamps[event.topic].append(stamp_to_seconds(msg.header.stamp))
                metadata.camera_messages += 1
            metadata.index_camera_seconds += perf_counter() - start
            continue
    metadata.index_seconds = perf_counter() - index_start
    _debug(
        "metadata pass complete "
        f"events={metadata.index_events} topics={len(topics)} "
        f"elapsed={metadata.index_seconds:.2f}s tf={metadata.tf_messages} "
        f"camera={metadata.camera_messages} "
        f"tf_deserialize={metadata.index_tf_seconds:.2f}s "
        f"camera_deserialize={metadata.index_camera_seconds:.2f}s"
    )
    return metadata


def _create_output_rosbag_writer(options: RuntimeOptions, config: ConverterConfig, scene_name: str):
    from .rosbag import SequentialBagReader
    from .rosbag import SequentialBagWriter
    from .rosbag import infer_storage_id

    storage_id = options.output_rosbag_storage_id or infer_storage_id(options.input_bags[0])
    output_path = options.output_base / scene_name / "rosbag2"
    if output_path.exists():
        shutil.rmtree(output_path)
    writer = SequentialBagWriter(output_path, storage_id=storage_id)
    writer.open()

    metadata = SequentialBagReader(options.input_bags).topic_metadata()
    pass_through = sorted(_pass_through_topics(config))
    created_pass_through = 0
    missing_pass_through = 0
    for topic in pass_through:
        topic_metadata = metadata.get(topic)
        if topic_metadata is None:
            print(f"Warning: configured rosbag output topic not found in input bag: {topic}")
            missing_pass_through += 1
            continue
        writer.create_existing_topic(topic_metadata)
        created_pass_through += 1

    lidar_sensor = config.conversion["lidar_sensor"]
    writer.create_topic(lidar_sensor["topic"], "sensor_msgs/msg/PointCloud2")
    generated_count = 1
    if lidar_sensor.get("lidar_info_topic"):
        writer.create_topic(
            lidar_sensor["lidar_info_topic"],
            "autoware_sensing_msgs/msg/ConcatenatedPointCloudInfo",
        )
        generated_count += 1
    _debug(
        f"output rosbag path={output_path} storage_id={storage_id} "
        f"configured_pass_through={len(pass_through)} created_pass_through={created_pass_through} "
        f"missing_pass_through={missing_pass_through} generated={generated_count}"
    )
    return writer


def _pass_through_topics(config: ConverterConfig) -> set[str]:
    generated = _generated_rosbag_topics(config)
    return {topic for topic in config.filter_topics if topic and topic not in generated}


def _generated_rosbag_topics(config: ConverterConfig) -> set[str]:
    topics = {
        config.conversion["lidar_sensor"]["topic"],
        config.conversion["lidar_sensor"].get("lidar_info_topic", ""),
    }
    return {topic for topic in topics if topic}


def _camera_topics(config: ConverterConfig) -> set[str]:
    return {camera["topic"] for camera in config.conversion.get("camera_sensors", [])}


def _metadata_topics(config: ConverterConfig) -> set[str]:
    return _camera_topics(config) | {"/tf", "/tf_static"}


def _write_concat_to_rosbag(rosbag_writer, config: ConverterConfig, frame) -> None:
    timestamp_ns = _stamp_to_nanoseconds(frame.cloud.header.stamp)
    lidar_sensor = config.conversion["lidar_sensor"]
    rosbag_writer.write_message(lidar_sensor["topic"], frame.cloud, timestamp_ns)
    if lidar_sensor.get("lidar_info_topic"):
        rosbag_writer.write_message(lidar_sensor["lidar_info_topic"], frame.info, timestamp_ns)


def _stamp_to_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _build_lidar_sources(config: ConverterConfig, calibration) -> list[LidarSource]:
    mappings = config.conversion["lidar_sensor"].get("lidar_sources_mapping", [])
    mapping_by_name = {_lidar_name_from_topic(item["topic"]): item for item in mappings}
    sources = []
    for topic in config.lidar_topics:
        mapping = mapping_by_name.get(topic.lidar_name)
        if mapping is None:
            raise ValueError(
                f"No lidar_sources_mapping entry for raw topic {topic.name} "
                f"(resolved lidar name: {topic.lidar_name})"
            )
        frame_id = mapping["frame_id"]
        base_to_sensor = _base_to_lidar_frame(
            calibration,
            topic.lidar_name,
            frame_id,
            topic.sensor_model,
        )
        sources.append(
            LidarSource(
                packet_topic=topic.name,
                info_topic=mapping["topic"],
                lidar_name=topic.lidar_name,
                frame_id=frame_id,
                is_reset_topic=topic.is_reset_topic,
                base_to_sensor=base_to_sensor,
                sensor_model=topic.sensor_model,
                return_mode=topic.return_mode,
                min_range=topic.min_range,
                max_range=topic.max_range,
                cloud_min_angle=topic.cloud_min_angle,
                cloud_max_angle=topic.cloud_max_angle,
                cut_angle=topic.cut_angle,
                dual_return_distance_threshold=topic.dual_return_distance_threshold,
            )
        )
    return sources


def _base_to_lidar_frame(calibration, lidar_name: str, frame_id: str, sensor_model: str):
    try:
        return calibration.base_to_frame(frame_id)
    except KeyError:
        pass
    if frame_id != f"{lidar_name}/lidar":
        raise KeyError(f"No transform from {calibration.base_frame} to {frame_id}")
    base_to_lidar_base = calibration.base_to_frame(f"{lidar_name}/lidar_base_link")
    return compose(base_to_lidar_base, _lidar_base_to_lidar(lidar_name, sensor_model))


def _lidar_base_to_lidar(lidar_name: str, sensor_model: str) -> RigidTransform:
    offsets = {
        "Pandar128E4X": 0.09,
        "PandarQT128": 0.0582,
    }
    if sensor_model not in offsets:
        raise KeyError(f"No lidar_base_link -> lidar offset for sensor_model={sensor_model}")
    return RigidTransform(
        parent=f"{lidar_name}/lidar_base_link",
        child=f"{lidar_name}/lidar",
        translation=(0.0, 0.0, offsets[sensor_model]),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def _topics_to_read(config: ConverterConfig) -> set[str]:
    topics = _pass_through_topics(config)
    topics.update(topic.name for topic in config.lidar_topics)
    topics.update(camera["topic"] for camera in config.conversion.get("camera_sensors", []))
    topics.add(str(config.distortion_options.get("input_twist_topic")))
    topics.add(str(config.distortion_options.get("input_imu_topic")))
    ins_mapping = config.conversion.get("ins_topic_mapping") or {}
    topics.update(topic for topic in ins_mapping.values() if isinstance(topic, str))
    topics.update(topic["name"] for topic in config.other_topics if "name" in topic)
    return topics


def _motion_imu_transforms(calibration) -> dict[str, Any]:
    transforms: dict[str, Any] = {}
    for parent, child in calibration.transforms or {}:
        if child.endswith("/imu_link") or child == calibration.base_frame:
            transforms[child] = calibration.base_to_frame(child)
    transforms[calibration.base_frame] = calibration.base_to_frame(calibration.base_frame)
    return transforms


def _event_in_range(storage_time_ns: int, start_time: float | None, end_time: float | None) -> bool:
    timestamp = storage_time_ns * 1e-9
    if start_time is not None and timestamp < start_time:
        return False
    if end_time is not None and timestamp > end_time:
        return False
    return True


def _is_after_end(storage_time_ns: int, end_time: float | None) -> bool:
    if end_time is None:
        return False
    return storage_time_ns * 1e-9 > end_time


def _lidar_name_from_topic(topic: str) -> str:
    parts = topic.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "sensing" and parts[1] == "lidar":
        return parts[2]
    return parts[-2] if len(parts) > 1 else topic.strip("/")


def _scene_name(input_bags: list[Path]) -> str:
    if len(input_bags) == 1:
        return input_bags[0].name
    return input_bags[0].parent.name or "multi_bag_scene"


def _print_dry_run(
    config: ConverterConfig,
    calibration,
    sources: list[LidarSource],
    options: RuntimeOptions,
    camera_calibrations: dict[str, CameraCalibration],
) -> None:
    print("Converter dry run")
    print(f"  config: {options.config_path}")
    print(f"  input_bags: {[str(path) for path in options.input_bags]}")
    print(f"  output_base: {options.output_base}")
    print(f"  vehicle_id: {options.vehicle_id}")
    print(f"  vehicle_model: {options.vehicle_model}")
    print(f"  sensor_model: {options.sensor_model}")
    print(f"  calibration_root: {calibration.root}")
    print(f"  concat_options: {config.concat_options}")
    print("  lidar sources:")
    for source in sources:
        print(
            f"    {source.packet_topic} -> {source.info_topic} "
            f"({source.frame_id}, reset={source.is_reset_topic}, "
            f"model={source.sensor_model}, return_mode={source.return_mode})"
        )
    print("  camera topics:")
    for camera in config.conversion.get("camera_sensors", []):
        calib = camera_calibrations[camera["topic"]]
        print(
            f"    {camera['topic']} -> {camera['channel']} "
            f"frame={calib.frame_id} size={calib.width}x{calib.height} "
            f"delay={camera.get('delay_msec', 0)}ms"
        )
