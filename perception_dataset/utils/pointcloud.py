"""Point-cloud helpers for reading and exporting T4 LIDAR_CONCAT data."""

from pathlib import Path
import shutil
from typing import Dict, List

import numpy as np
from t4_devkit.dataclass import LidarPointCloud

from perception_dataset.constants import (
    LIDAR_CONCAT_CHANNEL,
)
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

# Upper bound for a plausible |x|, |y| or |z| in the sensor frame; used to
# reject wrong stride guesses (misaligned reshapes leak time offsets etc.
# into the coordinate columns).
_MAX_REASONABLE_COORDINATE_M = 10_000.0


def valid_point_mask(points: np.ndarray) -> np.ndarray:
    """Return the validity mask applied when exporting points to Kognic.

    ``save_pointcloud_csv`` fails the conversion when this mask is not
    all-True. Semantic-segmentation labels use the same mask so their RLE
    length always matches the uploaded point count.

    Args:
        points (np.ndarray): Point records whose first four columns are
            ``x``, ``y``, ``z``, and intensity.

    Returns:
        np.ndarray: Boolean mask selecting finite, reasonably-sized points.
    """
    return np.isfinite(points[:, 0:4]).all(axis=1) & (
        np.abs(points[:, 0:3]) < _MAX_REASONABLE_COORDINATE_M
    ).all(axis=1)


def extract_pointclouds(
    seq_path: Path,
    out_dir: Path,
    lidar_channels: List[str],
    frame_records: List[Dict[str, dict]],
    channel_to_token: Dict[str, str],
) -> None:
    """Write per-frame CSV point clouds for all requested lidar channels.

    Args:
        seq_path (Path): Root directory of the source T4 sequence.
        out_dir (Path): Destination root. Files are written below
            ``lidar/<lidar_channel>``.
        lidar_channels (List[str]): Lidar sensor channels to extract.
        frame_records (List[Dict[str, dict]]): Ordered frame mappings from
            channel names to sample-data records.
        channel_to_token (Dict[str, str]): Mapping from channel names to sensor
            tokens.

    Returns:
        None

    Raises:
        FileNotFoundError: If required point-cloud or concat-info data is
            missing.
        ValueError: If a point-cloud binary layout is inconsistent.
    """
    stats: Dict[str, Dict[str, int]] = {}
    sensor_token_by_channel: Dict[str, str] = {}
    for lidar_channel in lidar_channels:
        sensor_token = channel_to_token.get(lidar_channel)
        if sensor_token is None:
            logger.warning(f"LiDAR {lidar_channel} not found in {seq_path}; skipping")
            continue
        sensor_token_by_channel[lidar_channel] = sensor_token
        stats[lidar_channel] = {"count": 0, "blank_count": 0}
        (out_dir / "lidar" / lidar_channel).mkdir(parents=True, exist_ok=True)

    if not sensor_token_by_channel:
        return

    for frame_record in frame_records:
        concat_sample_data = frame_record.get(LIDAR_CONCAT_CHANNEL)
        if concat_sample_data is None:
            continue

        bin_path = seq_path / concat_sample_data.filename
        if not bin_path.exists():
            raise FileNotFoundError(f"Required LIDAR_CONCAT point cloud is missing: {bin_path}")

        info_filename = concat_sample_data.info_filename
        has_per_sensor_channel = any(
            lidar_channel != LIDAR_CONCAT_CHANNEL for lidar_channel in sensor_token_by_channel
        )
        if has_per_sensor_channel and not info_filename:
            raise FileNotFoundError(
                f"LIDAR_CONCAT_INFO is required but missing in sample_data for "
                f"sample_data {concat_sample_data.token}"
            )

        info_path = seq_path / info_filename if info_filename else None
        if info_path is not None and not info_path.exists():
            raise FileNotFoundError(f"Required LIDAR_CONCAT_INFO file is missing: {info_path}")

        pointcloud = LidarPointCloud.from_file(
            str(bin_path),
            metainfo_filepath=str(info_path) if info_path is not None else None,
        )
        source_pointclouds = (
            pointcloud.split_by_sensor() if info_path is not None and pointcloud.metainfo else {}
        )
        sources_by_token = (
            {source.sensor_token: source for source in pointcloud.metainfo.sources}
            if info_path is not None and pointcloud.metainfo
            else {}
        )

        concat_timestamp_ns = int(concat_sample_data.timestamp) * 1000
        for lidar_channel, sensor_token in sensor_token_by_channel.items():
            lidar_dir = out_dir / "lidar" / lidar_channel
            if lidar_channel == LIDAR_CONCAT_CHANNEL:
                csv_path = lidar_dir / f"{concat_timestamp_ns}.csv"
                save_pointcloud_csv(csv_path, concat_timestamp_ns, pointcloud.points.T)
                stats[lidar_channel]["count"] += 1
                continue

            source = sources_by_token.get(sensor_token)
            timestamp_ns = (
                source.stamp.sec * 1_000_000_000 + source.stamp.nanosec
                if source is not None
                else None
            )
            if not timestamp_ns:
                timestamp_ns = concat_timestamp_ns
            csv_path = lidar_dir / f"{timestamp_ns}.csv"

            source_pointcloud = source_pointclouds.get(sensor_token)
            if source_pointcloud is None or source_pointcloud.num_points() == 0:
                # The sensor contributed no points to this concat sweep (dropped
                # out, or started after the recording began). Still write a
                # header-only CSV: ensures the uploader still recognizes this frame even though it has no points.
                save_pointcloud_csv(csv_path, timestamp_ns, np.empty((0, 4), dtype=np.float32))
                stats[lidar_channel]["blank_count"] += 1
                continue

            save_pointcloud_csv(csv_path, timestamp_ns, source_pointcloud.points.T)
            stats[lidar_channel]["count"] += 1

    for lidar_channel, channel_stats in stats.items():
        logger.info(
            f"{lidar_channel}: {channel_stats['count']} point clouds extracted, "
            f"{channel_stats['blank_count']} blank frames written"
        )


def save_pointcloud_csv(csv_path: Path, timestamp_ns: int, points: np.ndarray) -> None:
    """Write *points* to a CSV at *csv_path*.

    The output columns are: ``ts_gps, x, y, z, intensity``.

    Args:
        csv_path (Path): Destination CSV path.
        timestamp_ns (int): Capture timestamp in nanoseconds.
        points (np.ndarray): Point records whose first four columns are
            ``x``, ``y``, ``z``, and intensity.

    Returns:
        None

    Raises:
        ValueError: If any point has a non-finite value or a coordinate beyond
            ``_MAX_REASONABLE_COORDINATE_M`` (corrupt lidar returns). Kognic
            rejects the whole scene on any coordinate outside its int32-backed
            range, so the conversion fails instead of staging a scene the
            upload cannot process.
    """
    valid = valid_point_mask(points)
    if not valid.all():
        bad_indices = np.flatnonzero(~valid)
        raise ValueError(
            f"{csv_path}: {len(bad_indices)} of {len(points)} points have non-finite or "
            f"out-of-range coordinates (|x,y,z| must be < {_MAX_REASONABLE_COORDINATE_M} m); "
            f"first bad point at index {int(bad_indices[0])}: "
            f"{points[bad_indices[0], 0:4].tolist()}"
        )

    with open(csv_path, "w") as f:
        f.write("ts_gps,x,y,z,intensity\n")
        for x, y, z, intensity, *_ in points:
            f.write(
                f"{timestamp_ns},{float(x):.6f},{float(y):.6f},{float(z):.6f},{float(intensity):.6f}\n"
            )


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file while creating its destination directory.

    Args:
        src (Path): Source file path.
        dst (Path): Destination file path.

    Returns:
        None
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
