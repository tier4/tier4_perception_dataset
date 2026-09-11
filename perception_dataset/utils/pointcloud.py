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
    lidar_channel: str,
    frame_records: List[Dict[str, dict]],
    channel_to_token: Dict[str, str],
) -> None:
    """Write per-frame CSV point clouds for a lidar channel.

    Args:
        seq_path (Path): Root directory of the source T4 sequence.
        out_dir (Path): Destination root. Files are written below
            ``lidar/<lidar_channel>``.
        lidar_channel (str): Lidar sensor channel to extract.
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
    sensor_token = channel_to_token.get(lidar_channel)
    if sensor_token is None:
        logger.warning(f"LiDAR {lidar_channel} not found in {seq_path}; skipping")
        return

    lidar_dir = out_dir / "lidar" / lidar_channel
    lidar_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    blank_count = 0
    for frame_record in frame_records:
        concat_sample_data = frame_record.get(LIDAR_CONCAT_CHANNEL)
        if concat_sample_data is None:
            continue

        bin_path = seq_path / concat_sample_data.filename
        if not bin_path.exists():
            raise FileNotFoundError(f"Required LIDAR_CONCAT point cloud is missing: {bin_path}")

        if lidar_channel == LIDAR_CONCAT_CHANNEL:
            timestamp_ns = int(concat_sample_data.timestamp) * 1000
            metainfo_path = (
                seq_path / concat_sample_data.info_filename
                if concat_sample_data.info_filename
                else None
            )
            points = LidarPointCloud.from_file(
                str(bin_path),
                metainfo_filepath=str(metainfo_path) if metainfo_path else None,
            ).points.T
            csv_path = lidar_dir / f"{timestamp_ns}.csv"
            save_pointcloud_csv(csv_path, timestamp_ns, points)
            count += 1
            continue

        info_filename = concat_sample_data.info_filename
        if not info_filename:
            raise FileNotFoundError(
                f"LIDAR_CONCAT_INFO is required but missing in sample_data for "
                f"sample_data {concat_sample_data.token}"
            )

        info_path = seq_path / info_filename
        if not info_path.exists():
            raise FileNotFoundError(f"Required LIDAR_CONCAT_INFO file is missing: {info_path}")

        # Loading invokes PointCloud._validate_metainfo before slicing sources.
        pointcloud = LidarPointCloud.from_file(
            str(bin_path),
            metainfo_filepath=str(info_path),
        )
        source_pointcloud = pointcloud.split_by_sensor().get(sensor_token)

        source = next(
            (src for src in pointcloud.metainfo.sources if src.sensor_token == sensor_token),
            None,
        )

        # A zero-length source carries a zero stamp ({sec: 0, nanosec: 0}), so
        # fall back to the concat sweep's timestamp; sweeps are ~1e8 ns apart,
        # hence the file still sorts into its own frame position.
        timestamp_ns = (
            source.stamp.sec * 1_000_000_000 + source.stamp.nanosec
            if source is not None
            else None
        )
        if not timestamp_ns:
            timestamp_ns = int(concat_sample_data.timestamp) * 1000
        csv_path = lidar_dir / f"{timestamp_ns}.csv"

        if source_pointcloud is None or source_pointcloud.num_points() == 0:
            # The sensor contributed no points to this concat sweep (dropped
            # out, or started after the recording began). Still write a
            # header-only CSV: ensures the uploader still recognizes this frame even though it has no points.
            save_pointcloud_csv(csv_path, timestamp_ns, np.empty((0, 4), dtype=np.float32))
            blank_count += 1
            continue

        save_pointcloud_csv(csv_path, timestamp_ns, source_pointcloud.points.T)
        count += 1

    logger.info(
        f"{lidar_channel}: {count} point clouds extracted, {blank_count} blank frames written"
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
