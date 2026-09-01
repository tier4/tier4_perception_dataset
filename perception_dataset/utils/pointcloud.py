"""Point-cloud helpers for reading and exporting T4 LIDAR_CONCAT data."""

import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional

import numpy as np

from perception_dataset.constants import (
    LIDAR_CONCAT_CHANNEL,
    LIDAR_CONCAT_NUM_POINT_FEATURES,
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


def point_stride_from_info(bin_path: Path, total_points: int) -> int:
    """Derive the number of floats in each point record.

    T4 datasets ship LIDAR_CONCAT ``.pcd.bin`` files with varying per-point
    layouts (e.g. x,y,z,intensity,ring or x,y,z,intensity,ring,lidar_id,
    time_offset), so the stride must be derived per file rather than assumed.

    Args:
        bin_path (Path): Path to the binary point-cloud file.
        total_points (int): Number of points declared by ``LIDAR_CONCAT_INFO``.

    Returns:
        int: Number of 32-bit floats in each point record.

    Raises:
        ValueError: If the file size is inconsistent with ``total_points`` or
            the derived stride contains fewer than four floats.
    """
    size_bytes = bin_path.stat().st_size
    if size_bytes % 4 != 0:
        raise ValueError(f"{bin_path}: file size {size_bytes} bytes is not divisible by 4 (float32)")
    n_floats = size_bytes // 4
    stride, remainder = divmod(n_floats, total_points)
    if remainder != 0 or stride < 4:
        raise ValueError(
            f"{bin_path}: {n_floats} floats is not an integer multiple (>=4) of the "
            f"{total_points} points declared in LIDAR_CONCAT_INFO"
        )
    return stride


def detect_point_stride(floats: np.ndarray, bin_path: Path) -> int:
    """Guess the floats-per-point stride when no LIDAR_CONCAT_INFO is available.

    Tries ``LIDAR_CONCAT_NUM_POINT_FEATURES`` first, then other strides,
    accepting the first one that yields finite, plausibly-sized coordinates.

    Args:
        floats (np.ndarray): Flat array of point-cloud values.
        bin_path (Path): Source path included in diagnostics.

    Returns:
        int: Detected number of floats in each point record.

    Raises:
        ValueError: If no plausible point stride can be detected.
    """
    candidates = [LIDAR_CONCAT_NUM_POINT_FEATURES] + [
        stride for stride in range(4, 17) if stride != LIDAR_CONCAT_NUM_POINT_FEATURES
    ]
    for stride in candidates:
        if len(floats) == 0 or len(floats) % stride != 0:
            continue
        xyz = floats.reshape(-1, stride)[:, :3]
        if np.isfinite(xyz).all() and np.abs(xyz).max() < _MAX_REASONABLE_COORDINATE_M:
            if stride != LIDAR_CONCAT_NUM_POINT_FEATURES:
                logger.warning(
                    f"{bin_path}: detected {stride} floats per point "
                    f"(expected {LIDAR_CONCAT_NUM_POINT_FEATURES})"
                )
            return stride
    raise ValueError(f"{bin_path}: could not determine the point stride")


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
        ValueError: If a point-record stride cannot be derived or detected.
    """
    sensor_token = channel_to_token.get(lidar_channel)
    if sensor_token is None:
        logger.warning(f"LiDAR {lidar_channel} not found in {seq_path}; skipping")
        return

    lidar_dir = out_dir / "lidar" / lidar_channel
    lidar_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_record in frame_records:
        concat_sample_data = frame_record.get(LIDAR_CONCAT_CHANNEL)
        if concat_sample_data is None:
            continue

        bin_path = seq_path / concat_sample_data.filename
        if not bin_path.exists():
            raise FileNotFoundError(f"Required LIDAR_CONCAT point cloud is missing: {bin_path}")

        if lidar_channel == LIDAR_CONCAT_CHANNEL:
            timestamp_ns = int(concat_sample_data.timestamp) * 1000
            floats = np.fromfile(bin_path, dtype=np.float32)
            points = floats.reshape(-1, detect_point_stride(floats, bin_path))
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

        total_points = sum(int(src["length"]) for src in info["sources"])
        stride = point_stride_from_info(bin_path, total_points)
        bytes_per_point = stride * 4

        with open(bin_path, "rb") as f:
            f.seek(idx_begin * bytes_per_point)
            raw = f.read(length * bytes_per_point)

        timestamp_ns = stamp_to_ns(source.get("stamp"))
        if timestamp_ns is None:
            timestamp_ns = int(concat_sample_data.timestamp) * 1000

        points = np.frombuffer(raw, dtype=np.float32).reshape(-1, stride)
        csv_path = lidar_dir / f"{timestamp_ns}.csv"
        save_pointcloud_csv(csv_path, timestamp_ns, points)
        count += 1

    logger.info(f"{lidar_channel}: {count} point clouds extracted")


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


def stamp_to_ns(stamp: Optional[dict]) -> Optional[int]:
    """Convert a ROS-style timestamp to nanoseconds.

    Args:
        stamp (Optional[dict]): Mapping containing ``sec`` and ``nanosec``.

    Returns:
        Optional[int]: Timestamp in nanoseconds, or ``None`` when ``stamp`` is
            empty.
    """
    if not stamp:
        return None
    return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])


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
