"""Point-cloud helpers for the T4 → Kognic conversion pipeline."""

import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional

import numpy as np

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

# Binary format constants for LIDAR_CONCAT .bin files.
_LIDAR_CONCAT_CHANNEL = "LIDAR_CONCAT"
_NUM_POINT_FEATURES = 5  # x, y, z, intensity, auxiliary/ring_idx
_BYTES_PER_POINT = _NUM_POINT_FEATURES * 4  # float32


def extract_pointclouds(
    seq_path: Path,
    out_dir: Path,
    lidar_channel: str,
    frame_records: List[Dict[str, dict]],
    channel_to_token: Dict[str, str],
) -> None:
    """Write per-frame CSV point-cloud files for *lidar_channel*.

    Parameters
    ----------
    seq_path:
        Root path of the T4 sequence (source files are read from here).
    out_dir:
        Destination root; files are written to ``out_dir/lidar/<lidar_channel>/``.
    lidar_channel:
        Name of the LiDAR sensor channel to extract.
    frame_records:
        Ordered list of per-frame dicts mapping channel name → sample_data record.
    channel_to_token:
        Mapping from sensor channel name to sensor token.
    """
    sensor_token = channel_to_token.get(lidar_channel)
    if sensor_token is None:
        logger.warning(f"LiDAR {lidar_channel} not found in {seq_path}; skipping")
        return

    lidar_dir = out_dir / "lidar" / lidar_channel
    lidar_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_record in frame_records:
        concat_sample_data = frame_record.get(_LIDAR_CONCAT_CHANNEL)
        if concat_sample_data is None:
            continue

        bin_path = seq_path / concat_sample_data["filename"]
        if not bin_path.exists():
            raise FileNotFoundError(f"Required LIDAR_CONCAT point cloud is missing: {bin_path}")

        if lidar_channel == _LIDAR_CONCAT_CHANNEL:
            timestamp_ns = int(concat_sample_data["timestamp"]) * 1000
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, _NUM_POINT_FEATURES)
            csv_path = lidar_dir / f"{timestamp_ns}.csv"
            save_pointcloud_csv(csv_path, timestamp_ns, points)
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

        timestamp_ns = stamp_to_ns(source.get("stamp"))
        if timestamp_ns is None:
            timestamp_ns = int(concat_sample_data["timestamp"]) * 1000

        points = np.frombuffer(raw, dtype=np.float32).reshape(-1, _NUM_POINT_FEATURES)
        csv_path = lidar_dir / f"{timestamp_ns}.csv"
        save_pointcloud_csv(csv_path, timestamp_ns, points)
        count += 1

    logger.info(f"{lidar_channel}: {count} point clouds extracted")


def save_pointcloud_csv(csv_path: Path, timestamp_ns: int, points: np.ndarray) -> None:
    """Write *points* to a Kognic-compatible CSV at *csv_path*.

    The output columns are: ``ts_gps, x, y, z, intensity``.
    """
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


def stamp_to_ns(stamp: Optional[dict]) -> Optional[int]:
    """Convert a ROS-style ``{sec, nanosec}`` stamp dict to nanoseconds."""
    if not stamp:
        return None
    return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])


def copy_file(src: Path, dst: Path) -> None:
    """Copy *src* to *dst*, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
