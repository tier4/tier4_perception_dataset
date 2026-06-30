"""Calibration helpers for the T4 → Kognic conversion pipeline."""

from pathlib import Path
from typing import Dict, Tuple

import kognic.io.model as KognicModel

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


def extract_calibration(
    channel_to_token: Dict[str, str],
    calib_by_sensor_token: Dict[str, dict],
    camera_channels,
    lidar_channels,
    sample_data_by_channel: Dict[str, list],
    seq_path: Path,
) -> Dict[str, KognicModel.BaseCalibration]:
    """Build a Kognic calibration dict for all cameras and LiDARs in the scene.

    Parameters
    ----------
    channel_to_token:
        Mapping from sensor channel name to sensor token.
    calib_by_sensor_token:
        Mapping from sensor_token to calibrated_sensor record.
    camera_channels:
        Ordered list of camera channel names to include.
    lidar_channels:
        List of LiDAR channel names to include.
    sample_data_by_channel:
        Mapping from channel name to sorted list of sample_data records.
    seq_path:
        Root path of the T4 sequence (used to locate image files on disk).
    """
    calibration: Dict[str, KognicModel.BaseCalibration] = {}

    for camera_channel in camera_channels:
        sensor_token = channel_to_token.get(camera_channel)
        if sensor_token is None:
            logger.warning(f"Camera {camera_channel} not found in {seq_path}; skipping")
            continue
        if not _has_existing_channel_file(sample_data_by_channel, seq_path, camera_channel):
            logger.warning(f"Camera {camera_channel} has no files in {seq_path}; skipping")
            continue

        calib = calib_by_sensor_token[sensor_token]
        translation = calib.translation
        rotation = calib.rotation  # [w, x, y, z]
        intrinsic = calib.camera_intrinsic
        distortion = calib.camera_distortion
        width, height = read_image_dims(sample_data_by_channel, seq_path, camera_channel)

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

    for lidar_channel in lidar_channels:
        if lidar_channel not in channel_to_token:
            logger.warning(f"LiDAR {lidar_channel} not found in {seq_path}; skipping")
            continue
        calibration[lidar_channel] = KognicModel.LidarCalibration(
            position=KognicModel.Position(x=0.0, y=0.0, z=0.0),
            rotation_quaternion=KognicModel.RotationQuaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )

    return calibration


def read_image_dims(
    sample_data_by_channel: Dict[str, list],
    seq_path: Path,
    camera_channel: str,
) -> Tuple[int, int]:
    """Return ``(width, height)`` for *camera_channel*.

    First tries the metadata recorded in the sample_data records; falls back to
    opening the first JPEG on disk via Pillow for malformed T4 archives that
    omit image dimensions.
    """
    for sample_data in sample_data_by_channel.get(camera_channel, []):
        width = int(sample_data.width or 0)
        height = int(sample_data.height or 0)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_existing_channel_file(
    sample_data_by_channel: Dict[str, list],
    seq_path: Path,
    channel: str,
) -> bool:
    return any(
        (seq_path / sample_data.filename).exists()
        for sample_data in sample_data_by_channel.get(channel, [])
    )
