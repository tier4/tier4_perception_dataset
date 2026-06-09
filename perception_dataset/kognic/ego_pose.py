"""Ego-pose helpers for the T4 → Kognic conversion pipeline."""

from typing import Dict, List, Optional

import kognic.io.model as KognicModel
import numpy as np
from scipy.spatial.transform import Rotation

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

# Channel name used to select the reference sample_data for each frame.
_LIDAR_CONCAT_CHANNEL = "LIDAR_CONCAT"


def extract_ego_poses(
    frame_records: List[Dict[str, dict]],
    ego_pose_by_token: Dict[str, dict],
    camera_channels,
) -> Dict[str, KognicModel.EgoVehiclePose]:
    """Compute relative ego poses for every frame in *frame_records*.

    Poses are expressed relative to the first frame (T0-normalised).

    Parameters
    ----------
    frame_records:
        Ordered list of per-frame dicts mapping channel name → sample_data record.
    ego_pose_by_token:
        Mapping from ego_pose token to ego_pose record.
    camera_channels:
        Ordered list of camera channel names; used to pick the reference
        sample_data when LIDAR_CONCAT is absent (first camera with data wins).
    """
    frame_ego_tokens: List[str] = []
    for frame_record in frame_records:
        sample_data = _get_reference_sample_data(frame_record, camera_channels)
        if sample_data is not None:
            frame_ego_tokens.append(sample_data["ego_pose_token"])

    if not frame_ego_tokens:
        logger.warning("No ego poses found")
        return {}

    t0 = _build_transform(ego_pose_by_token[frame_ego_tokens[0]])
    t0_inv = np.linalg.inv(t0)

    ego_poses: Dict[str, KognicModel.EgoVehiclePose] = {}
    for frame_idx, ego_token in enumerate(frame_ego_tokens):
        transform = _build_transform(ego_pose_by_token[ego_token])
        relative_transform = t0_inv @ transform
        position = relative_transform[:3, 3]
        quat = _matrix_to_quat_wxyz(relative_transform[:3, :3])

        ego_poses[str(frame_idx)] = KognicModel.EgoVehiclePose(
            position=KognicModel.Position(
                x=round(float(position[0]), 10),
                y=round(float(position[1]), 10),
                z=round(float(position[2]), 10),
            ),
            rotation=KognicModel.RotationQuaternion(
                w=round(float(quat[0]), 10),
                x=round(float(quat[1]), 10),
                y=round(float(quat[2]), 10),
                z=round(float(quat[3]), 10),
            ),
        )

    return ego_poses


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_reference_sample_data(
    frame_record: Dict[str, dict],
    camera_channels,
) -> Optional[dict]:
    """Pick the canonical sample_data record for a frame.

    Preference order: LIDAR_CONCAT → first camera with data → any channel.
    """
    if _LIDAR_CONCAT_CHANNEL in frame_record:
        return frame_record[_LIDAR_CONCAT_CHANNEL]

    for camera_channel in camera_channels:
        if camera_channel in frame_record:
            return frame_record[camera_channel]

    return next(iter(frame_record.values()), None)


def _build_transform(ego_pose: dict) -> np.ndarray:
    transform = np.eye(4)
    rotation = ego_pose["rotation"]
    if Rotation is not None:
        transform[:3, :3] = Rotation.from_quat(
            [rotation[1], rotation[2], rotation[3], rotation[0]]
        ).as_matrix()
    else:
        transform[:3, :3] = _quat_wxyz_to_matrix(rotation)
    transform[:3, 3] = ego_pose["translation"]
    return transform


def _quat_wxyz_to_matrix(quat: list) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    norm = w * w + x * x + y * y + z * z
    scale = 2.0 / norm

    wx, wy, wz = scale * w * x, scale * w * y, scale * w * z
    xx, xy, xz = scale * x * x, scale * x * y, scale * x * z
    yy, yz, zz = scale * y * y, scale * y * z, scale * z * z

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    if Rotation is not None:
        quat_xyzw = Rotation.from_matrix(matrix).as_quat()
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    return np.array([w, x, y, z], dtype=np.float64)
