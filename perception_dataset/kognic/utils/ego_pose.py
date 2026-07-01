"""Ego-pose helpers for the T4 → Kognic conversion pipeline."""

from typing import Dict, List, Optional

import kognic.io.model as KognicModel
import numpy as np
from pyquaternion import Quaternion

from perception_dataset.constants import LIDAR_CONCAT_CHANNEL
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.transform import matrix_to_quaternion_wxyz, transform_matrix

logger = configure_logger(modname=__name__)


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
            frame_ego_tokens.append(sample_data.ego_pose_token)

    if not frame_ego_tokens:
        logger.warning("No ego poses found")
        return {}

    t0_inv = _build_transform(ego_pose_by_token[frame_ego_tokens[0]], inverse=True)

    ego_poses: Dict[str, KognicModel.EgoVehiclePose] = {}
    for frame_idx, ego_token in enumerate(frame_ego_tokens):
        transform = _build_transform(ego_pose_by_token[ego_token])
        relative_transform = t0_inv @ transform
        position = relative_transform[:3, 3]
        quat = matrix_to_quaternion_wxyz(relative_transform[:3, :3])

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
    if LIDAR_CONCAT_CHANNEL in frame_record:
        return frame_record[LIDAR_CONCAT_CHANNEL]

    for camera_channel in camera_channels:
        if camera_channel in frame_record:
            return frame_record[camera_channel]

    return next(iter(frame_record.values()), None)


def _build_transform(ego_pose, inverse: bool = False) -> np.ndarray:
    """Build a 4x4 transform from a T4 ego_pose record (rotation is ``[w, x, y, z]``)."""
    return transform_matrix(
        translation=np.array(ego_pose.translation),
        rotation=Quaternion(ego_pose.rotation),
        inverse=inverse,
    )
