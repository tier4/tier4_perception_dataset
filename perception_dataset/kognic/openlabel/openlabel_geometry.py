"""Geometry helpers shared by the T4 <-> Kognic OpenLABEL converters.

``T4ToOpenLabelConverter`` writes T4 boxes out as Kognic cuboids and
``OpenLabelToT4Converter`` reads them back; the forward/inverse transforms
live here so the two converters cannot drift apart.

A Kognic cuboid ``val`` is ``[x, y, z, qx, qy, qz, qw, sx, sy, sz]`` expressed
in the per-frame ego/base_link frame, yaw 0 facing +y. T4 boxes are in the
global frame, face +x, and carry size as ``[width, length, height]`` with a
wxyz quaternion.
"""

from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Kognic cuboids face +y at yaw 0 while T4/nuScenes boxes face +x.
ROTATION_T4_TO_KOGNIC = Rotation.from_euler("z", -90, degrees=True)


def quat_wxyz_to_xyzw(quat: list) -> List[float]:
    """Reorder a ``[w, x, y, z]`` quaternion to scipy's ``[x, y, z, w]``."""
    return [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]


def t4_box_to_cuboid_val(annotation: dict, ego_pose: dict) -> List[float]:
    """Build the 10-float Kognic cuboid from a global-frame T4 box."""
    rotation_ego = Rotation.from_quat(quat_wxyz_to_xyzw(ego_pose["rotation"]))
    position = rotation_ego.inv().apply(
        np.asarray(annotation["translation"], dtype=np.float64)
        - np.asarray(ego_pose["translation"], dtype=np.float64)
    )

    rotation_box = Rotation.from_quat(quat_wxyz_to_xyzw(annotation["rotation"]))
    qx, qy, qz, qw = (rotation_ego.inv() * rotation_box * ROTATION_T4_TO_KOGNIC).as_quat()

    width, length, height = (float(v) for v in annotation["size"])
    return [
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(qx),
        float(qy),
        float(qz),
        float(qw),
        width,
        length,
        height,
    ]


def cuboid_val_to_t4_box(
    val: List[float], ego_pose: dict, iso_rotated_cuboids: bool = False
) -> Tuple[List[float], List[float], List[float]]:
    """Undo :func:`t4_box_to_cuboid_val`.

    ``val`` is ``[x, y, z, qx, qy, qz, qw, sx, sy, sz]`` in the per-frame
    ego/base_link frame. Returns ``(translation, size, rotation)`` in the T4
    global frame, with rotation as a wxyz quaternion and size as
    ``[width, length, height]``. When *iso_rotated_cuboids* is true the cuboids
    already face +x (T4 convention) so the yaw correction is skipped.
    """
    rotation_ego = Rotation.from_quat(quat_wxyz_to_xyzw(ego_pose["rotation"]))

    position_ego = np.asarray(val[0:3], dtype=np.float64)
    translation = rotation_ego.apply(position_ego) + np.asarray(
        ego_pose["translation"], dtype=np.float64
    )

    rotation_cuboid = Rotation.from_quat([val[3], val[4], val[5], val[6]])
    if iso_rotated_cuboids:
        rotation_box = rotation_ego * rotation_cuboid
    else:
        rotation_box = rotation_ego * rotation_cuboid * ROTATION_T4_TO_KOGNIC.inv()
    qx, qy, qz, qw = rotation_box.as_quat()

    width, length, height = (float(v) for v in val[7:10])
    return (
        [float(translation[0]), float(translation[1]), float(translation[2])],
        [width, length, height],
        [float(qw), float(qx), float(qy), float(qz)],
    )
