"""Utilities for converting and composing three-dimensional transforms."""

from typing import List

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


def rotation_to_quaternion(rotation: List[float]) -> List[float]:
    """Convert Euler rotations in radians to a quaternion.

    Args:
        rotation (List[float]): A list of three floats [x, y, z] representing
            the rotation around the x, y, and z axes in radians.

    Returns:
        List[float]: A list of four floats [x, y, z, w] representing the quaternion.
    """
    r = Rotation.from_euler("xyz", rotation)
    return r.as_quat().tolist()


def quaternion_to_rotation(quaternion: List[float]) -> List[float]:
    """Convert a quaternion to Euler rotations in radians.

    Args:
        quaternion (List[float]): A list of four floats [x, y, z, w] representing
            the quaternion.

    Returns:
        List[float]: A list of three floats [x, y, z] representing the rotation
            around the x, y, and z axes in radians.
    """
    r = Rotation.from_quat(quaternion)
    return r.as_euler("xyz").tolist()


def compose_transform(trans1, rot1, trans2, rot2):
    """Compose two rigid transforms.

    The transform ``(trans1, rot1)`` is applied first, followed by
    ``(trans2, rot2)``. Translations use ``[x, y, z]`` order and rotations use
    ``[w, x, y, z]`` quaternion order.

    Args:
        trans1 (Sequence[float]): Translation of the first transform.
        rot1 (Sequence[float]): Quaternion of the first transform.
        trans2 (Sequence[float]): Translation of the second transform.
        rot2 (Sequence[float]): Quaternion of the second transform.

    Returns:
        Tuple[List[float], List[float]]: The composed translation and
            quaternion in ``[w, x, y, z]`` order.
    """
    # Convert quaternions to Rotation objects (scipy expects [x, y, z, w])
    r1 = Rotation.from_quat([rot1[1], rot1[2], rot1[3], rot1[0]])
    r2 = Rotation.from_quat([rot2[1], rot2[2], rot2[3], rot2[0]])

    # Compose rotations
    r = r2 * r1

    # Compose translations
    t = r2.apply(trans1) + trans2

    # Convert back to [w, x, y, z]
    quat = r.as_quat()  # [x, y, z, w]
    quat = [quat[3], quat[0], quat[1], quat[2]]

    return t.tolist(), quat


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion.

    This is the inverse of the rotation handling in :func:`transform_matrix`.

    Args:
        matrix (np.ndarray): A 3-by-3 rotation matrix.

    Returns:
        np.ndarray: Quaternion values in ``[w, x, y, z]`` order.
    """
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def transform_matrix(
    translation: np.ndarray = np.array([0, 0, 0]),
    rotation: Quaternion = Quaternion([1, 0, 0, 0]),
    inverse: bool = False,
) -> np.ndarray:
    """Convert a pose to a homogeneous transformation matrix.

    Adapted from the nuScenes devkit geometry utilities.

    Args:
        translation (np.ndarray): Translation in ``[x, y, z]`` order.
        rotation (Quaternion): Rotation quaternion in ``[w, x, y, z]`` order.
        inverse (bool): Whether to return the inverse transformation.

    Returns:
        np.ndarray: A 4-by-4 homogeneous transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm
