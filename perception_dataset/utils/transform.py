from typing import List

from scipy.spatial.transform import Rotation


def rotation_to_quaternion(rotation: List[float]) -> List[float]:
    """
    Convert a rotation vector (in radians) to a quaternion.

    Args:
        rotation (List[float]): A list of three floats [x, y, z] representing
            the rotation around the x, y, and z axes in radians.

    Returns:
        List[float]: A list of four floats [x, y, z, w] representing the quaternion.
    """
    r = Rotation.from_euler("xyz", rotation)
    return r.as_quat().tolist()


def quaternion_to_rotation(quaternion: List[float]) -> List[float]:
    """
    Convert a quaternion to a rotation vector (in radians).

    Args:
        quaternion (List[float]): A list of four floats [x, y, z, w] representing
            the quaternion.

    Returns:
        List[float]: A list of three floats [x, y, z] representing the rotation
            around the x, y, and z axes in radians.
    """
    r = Rotation.from_quat(quaternion)
    return r.as_euler("xyz").tolist()
