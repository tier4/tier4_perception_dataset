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


def compose_transform(trans1, rot1, trans2, rot2):
    """
    Compose two transforms: (trans1, rot1) followed by (trans2, rot2)
    rot: [w, x, y, z] format
    trans: [x, y, z]
    """
    # Convert quaternions to Rotation objects (scipy expects [x, y, z, w])
    r1 = Rotation.from_quat(rot1[1:] + [rot1[0]])
    r2 = Rotation.from_quat(rot2[1:] + [rot2[0]])

    # Compose rotations
    r = r2 * r1

    # Compose translations
    t = r2.apply(trans1) + trans2

    # Convert back to [w, x, y, z]
    quat = r.as_quat()  # [x, y, z, w]
    quat = [quat[3], quat[0], quat[1], quat[2]]

    return t.tolist(), quat
