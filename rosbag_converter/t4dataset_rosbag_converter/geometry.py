from __future__ import annotations

from dataclasses import dataclass
from math import cos
from math import sin
from pathlib import Path

import numpy as np
import yaml
from geometry_msgs.msg import TransformStamped


@dataclass(frozen=True)
class RigidTransform:
    parent: str
    child: str
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def inverse(self) -> "RigidTransform":
        matrix = np.linalg.inv(self.matrix())
        return rigid_from_matrix(self.child, self.parent, matrix)

    def matrix(self) -> np.ndarray:
        qx, qy, qz, qw = self.rotation_xyzw
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )
        matrix[:3, 3] = np.asarray(self.translation, dtype=np.float64)
        return matrix

    def to_msg(self) -> TransformStamped:
        msg = TransformStamped()
        msg.header.frame_id = self.parent
        msg.child_frame_id = self.child
        msg.transform.translation.x = float(self.translation[0])
        msg.transform.translation.y = float(self.translation[1])
        msg.transform.translation.z = float(self.translation[2])
        msg.transform.rotation.x = float(self.rotation_xyzw[0])
        msg.transform.rotation.y = float(self.rotation_xyzw[1])
        msg.transform.rotation.z = float(self.rotation_xyzw[2])
        msg.transform.rotation.w = float(self.rotation_xyzw[3])
        return msg


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = cos(roll * 0.5), sin(roll * 0.5)
    cp, sp = cos(pitch * 0.5), sin(pitch * 0.5)
    cy, sy = cos(yaw * 0.5), sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    return qx / norm, qy / norm, qz / norm, qw / norm


def quat_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    r = matrix[:3, :3]
    trace = float(np.trace(r))
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = (1.0 + r[0, 0] - r[1, 1] - r[2, 2]) ** 0.5 * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = (1.0 + r[1, 1] - r[0, 0] - r[2, 2]) ** 0.5 * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = (1.0 + r[2, 2] - r[0, 0] - r[1, 1]) ** 0.5 * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    return qx / norm, qy / norm, qz / norm, qw / norm


def rigid_from_matrix(parent: str, child: str, matrix: np.ndarray) -> RigidTransform:
    return RigidTransform(
        parent=parent,
        child=child,
        translation=tuple(float(v) for v in matrix[:3, 3]),
        rotation_xyzw=quat_from_matrix(matrix),
    )


def transform_from_yaml(parent: str, child: str, values: dict) -> RigidTransform:
    return RigidTransform(
        parent=parent,
        child=child,
        translation=(float(values["x"]), float(values["y"]), float(values["z"])),
        rotation_xyzw=quat_from_rpy(
            float(values.get("roll", 0.0)),
            float(values.get("pitch", 0.0)),
            float(values.get("yaw", 0.0)),
        ),
    )


def compose(parent_to_mid: RigidTransform, mid_to_child: RigidTransform) -> RigidTransform:
    if parent_to_mid.child != mid_to_child.parent:
        raise ValueError(
            f"Cannot compose {parent_to_mid.parent}->{parent_to_mid.child} with "
            f"{mid_to_child.parent}->{mid_to_child.child}"
        )
    return rigid_from_matrix(
        parent_to_mid.parent,
        mid_to_child.child,
        parent_to_mid.matrix() @ mid_to_child.matrix(),
    )


def identity(parent: str, child: str) -> RigidTransform:
    return RigidTransform(parent, child, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def load_transform_yaml(path: Path) -> dict[str, dict[str, RigidTransform]]:
    with path.open("r") as fp:
        raw = yaml.safe_load(fp) or {}
    transforms: dict[str, dict[str, RigidTransform]] = {}
    for parent, children in raw.items():
        transforms[parent] = {
            child: transform_from_yaml(parent, child, values)
            for child, values in (children or {}).items()
        }
    return transforms
