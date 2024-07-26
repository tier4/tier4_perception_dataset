from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from t4_devkit.schema import VisibilityLevel

if TYPE_CHECKING:
    from t4_devkit.typing import NDArrayF64

    from .box import Box3D


__all__ = ("view_points", "is_box_in_image")


def view_points(
    points: NDArrayF64,
    intrinsic: NDArrayF64,
    distortion: NDArrayF64 | None = None,
    *,
    normalize: bool = True,
) -> NDArrayF64:
    """Project 3d points on a 2d plane. It can be used to implement both perspective and orthographic projections.

    It first applies the dot product between the points and the view.

    Args:
        points (NDArrayF64): Matrix of points, which is the shape of (3, n) and (x, y, z) is along each column.
        intrinsic (NDArrayF64): nxn camera intrinsic matrix (n <= 4).
        distortion (NDArrayF64 | None, optional): Camera distortion coefficients, which is the shape of (n,) (n >= 5).
        normalize (bool, optional): Whether to normalize the remaining coordinate (along the 3rd axis).

    Returns:
        Projected points in the shape of (3, n). If `normalize=False`, the 3rd coordinate is the height.
    """
    assert intrinsic.shape[0] <= 4
    assert intrinsic.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    nbr_points = points.shape[1]

    points = np.concatenate((points, np.ones((1, nbr_points))))

    if distortion is not None:
        assert distortion.shape[0] >= 5
        D = distortion
        # distortion is [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
        while len(D) < 12:
            D = np.insert(D, len(D), 0)
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = D

        x_ = points[0]
        y_ = points[1]
        r2 = x_**2 + y_**2
        f1 = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
        f2 = x_ * y_
        x__ = x_ * f1 + 2 * p1 * f2 + p2 * (r2 + 2 * x_**2) + s1 * r2 + s2 * r2**2
        y__ = y_ * f1 + p1 * (r2 + 2 * y_**2) + 2 * p2 * f2 + s3 * r2 + s4 * r2**2
        u = viewpad[0, 0] * x__ + viewpad[0, 2]
        v = viewpad[1, 1] * y__ + viewpad[1, 2]
        points = np.stack([u, v, points[2, :]], axis=0)
    else:
        points = np.dot(viewpad, points)
        points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def is_box_in_image(
    box: Box3D,
    intrinsic: NDArrayF64,
    img_size: tuple[int, int],
    visibility: VisibilityLevel = VisibilityLevel.NONE,
) -> bool:
    """Check if a box is visible inside of an image without considering its occlusions.

    Args:
        box (Box3D): The box to be checked.
        intrinsic (NDArrayF64): 3x3 camera intrinsic matrix.
        img_size (tuple[int, int]): Image size in the order of (width, height).
        visibility (VisibilityLevel, optional): Enum member of VisibilityLevel.

    Returns:
        Return True if visibility condition is satisfied.
    """
    corners_3d = box.corners()
    corners_on_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    img_w, img_h = img_size
    is_visible = np.logical_and(corners_on_img[0, :] > 0, corners_on_img[0, :] < img_w)
    is_visible = np.logical_and(is_visible, corners_on_img[1, :] < img_h)
    is_visible = np.logical_and(is_visible, corners_on_img[1, :] > 0)
    is_visible = np.logical_and(is_visible, corners_on_img[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of camera.

    if visibility == VisibilityLevel.FULL:
        return all(is_visible) and all(in_front)
    elif visibility in (VisibilityLevel.MOST, VisibilityLevel.PARTIAL):
        return any(is_visible)
    elif visibility == VisibilityLevel.NONE:
        return True
    else:
        raise ValueError(f"Unexpected visibility: {visibility}")
