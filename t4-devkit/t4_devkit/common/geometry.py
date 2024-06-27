from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from nuscenes.utils.geometry_utils import view_points
from t4_devkit.schema import VisibilityLevel

if TYPE_CHECKING:
    from t4_devkit.typing import NDArrayF64

    from .box import Box3D


__all__ = ("is_box_in_image",)


def is_box_in_image(
    box: Box3D,
    intrinsic: NDArrayF64,
    img_size: tuple[int, int],
    visibility: VisibilityLevel = VisibilityLevel.NONE,
) -> bool:
    """Check if a box is visible inside of an image without considering its occlusions.

    Args:
    ----
        box (Box3D): The box to be checked.
        intrinsic (NDArrayF64): 3x3 camera intrinsic matrix.
        img_size (tuple[int, int]): Image size in the order of (width, height).
        visibility (VisibilityLevel, optional): Enum member of VisibilityLevel.
            Defaults to VisibilityLevel.NONE.

    Returns:
    -------
        bool: Return True if visibility condition is satisfied.
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
