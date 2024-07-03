from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

if TYPE_CHECKING:
    from t4_devkit.typing import RoiType

__all__ = ("Box3D", "Box2D")


class Box3D(Box):
    """An wrapper of NuScenes Box."""

    def __init__(
        self,
        center: List[float],
        size: List[float],
        orientation: Quaternion,
        label: int = np.nan,
        score: float = np.nan,
        velocity: Tuple = (np.nan, np.nan, np.nan),
        name: str = None,
        token: str = None,
    ) -> None:
        """Construct instance.

        Args:
            center (List[float]): Center of box given as (x, y, z).
            size (List[float]): Size of box given as (width, length, height).
            orientation (Quaternion): Box orientation.
            label (int, optional): Integer label.
            score (float, optional): Classification score.
            velocity (Tuple, optional): Box velocity given as (vx, vy, vz).
            name (str, optional): Box category name.
            token (str, optional): Unique string identifier.
        """
        super().__init__(center, size, orientation, label, score, velocity, name, token)


class Box2D:
    """A class to represent 2D box."""

    def __init__(
        self,
        roi: RoiType,
        label: int = -1,
        score: float = np.nan,
        name: str | None = None,
        token: str | None = None,
    ) -> None:
        """Construct instance.

        Args:
            roi (RoiType): Roi elements, which is the order of (xmin, ymin, xmax, ymax).
            label (int, optional): Box label.
            score (float, optional): Box score.
            name (str | None, optional): Category name.
            token (str | None, optional): Unique identifier token corresponding to `token` of `object_ann`.
        """
        self.xmin, self.ymin, self.xmax, self.ymax = roi
        self.label = int(label)
        self.score = float(score) if not np.isnan(score) else score
        self.name = name
        self.token = token

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin
