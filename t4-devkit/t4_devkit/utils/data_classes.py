from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from nuscenes.utils.data_classes import Box

if TYPE_CHECKING:
    from t4_devkit.typing import RoiType

__all__ = ("Box3D", "Box2D")


class Box3D(Box):
    """An wrapper of NuScenes Box."""


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
        ----
            roi (RoiType): Roi elements, which is the order of (xmin, ymin, xmax, ymax).
            label (int, optional): Box label. Defaults to -1.
            score (float, optional): Box score. Defaults to np.nan.
            name (str | None, optional): Category name. Defaults to None.
            token (str | None, optional): Unique identifier token corresponding to `token` of `object_ann`.
                Defaults to None.
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
