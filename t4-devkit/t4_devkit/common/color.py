from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

if TYPE_CHECKING:
    from numbers import Number

    from t4_devkit.typing import ArrayLike, NDArrayF64


def distance_color(
    distances: Number | ArrayLike,
    cmap: str | None = None,
    v_min: float = 3.0,
    v_max: float = 75.0,
) -> tuple[float, float, float] | NDArrayF64:
    """Return color map depending on distance values.

    Args:
        distances (Number | ArrayLike): Array of distances in the shape of (N,).
        cmap (str | None, optional): Color map name in matplotlib. If None, `turbo_r` will be used.
        v_min (float, optional): Min value to normalize.
        v_max (float, optional): Max value to normalize.

    Returns:
        Color map in the shape of (N,). If input type is any number, returns a color as
            `tuple[float, float, float]`. Otherwise, returns colors as `NDArrayF64`.
    """
    color_map = matplotlib.colormaps["turbo_r"] if cmap is None else matplotlib.colormaps[cmap]
    norm = matplotlib.colors.Normalize(v_min, v_max)
    return color_map(norm(distances))
