from __future__ import annotations

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation

from .base import DeepenSegmentation2D


class DeepenSegmentationPainting(DeepenSegmentation2D):
    """A class to handle 2D segmentation annotation annotated by painting."""

    def __init__(
        self,
        input_anno_file: str,
        input_base: str,
        dataset_corresponding: Dict[str, str],
    ) -> None:
        super().__init__(input_anno_file, input_base, dataset_corresponding)

    def to_bbox(self, mask: NDArray) -> tuple[int, int, int, int]:
        """Convert painting mask to bounding box using min/max vertices.

        Args:
            mask (NDArray): Segmentation mask filling painted are by 1.

        Returns:
            Tuple[float, float, float, float]: Bounding box corners, ordering (xmin, ymin, xmax, ymax).
        """
        indices = np.where(mask == 1)

        xmin, ymin = np.min(indices, axis=1)
        xmax, ymax = np.max(indices, axis=1)

        return xmin, ymin, xmax, ymax

    def to_deepen_annotations(self) -> List[DeepenAnnotation]:
        annotations = []
        return annotations
