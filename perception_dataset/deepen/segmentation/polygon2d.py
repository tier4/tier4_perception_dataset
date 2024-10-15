from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation

from .base import DeepenSegmentation2D
from .encode import encode_polygons


def load_json(filepath: Path) -> Any:
    if not filepath.is_file():
        raise FileNotFoundError(f"Could not find a file: {filepath}")

    with filepath.open("r") as f:
        data = json.load(f)

    return data


class DeepenSegmentationPolygon(DeepenSegmentation2D):
    def __init__(
        self,
        input_anno_file: str,
        input_base: str,
        dataset_corresponding: Dict[str, str],
    ) -> None:
        super().__init__(input_anno_file, input_base, dataset_corresponding)

        self.annotations: Dict[str, Any] = load_json(input_anno_file)
        self.width, self.height = self._get_image_size()  # TODO

    def _get_image_size(self) -> Tuple[int]:
        labels: List[Dict[str, Any]] = self.annotations.get("labels", [])
        if labels:
            pass
        else:
            raise ValueError("No labels found in the annotation data.")

    def to_bbox(
        self,
        polygons: List[List[List[float]]],
    ) -> Tuple[float, float, float, float]:
        """Convert polygons to bounding box using min/max vertices.

        Args:
            polygons (List[List[List[float]]]): List of polygon vertices.

        Returns:
            Tuple[float, float, float, float]: Bounding box corners, ordering (xmin, ymin, xmax, ymax).
        """
        xy = [(point[0], point[1]) for polygon in polygons for point in polygon]

        xmin, ymin = np.min(xy, axis=0)
        xmax, ymax = np.max(xy, axis=0)

        return xmin, ymin, xmax, ymax

    def to_deepen_annotations(self) -> List[DeepenAnnotation]:
        annotations = []
        labels: List[Dict[str, Any]] = self.annotations.get("labels", [])
        for label in labels:
            data = label.copy()

            # insert label_type
            data["label_type"] = "2d_segmentation"

            # update "polygons" to "two_d_box" and "two_d_mask"
            polygons = data.pop("polygons")
            data["two_d_box"] = self.to_bbox(polygons)
            data["two_d_mask"] = encode_polygons(polygons, self.width, self.height)

            annotations.append(DeepenAnnotation.from_dict(data))

        return annotations
