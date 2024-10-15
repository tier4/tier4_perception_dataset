from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os.path as osp
from typing import Any, Dict, List, Tuple, TypeVar

from PIL import Image
import numpy as np
import pycocotools.mask as cocomask

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation

Polygons2DLike = TypeVar("Polygon2DLike", List[List[List[float]]])


@dataclass
class DeepenSegmentationPolygon(DeepenAnnotation):

    @classmethod
    def from_file(
        cls,
        ann_file: str,
        data_root: str,
        camera2index: Dict[str, int],
    ) -> List[DeepenSegmentationPolygon]:
        """Return a list of `DeepenSegmentationPolygon`s from files.

        Args:
            ann_file (str): Annotation file (.json).
            data_dir (str): Root directory path of the T4 dataset.
            camera2index (Dict[str, int]):

        Returns:
            List[DeepenSegmentationPolygon]: List of converted `DeepenSegmentationPolygon`s.
        """
        with open(ann_file, "r") as f:
            data: Dict[str, Any] = json.load(f)

        labels: List[Dict[str, Any]] = data["labels"]

        output: List[DeepenSegmentationPolygon] = []
        for label in labels:
            # Extract required fields with defaults where appropriate
            dataset_id = label["dataset_id"]
            file_id = label["file_id"]
            label_category_id = label["label_category_id"]
            label_id = label["label_id"]
            label_type = "2d_segmentation"
            sensor_id = label["sensor_id"]
            labeller_email = label["labeller_email"]
            attributes = label.get("attributes", {})

            # Convert sensor_id to camera name
            camera_name: str | None = None
            for name, index in camera2index.items():
                if f"sensor_{index}" == sensor_id:
                    camera_name = name
                    break
            if camera_name is None:
                raise ValueError("There is no corresponding sensor ID.")

            # Get image size
            image = Image.open(osp.join(data_root, "data", camera_name, file_id))
            width, height = image.size

            # Extract polygons
            polygons = label["polygons"]
            two_d_box = _bbox_from_polygons(polygons)
            two_d_mask = _rle_from_polygons(polygons, width, height)

            ann = DeepenSegmentationPolygon(
                dataset_id=dataset_id,
                file_id=file_id,
                label_category_id=label_category_id,
                label_id=label_id,
                label_type=label_type,
                sensor_id=sensor_id,
                labeller_email=labeller_email,
                attributes=attributes,
                two_d_box=two_d_box,
                two_d_mask=two_d_mask,
            )
            output.append(ann)
        return output


def _bbox_from_polygons(polygons: Polygons2DLike) -> Tuple[float, float, float, float]:
    """Calculate bounding box corners from polygon vertices.

    Args:
        polygons (Polygons2DLike): 2D polygon vertices, such as `[[[x1, y1], [x2, y2], ...]]`.

    Returns:
        Tuple[float, float, float, float]: Bounding box corners, ordering `[xmin, ymin, xmax, ymax]`.
    """
    xy = np.array([point for polygon in polygons for point in polygon])

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)

    return xmin, ymin, xmax, ymax


def _rle_from_polygons(polygons: Polygons2DLike, width: int, height: int) -> Dict[str, Any]:
    """Encode polygons to RLE format mask.

    Args:
        polygons (Polygons2DLike): 2D polygon vertices.
        width (int): Image width.
        height (int): Image height.

    Returns:
        Dict[str, Any]: RLE format mask.
    """
    flattened = [coord for polygon in polygons for point in polygon for coord in point]

    rle_objects = cocomask.frPyObjects(flattened, height, width)
    rle = cocomask.merge(rle_objects)

    rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")

    return rle
