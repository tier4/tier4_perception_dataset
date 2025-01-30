from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os.path as osp
from typing import Any, Dict, List, Optional

from PIL import Image
import numpy as np
import pycocotools.mask as cocomask

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation, LabelType

__all__ = ["DeepenSegmentationPolygon2D"]


Polygons2DLike = list[list[list[float]]]


@dataclass
class DeepenSegmentationPolygon2D(DeepenAnnotation):

    @classmethod
    def from_file(
        cls,
        ann_file: str,
        data_root: str,
        camera2index: Dict[str, int],
        dataset_corresponding: Dict[str, str],
        *,
        as_dict: bool = True,
    ) -> List[DeepenSegmentationPolygon2D | Dict[str, Any]]:
        """Return a list of `DeepenSegmentationPolygon2D`s from files.

        Args:
            ann_file (str): Annotation file (.json).
            data_dir (str): Root directory path of the T4 dataset.
            camera2index (Dict[str, int]): Key-value mapping of camera and its index.
            dataset_corresponding (Dict[str, str]): Key-value mapping of T4 dataset name and Deepen ID.
            as_dict (bool, optional): Whether to output objects as dict or its instance.
                Defaults to True.

        Returns:
            List[DeepenSegmentationPolygon2D | Dict[str, Any]]: If `as_dict=False`,
                returns a list of converted `DeepenSegmentationPolygon2D`s.
                Otherwise, returns a list of converted dicts.
        """
        with open(ann_file, "r") as f:
            data: Dict[str, Any] = json.load(f)

        labels: List[Dict[str, Any]] = data["labels"]

        output: List[DeepenSegmentationPolygon2D] = []
        for label in labels:
            # Extract required fields with defaults where appropriate
            dataset_id = label["dataset_id"]
            file_id = label["file_id"]
            label_category_id = label["label_category_id"]
            label_id = label["label_id"]
            label_type = LabelType.SEGMENTATION_2D.value
            sensor_id = label["sensor_id"]
            labeller_email = label["labeller_email"]
            attributes = label.get("attributes", {})

            # Convert sensor_id to camera name
            camera_name: str | None = None
            # NOTE: camera2index starts from 0, but deepen sensor id starts from 1.
            for name, index in camera2index.items():
                if f"sensor{index + 1}" == sensor_id:
                    camera_name = name
                    t4_sensor_id: str = f"sensor{index}"
                    break
            if camera_name is None:
                raise ValueError("There is no corresponding sensor ID.")

            dataset_name: Optional[str] = None
            for t4_name, deepen_id in dataset_corresponding.items():
                if deepen_id == dataset_id:
                    dataset_name = t4_name
                    break

            if dataset_name is None:
                raise ValueError(f"There is no T4 dataset corresponding to {dataset_id}.")

            # Get image size
            image = Image.open(osp.join(data_root, dataset_name, "data", camera_name, file_id))
            width, height = image.size

            # Extract polygons
            polygons = label["polygons"]
            two_d_box = _bbox_from_polygons(polygons)
            two_d_mask = _rle_from_polygons(polygons, width, height)

            ann = DeepenSegmentationPolygon2D(
                dataset_id=dataset_id,
                file_id=file_id,
                label_category_id=label_category_id,
                label_id=label_id,
                label_type=label_type,
                sensor_id=t4_sensor_id,
                labeller_email=labeller_email,
                attributes=attributes,
                box=two_d_box,
                two_d_mask=two_d_mask,
            )
            if as_dict:
                output.append(ann.to_dict())
            else:
                output.append(ann)
        return output


def _bbox_from_polygons(polygons: Polygons2DLike) -> List[float]:
    """Calculate bounding box corners from polygon vertices.

    Args:
        polygons (Polygons2DLike): 2D polygon vertices, such as `[[[x1, y1], [x2, y2], ...]]`.

    Returns:
        List[float]: Bounding box corners, ordering `[xmin, ymin, width, height]`.
    """
    xy = np.array([point for polygon in polygons for point in polygon])

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)

    return [xmin, ymin, xmax - xmin, ymax - ymin]


def _rle_from_polygons(polygons: Polygons2DLike, width: int, height: int) -> Dict[str, Any]:
    """Encode polygons to RLE format mask.

    Args:
        polygons (Polygons2DLike): 2D polygon vertices.
        width (int): Image width.
        height (int): Image height.

    Returns:
        Dict[str, Any]: RLE format mask.
    """
    flattened = [[coord for point in polygon for coord in point] for polygon in polygons]

    rle_objects = cocomask.frPyObjects(flattened, height, width)
    rle_hw = cocomask.merge(rle_objects)

    mask_hw = cocomask.decode(rle_hw)
    rle_wh = cocomask.encode(np.asfortranarray(mask_hw.T))
    rle_wh["counts"] = base64.b64encode(rle_wh["counts"]).decode("ascii")

    return rle_wh
