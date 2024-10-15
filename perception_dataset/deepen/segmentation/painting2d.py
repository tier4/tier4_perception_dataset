from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os.path as osp
from typing import Any, Dict, List, Tuple

from PIL import Image
import numpy as np
from numpy.typing import NDArray
import pycocotools.mask as cocomask
import skimage

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation


@dataclass
class DeepenSegmentationPainting(DeepenAnnotation):
    """A class to handle 2D segmentation annotation annotated by painting."""

    @classmethod
    def from_file(
        cls,
        ann_file: str,
        data_root: str,
        camera2index: Dict[str, int],
        data_corresponding: Dict[str, str],
    ) -> List[DeepenSegmentationPainting]:
        """
        Args:
            ann_file (str): Annotation files path compressed as `.zip`.
            data_root (str): Root directory of the T4 dataset.
            camera2index (Dict[str, int]): Name mapping from camera name to camera index.

        Returns:
            List[DeepenSegmentationPainting]: List of converted `DeepenSegmentationPainting`s.
        """
        # TODO: Execute preprocess first
        ann_dir = cls._preprocess(ann_file, data_root, camera2index)

        dataset_id = data_corresponding  # [] # TODO

        # Load metadata
        with open(osp.join(ann_dir, "metadata.json"), "r") as f:
            # {sensor_id: {filename: [categories...]}}
            metadata: Dict[str, Dict[str, List[str]]] = json.load(f)

        output: List[DeepenSegmentationPainting] = []
        for sensor_id, meta in metadata.items():
            # Convert sensor_id to camera name
            camera_name: str | None = None
            for name, index in camera2index.items():
                if f"sensor_{index}" == sensor_id:
                    camera_name = name
                    break
            if camera_name is None:
                raise ValueError("There is no corresponding sensor ID.")

            for filename, categories in meta.items():
                # Get image size
                image = Image.open(osp.join(data_root, "data", camera_name, filename))
                width, height = image.size

                # Load mask from .npy
                mask: NDArray = np.load(ann_dir, camera_name, filename.replace(".jpg", "_jpg.npy"))
                mask = mask.reshape((height, width))

                bbox_and_rle_list = _bbox_and_rle_from_mask(mask)
                for two_d_box, two_d_mask in bbox_and_rle_list:
                    ann = DeepenSegmentationPainting(
                        dataset_id=dataset_id,
                        file_id=filename,
                        # TODO
                        # label_category_id=label_category_id,
                        # label_id=label_id,
                        label_type="2d_segmentation",
                        sensor_id=sensor_id,
                        two_d_box=two_d_box,
                        two_d_mask=two_d_mask,
                    )
                    output.append(ann)
        return output

    @staticmethod
    def _preprocess(ann_file: str, data_root: str, camera2index: Dict[str, int]) -> str:
        """Execute preprocess for annotation files.

        Args:
            ann_file (str): Annotation files path compressed as `.zip`.
            data_root (str): Root directory of the T4 dataset.
            camera2index (Dict[str, int]): Name mapping from camera name to camera index.

        Returns:
            str: Directory path of annotation files.
        """
        # TODO
        pass


def _bbox_and_rle_from_mask(
    mask: NDArray,
) -> List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]]:
    """Return a set of boxes and masks of each instance.

    Args:
        mask (NDArray): Mask array.

    Returns:
        List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]]: Boxes (xmin, ymin, xmax, ymax)
            and RLE masks.
    """
    label_mask = skimage.measure.label(mask, connectivity=1)
    num_instances = label_mask.max()

    output = []
    for instance_id in range(1, num_instances + 1):
        # bbox
        indices = np.where(label_mask == instance_id)
        xmin, ymin = np.min(indices, axis=1)
        xmax, ymax = np.max(indices, axis=1)
        bbox = (xmin, ymin, xmax, ymax)

        # RLE
        instance_mask = np.asfortranarray(label_mask == instance_id, dtype=np.uint8)
        rle = cocomask.decode(instance_mask)
        rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")

        output.append((bbox, rle))
    return output
