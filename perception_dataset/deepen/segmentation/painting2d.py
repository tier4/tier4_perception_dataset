from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os.path as osp
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import zipfile

from PIL import Image
import numpy as np
from numpy.typing import NDArray
import pycocotools.mask as cocomask
import skimage

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation
from perception_dataset.deepen.deepen_to_t4_converter import LabelType

__all__ = ["DeepenSegmentationPainting2D"]


@dataclass
class DeepenSegmentationPainting2D(DeepenAnnotation):
    """A class to handle 2D segmentation annotation annotated by painting."""

    @classmethod
    def from_file(
        cls,
        ann_file: str,
        data_root: str,
        camera2index: Dict[str, int],
        data_corresponding: Dict[str, str],
        *,
        as_dict: bool = True,
    ) -> List[DeepenSegmentationPainting2D]:
        """Return a list of `DeepenSegmentationPainting2D`s from files.

        Args:
            ann_file (str): Annotation files path compressed as `.zip`.
            data_root (str): Root directory of the T4 dataset.
            camera2index (Dict[str, int]): Name mapping from camera name to camera index.
            as_dict (bool, optional): Whether to output objects as dict or its instance.
                Defaults to True.

        Returns:
            List[DeepenSegmentationPainting2D]: List of converted `DeepenSegmentationPainting2D`s.
        """
        # TODO: Execute preprocess first
        ann_dir = cls._preprocess(ann_file, data_root, camera2index)

        dataset_id = data_corresponding  # [] # TODO

        # Load metadata
        with open(osp.join(ann_dir, "metadata.json"), "r") as f:
            # {sensor_id: {filename: [categories...]}}
            metadata: Dict[str, Dict[str, List[str]]] = json.load(f)

        output: List[DeepenSegmentationPainting2D] = []
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
                    ann = DeepenSegmentationPainting2D(
                        dataset_id=dataset_id,
                        file_id=filename,
                        # TODO
                        # label_category_id=label_category_id,
                        # label_id=label_id,
                        label_type=LabelType.SEGMENTATION_2D.value,
                        sensor_id=sensor_id,
                        box=two_d_box,
                        two_d_mask=two_d_mask,
                    )
                    if as_dict:
                        output.append(ann.to_dict())
                    else:
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
) -> List[Tuple[List[float], Dict[str, Any]]]:
    """Return a set of boxes and masks of each instance.

    Args:
        mask (NDArray): Mask array.

    Returns:
        List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]]: Boxes (xmin, ymin, width, height)
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
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

        # RLE
        instance_mask = np.asfortranarray(label_mask == instance_id, dtype=np.uint8)
        rle = cocomask.decode(instance_mask)
        rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")

        output.append((bbox, rle))
    return output


def _format_annotation_directory(ann_file: str, data_root: Path) -> None:
    """
    Removes unnecessary directory levels and moves files up one directory.

    Args:
        ann_file (str): Annotation file name (.zip).
        data_root (Path): Root directory of annotation files.

    Details:
    ```
    - Before:
        data_root/
            ├── tmp/
            │   └── data_dir/
            │       ├── Semantic Segmentation - metadata.json
            │       ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
            │       ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
            │       ├── Semantic Segmentation - sensor2 - data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy
    - After:
        data_root/
            ├── metadata.json
            ├── CAM_TRAFFIC_LIGHT_NEAR/
            │   ├── data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
            │   ├── data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
            |   ...
            ├── SOME_CAMERA/
            │   ├── data_SOME_CAMERA_00000_jpg.npy
            ...
    ```
    """
    zip_file = data_root.joinpath(ann_file)
    temp_dir = tempfile.TemporaryDirectory(data_root.joinpath("tmp_annotation"))

    with zipfile.ZipFile(zip_file.as_posix()) as zf:
        zf.extractall(temp_dir.name)
