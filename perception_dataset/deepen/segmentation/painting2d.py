from __future__ import annotations

import base64
from dataclasses import dataclass
from glob import glob
import json
import os.path as osp
import shutil
from typing import Any, Dict, List, Tuple

from PIL import Image
import numpy as np
from numpy.typing import NDArray
import pycocotools.mask as cocomask
import skimage

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation, LabelType

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
        dataset_corresponding: Dict[str, str],
        *,
        as_dict: bool = True,
    ) -> List[DeepenSegmentationPainting2D]:
        """Return a list of `DeepenSegmentationPainting2D`s from files.

        Args:
            ann_file (str): Annotation files path compressed as `.zip`.
            data_root (str): Root directory of the T4 dataset.
            camera2index (Dict[str, int]): Name mapping from camera name to camera index.
            dataset_corresponding (Dict[str, str]): Key-value mapping of T4 dataset name and Deepen ID.
            as_dict (bool, optional): Whether to output objects as dict or its instance.
                Defaults to True.

        Returns:
            List[DeepenSegmentationPainting2D]: List of converted `DeepenSegmentationPainting2D`s.
        """
        output: List[DeepenSegmentationPainting2D] = []
        for t4_dataset_name, dataset_id in dataset_corresponding.items():

            # Parse annotation ZIP file
            annotations = _parse_annotation(ann_file, camera2index)

            for ann_info, id2category in annotations:
                # Get image size
                image = Image.open(osp.join(data_root, t4_dataset_name, ann_info.image))
                width, height = image.size

                # Load mask from .npy
                mask: NDArray = np.load(ann_info.mask)
                mask = mask.reshape((height, width))

                # Get sensor ID
                sensor_id = f"sensor{camera2index[ann_info.camera_name]}"

                instances = _mask_to_instances(mask, id2category)
                for instance_info in instances:
                    ann = DeepenSegmentationPainting2D(
                        dataset_id=dataset_id,
                        file_id=osp.basename(ann_info.image),
                        label_category_id=instance_info.category,
                        label_id=f"{instance_info.category}:{instance_info.instance_id}",
                        label_type=LabelType.SEGMENTATION_2D.value,
                        sensor_id=sensor_id,
                        box=instance_info.bbox,
                        two_d_mask=instance_info.mask,
                    )
                    if as_dict:
                        output.append(ann.to_dict())
                    else:
                        output.append(ann)
        return output


@dataclass
class AnnInfo:
    image: str
    mask: str
    camera_name: str
    categories: List[str]


@dataclass
class InstanceInfo:
    bbox: tuple[float, float, float, float]
    mask: Dict[str, Any]
    category: str
    instance_id: int


def _parse_annotation(
    ann_file: str,
    camera2index: Dict[str, int],
) -> List[Tuple[AnnInfo, Dict[str, int]]]:
    """Execute preprocess for annotation files.

    Annotation file details:
    ```
    - Directory structure in .zip file:
        annotation.zip/
            ├── tmp/
            │   └── data_dir/
            │       ├── Semantic Segmentation - metadata.json
            │       ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00000_jpg.npy
            │       ├── Semantic Segmentation - sensor1 - data_CAM_TRAFFIC_LIGHT_NEAR_00001_jpg.npy
            │       ├── Semantic Segmentation - sensor2 - data_CAM_TRAFFIC_LIGHT_FAR_00000_jpg.npy
    ```

    Args:
        ann_file (str): Annotation file path compressed as `.zip`.
        data_root (str): Root directory of the T4 dataset.
        camera2index (Dict[str, int]): Name mapping from camera name to camera index.

    Returns:
        List[Tuple[AnnInfo, Dict[str, int]]]: List of pairs of annotation information and
            key-value mapping of category ID and its name at each image.
    """
    in_ann_dir = osp.dirname(ann_file)
    out_ann_dir = osp.join(in_ann_dir, "deepen_annotation")
    shutil.unpack_archive(ann_file, out_ann_dir)

    metadata_filepath = glob(osp.join(out_ann_dir, "**/*metadata.json"), recursive=True)[0]
    with open(metadata_filepath) as f:
        metadata: Dict[str, Dict[str, List[str]]] = json.load(f)

    npy_filepaths = glob(osp.join(out_ann_dir, "**/*.npy"), recursive=True)

    output: List[Tuple[AnnInfo, Dict[int, str]]] = []
    for sensor_id, items in metadata.items():
        for image, categories in items.items():
            id2category = {i + 1: name for i, name in enumerate(categories)}
            camera_name = None
            for name, sensor_idx in camera2index.items():
                if f"sensor{sensor_idx}" == sensor_id:
                    camera_name = name
            if camera_name is None:
                raise ValueError(f"There is no corresponding camera for {image}")

            # data_<CAMERA_NAME>_xxx.jpt -> xxx.jpg
            image_filename = image.replace("data_", "").replace(f"{camera_name}_", "")

            candidate_npy = image.replace(".jpg", "_jpg.npy")
            for npy_filepath in npy_filepaths:
                if candidate_npy in osp.basename(npy_filepath):
                    info = AnnInfo(
                        image=osp.join("data", camera_name, image_filename),
                        mask=npy_filepath,
                        camera_name=camera_name,
                        categories=categories,
                    )
                    output.append((info, id2category))
                    break
    return output


def _mask_to_instances(
    mask: NDArray,
    id2category: Dict[int, str],
) -> List[InstanceInfo]:
    """Return a set of boxes and masks of each instance.

    Args:
        mask (NDArray): Mask array.
        id2category (Dict[int, str]): Key-value mapping of category ID and its name.

    Returns:
        List[InstanceInfo]: List of `InstanceInfo` objects.
    """
    label_mask = skimage.measure.label(mask, connectivity=1)
    num_instances = label_mask.max()

    output = []
    for instance_id in range(1, num_instances + 1):
        # bbox
        indices = np.where(label_mask == instance_id)
        xmin, ymin = np.min(indices, axis=1)
        xmax, ymax = np.max(indices, axis=1)

        # NOTE: json.dumps raise TypeError for Numpy.int64
        bbox = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]

        unique, freq = np.unique(mask[indices], return_counts=True)
        category_id = unique[np.argmax(freq)]
        category = id2category[category_id]

        # RLE
        instance_mask = np.asfortranarray(label_mask == instance_id, dtype=np.uint8)
        rle = cocomask.encode(instance_mask)
        rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")

        info = InstanceInfo(bbox=bbox, mask=rle, category=category, instance_id=instance_id)
        output.append(info)

    return output
