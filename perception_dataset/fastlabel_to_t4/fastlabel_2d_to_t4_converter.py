import base64
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os.path as osp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pycocotools.mask as cocomask
from tqdm import tqdm

from perception_dataset.constants import LABEL_PATH_ENUM
from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter
from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from perception_dataset.utils.label_converter import LabelConverter
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

Points2DLike = list[list[list[float]]]


class FastLabel2dToT4Converter(DeepenToT4Converter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        input_anno_base: str,
        dataset_corresponding: Dict[str, int],
        overwrite_mode: bool,
        description: Dict[str, Dict[str, str]],
        input_bag_base: Optional[str],
        topic_list: Union[Dict[str, List[str]], List[str]],
    ):
        super().__init__(
            input_base,
            output_base,
            input_anno_base,
            dataset_corresponding=None,
            overwrite_mode=overwrite_mode,
            description=description,
            input_bag_base=input_bag_base,
            topic_list=topic_list,
            t4_dataset_dir_name=None,
            ignore_interpolate_label=False,
        )
        self._input_base = Path(input_base)
        self._output_base = Path(output_base)
        self._t4dataset_name_to_merge: Dict[str, str] = dataset_corresponding
        self._camera2idx = description.get("camera_index")
        self._input_anno_files: List[Path] = []
        for f in Path(input_anno_base).rglob("*.json"):
            self._input_anno_files.append(f)
        self._label_converter = LabelConverter(
            label_path=LABEL_PATH_ENUM.OBJECT_LABEL,
            attribute_path=LABEL_PATH_ENUM.ATTRIBUTE,
        )

    def convert(self):
        # Load and format Fastlabel annotations
        anno_jsons_dict = self._load_annotation_jsons()
        fl_annotations = self._format_fastlabel_annotation(anno_jsons_dict)

        for t4dataset_name in self._t4dataset_name_to_merge.keys():
            # Check if input directory exists
            input_dir = self._input_base / t4dataset_name
            input_annotation_dir = input_dir / "annotation"
            if not osp.exists(input_annotation_dir):
                logger.warning(f"input_dir {input_dir} not exists.")
                continue

            # Check if output directory already exists
            output_dir = self._output_base / t4dataset_name
            output_dir = output_dir / "t4_dataset"
            if self._input_bag_base is not None:
                input_bag_dir = Path(self._input_bag_base) / t4dataset_name
            if osp.exists(output_dir):
                logger.warning(f"{output_dir} already exists.")
                is_dir_exist = True
            if self._overwrite_mode or not is_dir_exist:
                # Remove existing output directory
                shutil.rmtree(output_dir, ignore_errors=True)
                # Copy input data to output directory
                self._copy_data(input_dir, output_dir)
                # Make rosbag
                if self._input_bag_base is not None and not osp.exists(
                    osp.join(output_dir, "input_bag")
                ):
                    self._find_start_end_time(input_dir)
                    self._make_rosbag(str(input_bag_dir), str(output_dir))
            else:
                raise ValueError("If you want to overwrite files, use --overwrite option.")

            # Start converting annotations
            annotation_files_generator = AnnotationFilesGenerator(description=self._description)
            annotation_files_generator.convert_one_scene(
                input_dir=input_dir,
                output_dir=output_dir,
                scene_anno_dict=fl_annotations[t4dataset_name],
                dataset_name=t4dataset_name,
            )

    def _load_annotation_jsons(
        self, t4_datasets: Optional[List[str]] = None, delimiter: Optional[str] = None
    ) -> Dict[str, List[dict[str, Any]]]:
        """Load annotations from all JSON files in the input directory and return as a dictionary."""
        logger.info("Loading annotation JSON files")
        anno_dict = defaultdict(list)
        if t4_datasets is None:
            for file in self._input_anno_files:
                with open(file) as f:
                    anno_dict[file.name] = json.load(f)
                    return anno_dict

        pbar = tqdm(total=len(self._input_anno_files), desc="Loading annotation files")
        for file in self._input_anno_files:
            pbar.update(1)
            t4_dataset_name = file.name.split(delimiter)[0]
            for dataset in t4_datasets:
                if dataset == t4_dataset_name:
                    break
            else:
                continue
            with open(file) as f:
                one_label = json.load(f)
                anno_dict[dataset].extend(one_label)
        pbar.close()
        return anno_dict

    def _process_annotation(self, dataset_name, annotation):
        filename: str = annotation["name"].split("/")[-1]
        file_id: int = int(filename.split(".")[0])
        frame_no: int = file_id + 1
        camera = annotation["name"].split("/")[-2]

        width = annotation["width"]
        height = annotation["height"]

        labels = []
        for a in annotation["annotations"]:
            occlusion_state = "occlusion_state.none"
            visibility = "Not available"
            instance_id = ""

            for att in a["attributes"]:
                if att["key"] == "id":
                    instance_id = att["value"]
                if "occlusion_state" in att["key"]:
                    for v in att["value"]:
                        if frame_no in range(v[0], v[1]):
                            occlusion_state = "occlusion_state." + att["key"].split("_")[-1]
                            visibility = self._convert_occlusion_to_visibility(
                                att["key"].split("_")[-1]
                            )
                            break

            category_label = self._label_converter.convert_label(a["title"])
            label_t4_dict = {
                "category_name": category_label,
                "instance_id": instance_id,
                "attribute_names": [occlusion_state],
                "visibility_name": visibility,
            }

            if a["type"] == "bbox":
                label_t4_dict.update(
                    {
                        "two_d_box": a["points"],
                        "sensor_id": self._camera2idx[camera],
                    }
                )
            elif a["type"] == "segmentation":
                label_t4_dict.update(
                    {
                        "two_d_segmentation": _rle_from_points(a["points"], width, height),
                        "sensor_id": self._camera2idx[camera],
                    }
                )
                if (
                    self._label_converter.is_object_label(category_label)
                    and category_label not in self._surface_categories
                ):
                    label_t4_dict["two_d_box"] = _convert_polygon_to_bbox(a["points"][0][0])
            labels.append(label_t4_dict)

        return dataset_name, file_id, labels

    def _format_fastlabel_annotation(self, annotations: Dict[str, List[Dict[str, Any]]]):
        """
        e.g. of input_anno_file(fastlabel):
        "DBv2.0_1-1.json": [
        {
            "name": "CAM_BACK/0.png",
            "width": 1440,
            "height": 1080,
            "annotations": [
            {
                "type": "bbox",
                "title": "pedestrian.adult",
                "value": "adult",
                "color": "#91EEFB",
                "attributes": [
                {
                    "type": "text",
                    "name": "ID",
                    "key": "id",
                    "value": "cc0cc1eb4cc15118d10ee42f0426cc00"
                },
                {
                    "type": "frameSelect",
                    "name": "Occlusion_State_none",
                    "key": "occlusion_state_none",
                    "value": []
                },
                {
                    "type": "frameSelect",
                    "name": "Occlusion_State_partial",
                    "key": "occlusion_state_partial",
                    "value": [
                    [
                        1,
                        9
                    ]
                    ]
                },
                {
                    "type": "frameSelect",
                    "name": "Occlusion_State_most",
                    "key": "occlusion_state_most",
                    "value": [
                    [
                        10,
                        19
                    ]
                    ]
                }
                ],
                "points": [
                    1221.25,
                    488.44,
                    1275.38,
                    570.47
                ],
                "rotation": 0,
                "autogenerated": false
            },
        },
        ....
        ],
        ....
        """
        fl_annotations: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

        with ProcessPoolExecutor() as executor:
            futures = []
            for filename, ann_list in sorted(annotations.items()):
                dataset_name: str = Path(filename).stem
                for ann in ann_list:
                    futures.append(executor.submit(self._process_annotation, dataset_name, ann))

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing {dataset_name} labels",
                ):
                    dataset_name, file_id, labels = future.result()
                    if dataset_name not in fl_annotations:
                        fl_annotations[dataset_name] = defaultdict(list)
                    for label_t4_dict in labels:
                        fl_annotations[dataset_name][file_id].append(label_t4_dict)

        return fl_annotations


def _rle_from_points(points: Points2DLike, width: int, height: int) -> Dict[str, Any]:
    """Encode points to RLE format mask.

    Points format of 2D segmentation in FastLabel:
    ```
    "points": [
        [
            [...], // outer points (1)
            [...], // hollowed out points (1)
        ],
        [
            [...], // outer points (2)
            [...], // hollowed out points (2)
        ]
    ],
    ```

    Args:
        points (Points2DLike): 2D points, such as `[[[o1, o1, o2, o2, ...], [ho1, ho1, ho2, ho2, ...], ...]]`.
        width (int): Image width.
        height (int): Image height.

    Returns:
        Dict[str, Any]: RLE format mask.
    """
    final_mask = np.zeros((height, width, 1), dtype=np.uint8)

    for polygon in points:
        outer_polygon = polygon[0]  # outer points

        outer_rle = cocomask.frPyObjects([outer_polygon], height, width)
        outer_mask = cocomask.decode(outer_rle)
        combined_mask = outer_mask
        for i in range(1, len(polygon)):
            hollow_polygon = polygon[i]  # hollowed out points
            hollow_rle = cocomask.frPyObjects([hollow_polygon], height, width)
            hollow_mask = cocomask.decode(hollow_rle)
            combined_mask = combined_mask - hollow_mask
        final_mask = np.maximum(final_mask, combined_mask)
    # encode RLE
    rle = cocomask.encode(np.asfortranarray(np.squeeze(final_mask)))
    rle["counts"] = base64.b64encode(rle["counts"]).decode("ascii")
    return rle


def _convert_polygon_to_bbox(polygon: List[int]) -> List[float]:
    """Convert polygon points to bounding box.

    Args:
        polygon: 2D points, such as `[x1, y1, x2, y2 ....]`.

    Returns:
        List[float]: Bounding box in [x1, y1, x2, y2] format.
    """
    x_coords = polygon[0::2]
    y_coords = polygon[1::2]

    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    return [xmin, ymin, xmax, ymax]
