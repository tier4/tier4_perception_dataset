import json
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List

from nuimages import NuImages
from tqdm import tqdm

from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter
from perception_dataset.t4_dataset.classes.attribute import AttributeTable
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class T4dataset2DAttributeMerger(DeepenToT4Converter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        input_anno_base: str,
        overwrite_mode: bool,
        dataset_corresponding: Dict[str, int],
        description: Dict[str, Dict[str, str]],
    ):
        self._input_base = Path(input_base)
        self._output_base = Path(output_base)
        self._input_anno_files: List[Path] = []
        for f in Path(input_anno_base).rglob("*.json"):
            self._input_anno_files.append(f)
        self._overwrite_mode: bool = overwrite_mode
        self._t4dataset_name_to_merge: Dict[str, str] = dataset_corresponding
        self._description: Dict[str, Dict[str, str]] = description
        self._ignore_interpolate_label: bool = True

        # Initialize attribute table with empty values
        self._attribute_table = AttributeTable(
            name_to_description={},
            default_value="",
        )

    def convert(self):
        # Load Deepen annotation from JSON file
        deepen_anno_json_dict = self._load_deepen_annotations()
        deepen_anno_json = self._format_2d_annotation(deepen_anno_json_dict)

        # Format Deepen annotation into a more usable structure
        scenes_anno_dict: Dict[str, Dict[str, Any]] = self._format_deepen_annotation(
            deepen_anno_json["labels"], self._description["camera_index"]
        )

        for t4dataset_name, dataset_id in self._t4dataset_name_to_merge.items():
            input_dir = self._input_base / t4dataset_name
            output_dir = self._output_base / t4dataset_name

            if not input_dir.exists():
                logger.warning(f"{input_dir} does not exist")
                continue

            is_dir_exist = output_dir.exists()
            if self._overwrite_mode or not is_dir_exist:
                shutil.rmtree(output_dir, ignore_errors=True)
                self._copy_data(input_dir, output_dir)
            else:
                raise ValueError("If you want to overwrite files, use --overwrite option.")

            # Start merging attributes
            nuim = NuImages(
                version="annotation", dataroot=self._input_base / t4dataset_name, verbose=False
            )
            out_object_ann, out_attribute = nuim.object_ann, nuim.attribute

            for each_object_ann in tqdm(out_object_ann):
                # Find corresponding annotation
                max_iou_anno = self._find_corresponding_annotation(
                    nuim, each_object_ann, scenes_anno_dict[dataset_id]
                )
                if max_iou_anno is None:
                    continue
                # Get category name
                category_name: str = nuim.get("category", each_object_ann["category_token"])[
                    "name"
                ].lower()

                # Append attribute
                self._update_attribute_table(max_iou_anno, out_attribute)
                self._update_object_annotation(
                    each_object_ann, max_iou_anno, out_attribute, category_name
                )

            # Save modified data to files
            object_ann_filename = output_dir / "annotation" / "object_ann.json"
            attribute_filename = output_dir / "annotation" / "attribute.json"

            with open(object_ann_filename, "w") as f:
                json.dump(out_object_ann, f, indent=4)

            with open(attribute_filename, "w") as f:
                json.dump(out_attribute, f, indent=4)

    def _load_deepen_annotation(self):
        with open(self._input_anno_file) as f:
            return json.load(f)

    def _load_deepen_annotations(self):
        """Load Deepen annotations from all JSON files in the input directory and return as a dictionary."""
        deepen_anno_dict = {}
        for file in self._input_anno_files:
            with open(file) as f:
                deepen_anno_dict[file.name] = json.load(f)
        return deepen_anno_dict

    def _format_2d_annotation(self, deepen_anno_json_dict):
        """Format 2D annotations from Deepen json in order to correspond with 3D Deepen json."""
        formatted_deepen_anno_dict = {"labels": []}
        for file_name, deepen_anno_json in deepen_anno_json_dict.items():
            for anno in deepen_anno_json["labels"]:
                anno["dataset_id"] = file_name
            formatted_deepen_anno_dict["labels"].extend(deepen_anno_json["labels"])
        return formatted_deepen_anno_dict

    def _find_corresponding_annotation(
        self, nuim: NuImages, object_ann: Dict[str, Any], scenes_anno: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find corresponding annotation in Deepen annotation
        Args:
            nuim (NuImages): NuImages object
            object_ann (Dict[str, Any]): object annotation
            scenes_anno (Dict[str, Any]): Deepen annotation
        """
        filename = nuim.get("sample_data", object_ann["sample_data_token"])["filename"]
        camera_name = filename.split("/")[1]
        frame_no = int(re.findall(r"\d+", filename.split("/")[2])[0])

        # find largest IoU annotation
        frame_annotations = [
            a
            for a in scenes_anno[frame_no]
            if a["sensor_id"] == self._description["camera_index"][camera_name]
        ]
        max_iou = 0
        max_iou_anno = None
        for anno in frame_annotations:
            iou = self._get_IoU(object_ann["bbox"], anno["two_d_box"])
            # NOTE: 0.75 is defined empirically
            if iou > max_iou and iou > 0.75:
                max_iou = iou
                max_iou_anno = anno

        return max_iou_anno

    def _get_IoU(self, bbox1: List[float], bbox2: List[float]):
        """
        Calculate IoU between two bounding boxes
        Args:
            bbox1 (List[float]): bounding box 1
            bbox2 (List[float]): bounding box 2
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        if x1 >= x2 or y1 >= y2:
            return 0
        else:
            intersection = (x2 - x1) * (y2 - y1)
            union = (
                (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                - intersection
            )
            return intersection / union

    def _update_attribute_table(self, max_iou_anno, out_attribute):
        attribute_names = [a["name"] for a in out_attribute]
        for attr_name in max_iou_anno["attribute_names"]:
            if attr_name not in attribute_names:
                out_attribute.append(
                    {
                        "token": self._attribute_table.get_token_from_name(name=attr_name),
                        "name": attr_name,
                        "description": "",
                    }
                )

    def _update_object_annotation(
        self, each_object_ann, max_iou_anno, out_attribute, category_name
    ):
        for attr_name in max_iou_anno["attribute_names"]:
            # Ignore pedestrian and cyclist for turn signal/brake lamp attributes
            if "pedestrian" in category_name or "bicycle" in category_name:
                if (
                    "turn_signal" in attr_name
                    or "blinker" in attr_name
                    or "brake_lamp" in attr_name
                ):
                    continue
            # update object_ann
            token = [a["token"] for a in out_attribute if a["name"] == attr_name][0]
            each_object_ann["attribute_tokens"].append(token)
