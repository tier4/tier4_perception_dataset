from collections import defaultdict
import json
import os.path as osp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter
from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class FastLabel2dToT4Converter(DeepenToT4Converter):

    # Attribution mapping except of object's instance ID and occlusion state.
    ATTRIBUTE_MAPPING = {
        "frame_by_frame_left": "left_blinker",
        "frame_by_frame_right": "right_blinker",
        "frame_by_frame_brake": "brake_lamp",
        "facing": "vehicle_front_or_rear_or_side",
    }

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
                logger.error(f"{output_dir} already exists.")
                is_dir_exist = True
            else:
                is_dir_exist = False

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

    def _load_annotation_jsons(self):
        """Load annotations from all JSON files in the input directory and return as a dictionary."""
        anno_dict = {}
        for file in self._input_anno_files:
            with open(file) as f:
                anno_dict[file.name] = json.load(f)
        return anno_dict

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
                "annotations": [
                {
                    "points": [
                    1221.25,
                    488.44,
                    1275.38,
                    570.47
                    ],
                    "rotation": 0,
                    "autogenerated": false
                }
                ]
            },
        },
        ....
        ],
        ....
        """
        fl_annotations: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

        for filename, ann_list in sorted(annotations.items()):
            dataset_name: str = Path(filename).stem
            for ann in ann_list:
                filename: str = ann["name"].split("/")[-1]
                file_id: int = int(filename.split(".")[0])
                frame_no: int = file_id + 1
                camera: str = ann["name"].split("/")[-2]

                if dataset_name not in fl_annotations:
                    fl_annotations[dataset_name] = defaultdict(list)

                for a in ann["annotations"]:
                    attribute_names: list[str] = []

                    occlusion_state: str = "occlusion_state.none"
                    visibility: str = "Not available"
                    for att in a["attributes"]:
                        if att["key"] == "id":
                            instance_id = att["value"]
                        elif "occlusion_state" in att["key"]:
                            for v in att["value"]:
                                if frame_no in range(v[0], v[1]):
                                    occlusion_state = (
                                        "occlusion_state." + att["key"].split("_")[-1]
                                    )
                                    visibility = self._convert_occlusion_to_visibility(
                                        att["key"].split("_")[-1]
                                    )
                                    break
                        else:
                            attribute_names.append(
                                self.ATTRIBUTE_MAPPING[att["key"]] + "." + att["value"]
                            )
                    attribute_names.append(occlusion_state)

                    label_t4_dict: Dict[str, Any] = {
                        # NOTE: Some annotations are missing "title", use "value" instead
                        "category_name": (a["title"] if "title" in a else a["value"]),
                        "instance_id": instance_id,
                        "attribute_names": attribute_names,
                        "visibility_name": visibility,
                    }
                    two_d_box: list[float] = (
                        a["annotations"][0]["points"] if "annotations" in a else a["points"]
                    )
                    label_t4_dict.update(
                        {
                            "two_d_box": two_d_box,
                            "sensor_id": self._camera2idx[camera],
                        }
                    )
                    fl_annotations[dataset_name][file_id].append(label_t4_dict)

        return fl_annotations
