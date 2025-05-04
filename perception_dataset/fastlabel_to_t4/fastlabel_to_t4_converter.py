from collections import defaultdict
import os.path as osp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from perception_dataset.constants import LABEL_PATH_ENUM
from perception_dataset.fastlabel_to_t4.fastlabel_2d_to_t4_converter import (
    FastLabel2dToT4Converter,
)
from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from perception_dataset.t4_dataset.resolver.keyframe_consistency_resolver import (
    KeyFrameConsistencyResolver,
)
from perception_dataset.utils.label_converter import LabelConverter
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.transform import rotation_to_quaternion

logger = configure_logger(modname=__name__)


class FastLabelToT4Converter(FastLabel2dToT4Converter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        input_anno_base: str,
        overwrite_mode: bool,
        description: Dict[str, Dict[str, str]],
        make_t4_dataset_dir: bool = True,
        input_bag_base: Optional[str] = None,
        topic_list: Optional[Union[Dict[str, List[str]], List[str]]] = None,
    ):
        super().__init__(
            input_base,
            output_base,
            input_anno_base,
            None,
            overwrite_mode,
            description,
            input_bag_base,
            topic_list,
        )
        self._make_t4_dataset_dir = make_t4_dataset_dir
        self._label_converter = LabelConverter(
            label_path=LABEL_PATH_ENUM.OBJECT_LABEL,
            attribute_path=LABEL_PATH_ENUM.ATTRIBUTE,
        )

    def convert(self) -> None:
        t4_datasets = sorted([d.name for d in self._input_base.iterdir() if d.is_dir()])
        anno_jsons_dict = self._load_annotation_jsons(t4_datasets, ".pcd")
        fl_annotations = self._format_fastlabel_3d_annotation(anno_jsons_dict)

        for t4dataset_name in t4_datasets:
            # Skip if the dataset is not in the annotation jsons
            if t4dataset_name not in fl_annotations.keys():
                logger.warning(f"{t4dataset_name} not in annotation jsons.")
                continue

            # Check if input directory exists
            input_dir = self._input_base / t4dataset_name
            input_annotation_dir = input_dir / "annotation"
            if not osp.exists(input_annotation_dir):
                logger.warning(f"input_dir {input_dir} not exists.")
                continue

            # Check if output directory already exists
            output_dir = self._output_base / t4dataset_name
            if self._make_t4_dataset_dir:
                output_dir = output_dir / "t4_dataset"
            if self._input_bag_base is not None:
                input_bag_dir = Path(self._input_bag_base) / t4dataset_name

            if osp.exists(output_dir):
                logger.warning(f"{output_dir} already exists.")
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

            if t4dataset_name not in fl_annotations.keys():
                logger.warning(f"No annotation for {t4dataset_name}")
                continue

            # Start updating annotations
            annotation_files_generator = AnnotationFilesGenerator(
                description=self._description, label_coordinates="lidar"
            )
            annotation_files_generator.convert_one_scene(
                input_dir=input_dir,
                output_dir=output_dir,
                scene_anno_dict=fl_annotations[t4dataset_name],
                dataset_name=t4dataset_name,
            )

            # fix non-keyframe (no-labeled frame) in t4 dataset
            modifier = KeyFrameConsistencyResolver()
            modifier.inspect_and_fix_t4_segment(Path(output_dir))

    def _format_fastlabel_3d_annotation(self, annotations: Dict[str, List[Dict[str, Any]]]):
        """
        e.g. of input_anno_file(fastlabel):
        [
            {
                "id": "675f15cb-f3c1-45df-b8e1-6daaa36402bd",
                "name": "r7ZRDFWf_2024-08-21T15-13-16+0900_10/00011.pcd",
                "status": "completed",
                "externalStatus": "approved",
                "url": "https://annotations.fastlabel.ai/workspaces/......",
                "annotations": [
                    {
                        "id": "9feb60dc-6170-4c2c-95d7-165b7862d12a",
                        "type": "cuboid",
                        "title": "truck",
                        "value": "truck",
                        "color": "#D10069",
                        "attributes": [
                            {
                                "type": "radio",
                                "name": "status",
                                "key": "status",
                                "title": "approval",
                                "value": "approval"
                            },
                            {
                                "type": "radio",
                                "name": "occlusion_state",
                                "key": "occlusion_state",
                                "title": "none",
                                "value": "none"
                            },
                            {
                                "type": "radio",
                                "name": "vehicle_state",
                                "key": "vehicle_state",
                                "title": "driving",
                                "value": "driving"
                            }
                        ],
                        "points": [
                            8.76,   // coordinate x
                            3.87,   // coordinate y
                            1.71,   // coordinate z
                            0.00,   // rotation x
                            0.00,   // rotation y
                            -0.03,  // rotation z
                            8.27,   // length x
                            2.47,   // length y
                            3.17    // length z
                        ],
                        "rotation": 0,
                        "keypoints": [],
                        "confidenceScore": -1
                    },
                    ....
                ]
            },
            ...
        ]
        """
        fl_annotations: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

        for filename, ann_list in sorted(annotations.items()):
            dataset_name: str = Path(filename).stem
            for ann in ann_list:
                filename: str = ann["name"].split("/")[-1]
                file_id: int = int(filename.split(".")[0])

                if dataset_name not in fl_annotations:
                    fl_annotations[dataset_name] = defaultdict(list)

                for a in ann["annotations"]:
                    visibility: str = "Not available"
                    instance_id = a["id"]

                    if "attributes" not in a or a["attributes"] is None:
                        logger.error(f"No attributes in {a}")
                    attributes = []
                    for att in a["attributes"]:
                        if att["name"] == "status":
                            continue
                        attributes.append(f"{att['name'].lower()}.{att['value']}")
                        if att["key"] == "occlusion_state":
                            visibility = self._convert_occlusion_to_visibility(att["value"])
                    category = self._label_converter.convert_label(a["title"])
                    label_t4_dict: Dict[str, Any] = {
                        "category_name": category,
                        "instance_id": instance_id,
                        "attribute_names": attributes,
                        "visibility_name": visibility,
                    }
                    points = a["points"]
                    q = rotation_to_quaternion(a["points"][3:6])
                    label_t4_dict.update(
                        {
                            "three_d_bbox": {
                                "translation": {
                                    "x": points[0],
                                    "y": points[1],
                                    "z": points[2],
                                },
                                "velocity": None,
                                "acceleration": None,
                                "size": {
                                    "length": points[6],
                                    "width": points[7],
                                    "height": points[8],
                                },
                                "rotation": {
                                    "x": q[0],
                                    "y": q[1],
                                    "z": q[2],
                                    "w": q[3],
                                },
                            },
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        }
                    )
                    fl_annotations[dataset_name][file_id].append(label_t4_dict)

        return fl_annotations
