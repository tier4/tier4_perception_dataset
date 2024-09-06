from collections import defaultdict
import json
import os.path as osp
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Union

from nuscenes.nuscenes import NuScenes

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.rosbag2.rosbag2_converter import Rosbag2Converter
from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from perception_dataset.t4_dataset.keyframe_consistency_resolver import KeyFrameConsistencyResolver
from perception_dataset.utils.logger import configure_logger
import perception_dataset.utils.misc as misc_utils

logger = configure_logger(modname=__name__)


class DeepenToT4Converter(AbstractConverter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        input_anno_file: str,
        dataset_corresponding: Dict[str, str],
        overwrite_mode: bool,
        description: Dict[str, Dict[str, str]],
        input_bag_base: Optional[str],
        topic_list: Union[Dict[str, List[str]], List[str]],
        t4_dataset_dir_name: str = "t4_dataset",
        ignore_interpolate_label: bool = False,
    ):
        super().__init__(input_base, output_base)

        self._input_anno_file: str = input_anno_file
        self._t4data_name_to_deepen_dataset_id: Dict[str, str] = dataset_corresponding
        self._overwrite_mode: bool = overwrite_mode
        self._description: Dict[str, Dict[str, str]] = description
        self._input_bag_base: Optional[str] = input_bag_base
        self._t4_dataset_dir_name: str = t4_dataset_dir_name
        self._start_sec: float = 0
        self._end_sec: float = 1e10
        self._ignore_interpolate_label: bool = ignore_interpolate_label

        self._topic_list_yaml: Union[List, Dict] = topic_list

    def convert(self):
        with open(self._input_anno_file) as f:
            deepen_anno_json = json.load(f)

        # format deepen annotation
        scenes_anno_dict: Dict[str, Dict[str, Any]] = self._format_deepen_annotation(
            deepen_anno_json["labels"], self._description["camera_index"]
        )

        # copy data and make time/topic filtered rosbag from non-annotated-t4-dataset and rosbag
        for t4data_name in self._t4data_name_to_deepen_dataset_id:
            is_dir_exist: bool = False
            output_dir = osp.join(self._output_base, t4data_name, self._t4_dataset_dir_name)
            input_dir = osp.join(self._input_base, t4data_name)
            if self._input_bag_base is not None:
                input_bag_dir = osp.join(self._input_bag_base, t4data_name)
            if osp.exists(output_dir):
                logger.error(f"{output_dir} already exists.")
                is_dir_exist = True

            if self._overwrite_mode or not is_dir_exist:
                shutil.rmtree(output_dir, ignore_errors=True)
                self._copy_data(input_dir, output_dir)
                if self._input_bag_base is not None:
                    self._find_start_end_time(input_dir)
                    self._make_rosbag(input_bag_dir, output_dir)
            else:
                raise ValueError("If you want to overwrite files, use --overwrite option.")

        # insert annotation to t4 dataset
        for t4data_name, dataset_id in self._t4data_name_to_deepen_dataset_id.items():
            output_dir = osp.join(self._output_base, t4data_name, self._t4_dataset_dir_name)
            input_dir = osp.join(self._input_base, t4data_name)
            annotation_files_generator = AnnotationFilesGenerator(description=self._description)
            annotation_files_generator.convert_one_scene(
                input_dir=input_dir,
                output_dir=output_dir,
                scene_anno_dict=scenes_anno_dict[dataset_id],
                dataset_name=t4data_name,
            )

        # fix non-keyframe (no-labeled frame) in t4 dataset
        for t4data_name in self._t4data_name_to_deepen_dataset_id.keys():
            output_dir = osp.join(self._output_base, t4data_name, self._t4_dataset_dir_name)
            modifier = KeyFrameConsistencyResolver()
            modifier.inspect_and_fix_t4_segment(Path(output_dir))

    def _copy_data(self, input_dir: str, output_dir: str):
        if input_dir != output_dir:
            logger.info(f"Copying {input_dir} to {output_dir} ... ")
            if osp.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.copytree(input_dir, output_dir)
            logger.info("Done!")

    def _find_start_end_time(self, t4_dataset_dir):
        nusc = NuScenes(version="annotation", dataroot=t4_dataset_dir, verbose=False)
        end_nusc_timestamp = 0
        for frame_index, sample in enumerate(nusc.sample):
            if frame_index == 0:
                self._start_sec = (
                    misc_utils.nusc_timestamp_to_unix_timestamp(sample["timestamp"]) - 2.0
                )
            if sample["timestamp"] > end_nusc_timestamp:
                end_nusc_timestamp = sample["timestamp"]
        self._end_sec = misc_utils.nusc_timestamp_to_unix_timestamp(end_nusc_timestamp) + 2.0

    def _make_rosbag(self, input_bag_dir: str, output_dir: str):
        logger.info(f"Copying {input_bag_dir} to {output_dir} ... ")
        output_bag_dir_temp: str = osp.join(output_dir, osp.basename(input_bag_dir))
        output_bag_dir: str = osp.join(output_dir, "input_bag")
        converter = Rosbag2Converter(
            input_bag_dir,
            output_bag_dir_temp,
            self._topic_list_yaml,
            self._start_sec,
            self._end_sec,
        )
        converter.convert()
        shutil.move(output_bag_dir_temp, output_bag_dir)

    def _convert_occlusion_to_visibility(self, name: str) -> str:
        if name == "full":
            return "none"
        elif name == "partial":
            return "most"
        elif name == "most":
            return "partial"
        else:
            return "full"

    def _format_deepen_annotation(
        self, label_dicts: List[Dict[str, Any]], camera_index: Optional[Dict[str, int]] = None
    ):
        """

        e.g.:
        [
            {
                "dataset_id": "DOnC2vK05ojPr7qiqCsk2Ee7",
                "file_id": "0.pcd",
                "label_category_id": "car",
                "label_id": "car:1",
                "label_type": "3d_bbox",
                "project_id": "defaultproject",
                "stage_id": "QA",
                "attributes": {
                    "state": "moving",
                    "occlusion": "none",
                    "cycle_state": "with_rider"
                },
                "attributes_source": {
                    "state": "manual",
                    "occlusion": "manual",
                    "cycle_state": "manual"
                },
                "create_time_millis": 1634623252175,
                "label_set_id": "default",
                "labeller_email": "grp-mlops-deepen3@tier4.jp",
                "sensor_id": "lidar",
                "three_d_bbox": {
                    "cx": 81526.54828555016,
                    "cy": 50383.480369180215,
                    "cz": 34.93298238813448,
                    "h": 1.5030299457129388,
                    "l": 4.895038637695593,
                    "w": 2.107137758889027,
                    "quaternion": {
                        "x": 0,
                        "y": 0,
                        "z": 0.7522213131298905,
                        "w": 0.6589105372303157
                    }
                },
                "update_time_millis": 1634623252175,
                "user_id": "grp-mlops-deepen1@tier4.jp",
                "version": 782
            },
        ]

        Args:
            anno_path (str): path to the deepen annotation file
            camera_index (Dict[str, int]): camera index dictionary
        """
        anno_dict: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        for label_dict in label_dicts:
            if (
                self._ignore_interpolate_label
                and label_dict["labeller_email"] == "auto_interpolation"
            ):
                continue
            dataset_id = label_dict["dataset_id"]
            filename = label_dict["file_id"].split(".")[0]
            file_id = int(re.sub(r"\D", "", filename[-6:]))

            if dataset_id not in anno_dict:
                anno_dict[dataset_id] = defaultdict(list)

            anno_label_category_id: str = label_dict["label_category_id"]
            anno_label_id: str = label_dict["label_id"]
            # in case the attributes is not set
            if "attributes" not in label_dict or label_dict["attributes"] is None:
                logger.warning(f"attributes is not set in {label_dict}")
                anno_attributes = {}
            else:
                anno_attributes: Dict[str, str] = label_dict["attributes"]
            if "Occlusion_State" in anno_attributes:
                visibility: str = self._convert_occlusion_to_visibility(
                    anno_attributes["Occlusion_State"]
                )
            elif "occlusion_state" in anno_attributes:
                visibility: str = self._convert_occlusion_to_visibility(
                    anno_attributes["occlusion_state"]
                )
            else:
                visibility: str = "Not available"
            label_t4_dict: Dict[str, Any] = {
                "category_name": anno_label_category_id,
                "instance_id": anno_label_id,
                "attribute_names": [
                    f"{name.lower()}.{state}" for name, state in anno_attributes.items()
                ],
                "visibility_name": visibility,
            }
            if label_dict["sensor_id"] == "lidar" or label_dict["label_type"] == "3d_bbox":
                anno_three_d_bbox: Dict[str, str] = label_dict["three_d_bbox"]
                label_t4_dict.update(
                    {
                        "three_d_bbox": {
                            "translation": {
                                "x": anno_three_d_bbox["cx"],
                                "y": anno_three_d_bbox["cy"],
                                "z": anno_three_d_bbox["cz"],
                            },
                            "velocity": None,
                            "acceleration": None,
                            "size": {
                                "width": anno_three_d_bbox["w"],
                                "length": anno_three_d_bbox["l"],
                                "height": anno_three_d_bbox["h"],
                            },
                            "rotation": {
                                "w": anno_three_d_bbox["quaternion"]["w"],
                                "x": anno_three_d_bbox["quaternion"]["x"],
                                "y": anno_three_d_bbox["quaternion"]["y"],
                                "z": anno_three_d_bbox["quaternion"]["z"],
                            },
                        },
                        "num_lidar_pts": 0,
                        "num_radar_pts": 0,
                    }
                )
            if label_dict["sensor_id"][:6] == "camera" or label_dict["label_type"] == "box":
                sensor_id = label_dict["sensor_id"][-1]
                if camera_index is not None:
                    for k in camera_index.keys():
                        # overwrite sensor_id for multiple camera only annotation (e.g traffic light)
                        if k in filename:
                            sensor_id = camera_index[k]
                            break
                anno_two_d_bbox: List = label_dict["box"]

                if anno_two_d_bbox[2] < 0 or anno_two_d_bbox[3] < 0:
                    logger.error(f"bbox width or height:{anno_two_d_bbox} < 0")
                    continue
                if len(anno_two_d_bbox) != 4:
                    logger.error(f"bbox length {len(anno_two_d_bbox)} != 4")
                    continue

                label_t4_dict.update(
                    {
                        "two_d_box": [
                            anno_two_d_bbox[0],
                            anno_two_d_bbox[1],
                            anno_two_d_bbox[0] + anno_two_d_bbox[2],
                            anno_two_d_bbox[1] + anno_two_d_bbox[3],
                        ],
                        "sensor_id": sensor_id,
                    }
                )

            anno_dict[dataset_id][file_id].append(label_t4_dict)

        return anno_dict
