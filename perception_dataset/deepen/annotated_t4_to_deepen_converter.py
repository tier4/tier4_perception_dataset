import glob
import json
import os
import os.path as osp
import time
from typing import Any, Dict, List

import numpy as np
from pyquaternion import Quaternion
from t4_devkit import Tier4

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import EXTENSION_ENUM, LABEL_PATH_ENUM
from perception_dataset.utils.label_converter import LabelConverter
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.transform import transform_matrix
from perception_dataset.utils.misc import get_frame_index_from_filename

logger = configure_logger(modname=__name__)


class AnnotatedT4ToDeepenConverter(AbstractConverter[None]):
    def __init__(self, input_base: str, output_base: str, camera_position: Dict):
        super().__init__(input_base, output_base)
        self._camera_position = camera_position
        self._label_converter = LabelConverter(
            label_path=LABEL_PATH_ENUM.OBJECT_LABEL,
            attribute_path=LABEL_PATH_ENUM.ATTRIBUTE,
        )

    def convert(self) -> None:
        start_time = time.time()

        for scene_dir in glob.glob(osp.join(self._input_base, "*")):
            if not osp.isdir(scene_dir):
                continue
            t4_dataset_path = osp.join(scene_dir, "t4_dataset")
            if not osp.isdir(t4_dataset_path):
                t4_dataset_path = scene_dir

            scene_name = osp.basename(scene_dir)
            self._convert_one_scene(
                t4_dataset_path,
                scene_name,
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.1f} [sec]")

    def _convert_one_scene(self, input_dir: str, scene_name: str):
        output_dir = self._output_base
        os.makedirs(output_dir, exist_ok=True)
        t4_dataset = Tier4(data_root=input_dir, verbose=False)

        logger.info(f"Converting {input_dir} to {output_dir}")
        output_label: List = []

        for sample_record in t4_dataset.sample:
            sample_token = sample_record.token
            logger.info(f"sample_token: {sample_token}")
            for anno_token in sample_record.ann_3ds:
                current_label_dict: Dict = {}
                current_label_dict["attributes_source"] = {}
                current_label_dict["create_time_millis"] = "null"
                current_label_dict["update_time_millis"] = "null"
                current_label_dict["dataset_id"] = ""
                current_label_dict["labeller_email"] = "test@dummy.com"
                current_label_dict["user_id"] = "test@dummy.com"
                current_label_dict["version"] = "null"
                current_label_dict["label_set_id"] = "default"
                current_label_dict["stage_id"] = "Labelling"
                anno = t4_dataset.get("sample_annotation", anno_token)

                instance_record = t4_dataset.get("instance", anno.instance_token)
                instance_index = t4_dataset.get_idx("instance", anno.instance_token) + 1
                category_record = t4_dataset.get("category", instance_record.category_token)
                visibility_record = t4_dataset.get("visibility", anno.visibility_token)

                for sensor, token in sample_record.data.items():
                    if "LIDAR" in sensor:
                        break

                sample_data_record = t4_dataset.get("sample_data", sample_record.data[sensor])
                file_id = osp.basename(sample_data_record.filename).replace(
                    EXTENSION_ENUM.PCDBIN.value, EXTENSION_ENUM.PCD.value
                )

                # Original T4 format names the file_id as 000000.pcd.bin for example.
                # We need to convert it to 0.pcd in this case.
                file_id = str(int(file_id.split(".")[0])) + EXTENSION_ENUM.PCD.value

                label_category_id = self._label_converter.convert_label(category_record.name)

                attributes_records = [
                    t4_dataset.get("attribute", token) for token in anno.attribute_tokens
                ]
                attributes_name = [
                    self._label_converter.convert_attribute(v.name) for v in attributes_records
                ]
                attributes = {v[0 : v.find(".")]: v[v.find(".") + 1 :] for v in attributes_name}
                if "occlusion_state" not in attributes:
                    attributes["occlusion_state"] = self._convert_to_visibility_occulusion(
                        visibility_record.level
                    )

                three_d_bbox = {
                    "cx": anno.translation[0],
                    "cy": anno.translation[1],
                    "cz": anno.translation[2],
                    "h": anno.size[2],
                    "l": anno.size[1],
                    "w": anno.size[0],
                    "quaternion": {
                        "x": anno.rotation[1],
                        "y": anno.rotation[2],
                        "z": anno.rotation[3],
                        "w": anno.rotation[0],
                    },
                }
                current_label_dict["three_d_bbox"] = three_d_bbox
                sensor_id = "lidar"
                label_type = "3d_bbox"

                current_label_dict["attributes"] = attributes
                current_label_dict["file_id"] = file_id
                current_label_dict["label_category_id"] = label_category_id
                current_label_dict["label_id"] = f"{label_category_id}:{instance_index}"
                current_label_dict["sensor_id"] = sensor_id
                current_label_dict["label_type"] = label_type

                output_label.append(current_label_dict)
                print(f"{label_category_id}:{instance_index}")

        if osp.exists(osp.join(input_dir, "annotation", "object_ann.json")):
            for sample_record in t4_dataset.sample:
                for cam, sensor_id in self._camera_position.items():
                    if cam not in sample_record.data:
                        continue
                    sample_camera_token = sample_record.data[cam]
                    print(f"cam:{cam}, sample_camera_token: {sample_camera_token}")
                    object_anns = [
                        o
                        for o in t4_dataset.object_ann
                        if o.sample_data_token == sample_camera_token
                    ]

                    image_frame_index = get_frame_index_from_filename(t4_dataset.get("sample_data", sample_camera_token).filename)
                    if image_frame_index is None:
                        print(f"Failed to get frame index from filename: {t4_dataset.get('sample_data', sample_camera_token).filename}. Skipping..")
                        continue
                    
                    for ann in object_anns:
                        current_label_dict: Dict = {}
                        category_token = ann.category_token
                        category_record = t4_dataset.get("category", category_token)
                        bbox = ann.bbox
                        bbox[2] = bbox[2] - bbox[0]
                        bbox[3] = bbox[3] - bbox[1]
                        label_type = "box"
                        current_label_dict["box"] = bbox

                        label_category_id = self._label_converter.convert_label(
                            category_record.name
                        )
                        try:
                            instance_index = t4_dataset.get_idx("instance", ann.instance_token) + 1
                            attributes_records = [
                                t4_dataset.get("attribute", token)
                                for token in ann.attribute_tokens
                            ]
                            attributes_name = [
                                self._label_converter.convert_attribute(v.name)
                                for v in attributes_records
                            ]
                            attributes = {
                                v[0 : v.find(".")]: v[v.find(".") + 1 :] for v in attributes_name
                            }

                            current_label_dict["attributes"] = attributes
                            current_label_dict["create_time_millis"] = "null"
                            current_label_dict["update_time_millis"] = "null"
                            current_label_dict["dataset_id"] = ""
                            current_label_dict["labeller_email"] = "test@dummy.com"
                            current_label_dict["user_id"] = "test@dummy.com"
                            current_label_dict["version"] = "null"
                            current_label_dict["label_set_id"] = "default"
                            current_label_dict["stage_id"] = "Labelling"
                            current_label_dict["file_id"] = f"{image_frame_index}.pcd"
                            current_label_dict["label_category_id"] = label_category_id
                            current_label_dict["label_id"] = (
                                f"{label_category_id}:{instance_index}"
                            )
                            current_label_dict["sensor_id"] = sensor_id
                            current_label_dict["label_type"] = label_type

                            output_label.append(current_label_dict)
                            print(f"{label_category_id}:{instance_index}")
                        except KeyError:
                            instance_id = ann.instance_token
                            print(f"There is no instance_id:{instance_id}")

        output_json = {"labels": output_label}
        with open(osp.join(output_dir, f"{scene_name}.json"), "w") as f:
            json.dump(output_json, f, indent=4)

        logger.info(f"Done Conversion: {input_dir} to {output_dir}")

    def _get_data(self, t4_dataset: Tier4, sensor_channel_token: str) -> Dict[str, Any]:
        sd_record = t4_dataset.get("sample_data", sensor_channel_token)
        cs_record = t4_dataset.get("calibrated_sensor", sd_record.calibrated_sensor_token)
        ep_record = t4_dataset.get("ego_pose", sd_record.ego_pose_token)

        sensor2ego_transform = transform_matrix(
            translation=cs_record.translation,
            rotation=Quaternion(cs_record.rotation),
        )
        ego2global_transform = transform_matrix(
            translation=ep_record.translation,
            rotation=Quaternion(ep_record.rotation),
        )

        sensor2global_transform = ego2global_transform @ sensor2ego_transform
        sensor2global_translation = sensor2global_transform[:3, 3]
        sensor2global_rotation = np.array(list(Quaternion(matrix=sensor2global_transform[:3, :3])))

        ret_dict = {
            "fileformat": sd_record.fileformat,
            "unix_timestamp": self._timestamp_to_sec(sd_record.timestamp),
            "sensor2global_transform": sensor2global_transform,
            "sensor2global_translation": sensor2global_translation,
            "sensor2global_rotation": sensor2global_rotation,
        }

        return ret_dict

    def _timestamp_to_sec(self, timestamp: int) -> float:
        return float(timestamp) * 1e-6

    def _convert_to_visibility_occulusion(self, name: str) -> str:
        if name == "none":
            return "full"
        elif name == "most":
            return "partial"
        elif name == "partial":
            return "most"
        else:
            return "none"
