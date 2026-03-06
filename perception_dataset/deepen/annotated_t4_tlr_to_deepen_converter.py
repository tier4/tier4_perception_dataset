import json
import os
import os.path as osp
from typing import Dict, List

from t4_devkit import Tier4

from perception_dataset.constants import LABEL_PATH_ENUM
from perception_dataset.deepen.annotated_t4_to_deepen_converter import AnnotatedT4ToDeepenConverter
from perception_dataset.utils.label_converter import TrafficLightLabelConverter
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.misc import get_frame_index_from_filename

logger = configure_logger(modname=__name__)


class AnnotatedT4TlrToDeepenConverter(AnnotatedT4ToDeepenConverter):
    def __init__(self, input_base: str, output_base: str, camera_position: Dict):
        super().__init__(input_base, output_base, camera_position)
        self._label_converter = TrafficLightLabelConverter(
            label_path=LABEL_PATH_ENUM.TRAFFIC_LIGHT_LABEL
        )

    def _convert_one_scene(self, input_dir: str, scene_name: str):
        output_dir = self._output_base
        os.makedirs(output_dir, exist_ok=True)
        t4_dataset = Tier4(data_root=input_dir, verbose=False)

        logger.info(f"Converting {input_dir} to {output_dir}")
        output_label: List = []

        if osp.exists(osp.join(input_dir, "annotation", "object_ann.json")):
            for sample_record in t4_dataset.sample:
                for cam, sensor_id in self._camera_position.items():
                    if cam not in sample_record.data:
                        continue
                    sample_camera_token = sample_record.data[cam]
                    image_frame_index = get_frame_index_from_filename(
                        t4_dataset.get("sample_data", sample_camera_token).filename
                    )

                    object_anns = [
                        object_ann
                        for object_ann in t4_dataset.object_ann
                        if object_ann.sample_data_token == sample_camera_token
                    ]

                    for ann in object_anns:
                        current_label_dict: Dict = {}
                        category_token = ann.category_token
                        category_record = t4_dataset.get("category", category_token)
                        roi = ann.bbox
                        # Convert Roi tuple to bbox list and transform to width/height format
                        bbox = [roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]]
                        label_type = "box"
                        current_label_dict["box"] = bbox

                        label_category_id = self._label_converter.convert_label(
                            category_record.name
                        )
                        try:
                            instance = t4_dataset.get("instance", ann.instance_token)
                            instance_name = instance.instance_name
                            traffic_light_id = instance_name.split("::")[1]
                            attributes: Dict = {
                                "Occlusion_State": "none",
                                "Truncation_State": "non-truncated",
                                "light_status": "on",
                            }  # TODO: Need to implement attributes parser

                            current_label_dict["attributes"] = attributes
                            current_label_dict["create_time_millis"] = "null"
                            current_label_dict["update_time_millis"] = "null"
                            current_label_dict["dataset_id"] = ""
                            current_label_dict["labeller_email"] = "scale"
                            current_label_dict["user_id"] = "scale"
                            current_label_dict["version"] = "null"
                            current_label_dict["label_set_id"] = "default"
                            current_label_dict["stage_id"] = "Labelling"
                            current_label_dict["file_id"] = f"{image_frame_index:05}.jpg"
                            current_label_dict["label_category_id"] = label_category_id
                            current_label_dict["label_id"] = (
                                f"{label_category_id}:{traffic_light_id}"
                            )
                            current_label_dict["sensor_id"] = sensor_id
                            current_label_dict["label_type"] = label_type

                            output_label.append(current_label_dict)
                        except KeyError:
                            instance_id = ann.instance_token
                            print(f"There is no instance_id:{instance_id}")

        output_json = {"labels": output_label}
        with open(osp.join(output_dir, f"{scene_name}.json"), "w") as f:
            json.dump(output_json, f, indent=4)

        logger.info(f"Done Conversion: {input_dir} to {output_dir}")
