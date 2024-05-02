import json
from pathlib import Path
from typing import Dict

from nuimages import NuImages
from nuscenes import NuScenes

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class T4dataset2D3DMerger(AbstractConverter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        dataset_corresponding: Dict[str, int],
    ):
        self._input_base = Path(input_base)
        self._output_base = Path(output_base)
        self._t4dataset_name_to_merge: Dict[str, str] = dataset_corresponding

    def convert(self):
        for output_3d_t4dataset_name in self._t4dataset_name_to_merge.keys():
            input_t4dataset_name = self._t4dataset_name_to_merge[output_3d_t4dataset_name]
            input_2d_annotation_dir = self._input_base / input_t4dataset_name / "annotation"
            if not input_2d_annotation_dir.exists():
                input_2d_annotation_dir = (
                    self._input_base / input_t4dataset_name / "t4_dataset/annotation"
                )
            if not input_2d_annotation_dir.exists():
                logger.warning(f"input_dir {input_2d_annotation_dir} not exists.")
                continue

            output_3d_annotation_dir = self._output_base / output_3d_t4dataset_name / "annotation"
            if not output_3d_annotation_dir.exists():
                logger.warning(f"output_dir {output_3d_annotation_dir} not exists.")
                continue

            out_attribute, attribute_in_out_token_map = self._merge_json_files(
                input_2d_annotation_dir, output_3d_annotation_dir, "attribute.json"
            )
            out_category, category_in_out_token_map = self._merge_json_files(
                input_2d_annotation_dir, output_3d_annotation_dir, "category.json"
            )
            out_instance, instance_in_out_token_map = self._merge_json_files(
                input_2d_annotation_dir, output_3d_annotation_dir, "instance.json"
            )
            out_visibility, visibility_in_out_token_map = self._merge_json_files(
                input_2d_annotation_dir, output_3d_annotation_dir, "visibility.json"
            )

            out_object_ann = self._update_object_ann(
                input_2d_annotation_dir,
                attribute_in_out_token_map,
                category_in_out_token_map,
                instance_in_out_token_map,
            )
            out_surface_ann = self._update_surface_ann(
                input_2d_annotation_dir, category_in_out_token_map
            )
            with open(output_3d_annotation_dir / "attribute.json", "w") as f:
                json.dump(out_attribute, f, indent=4)
            with open(output_3d_annotation_dir / "category.json", "w") as f:
                json.dump(out_category, f, indent=4)
            with open(output_3d_annotation_dir / "instance.json", "w") as f:
                json.dump(out_instance, f, indent=4)
            with open(output_3d_annotation_dir / "visibility.json", "w") as f:
                json.dump(out_visibility, f, indent=4)
            with open(output_3d_annotation_dir / "object_ann.json", "w") as f:
                json.dump(out_object_ann, f, indent=4)
            with open(output_3d_annotation_dir / "surface_ann.json", "w") as f:
                json.dump(out_surface_ann, f, indent=4)

    def _merge_json_files(self, input_dir, output_dir, filename):
        """
        Merge the input json file to the output json file
        Args:
            input_dir: input directory
            output_dir: output directory
            filename: json file name
        return:
            out_json_data: list of output json data
            in_out_token_map: mapping of input token to output token for the same name data
        """
        with open(input_dir / filename) as f:
            in_data: list[dict[str, str]] = json.load(f)
        with open(output_dir / filename) as f:
            out_data: list[dict[str, str]] = json.load(f)

        in_out_token_map = {}
        for in_d in in_data:
            for out_d in out_data:
                if "name" in in_d.keys():
                    if in_d["name"] == out_d["name"]:
                        in_out_token_map[in_d["token"]] = out_d["token"]
                        break
                elif "token" in in_d.keys():
                    if in_d["token"] == out_d["token"]:
                        in_out_token_map[in_d["token"]] = out_d["token"]
                        break

        out_data += [d for d in in_data if d["token"] not in in_out_token_map.keys()]

        return out_data, in_out_token_map

    def _update_object_ann(
        self,
        input_2d_annotation_dir,
        attribute_in_out_token_map,
        category_in_out_token_map,
        instance_in_out_token_map,
    ):
        """
        Update the attribute token, category token, and instance token in object annotation json file
        Args:
            input_2d_annotation_dir: input directory
            attribute_in_out_token_map: mapping of input attribute token to output attribute token
            category_in_out_token_map: mapping of input category token to output category token
            instance_in_out_token_map: mapping of input instance token to output instance token
        Return:
            object_ann: list of updated object annotation data
        """
        with open(input_2d_annotation_dir / "object_ann.json") as f:
            object_ann: list[dict[str, str]] = json.load(f)

        for obj in object_ann:
            for attribute_token in obj["attribute_tokens"]:
                if attribute_token in attribute_in_out_token_map.keys():
                    obj["attribute_tokens"].remove(attribute_token)
                    obj["attribute_tokens"].append(attribute_in_out_token_map[attribute_token])
            if obj["category_token"] in category_in_out_token_map.keys():
                obj["category_token"] = category_in_out_token_map[obj["category_token"]]
            if obj["instance_token"] in instance_in_out_token_map.keys():
                obj["instance_token"] = instance_in_out_token_map[obj["instance_token"]]

        return object_ann

    def _update_surface_ann(self, input_2d_annotation_dir, category_in_out_token_map):
        """
        Update the category token in surface annotation json file
        Args:
            input_2d_annotation_dir: input directory
            category_in_out_token_map: mapping of input category token to output category token
        Return:
            surface_ann: list of updated surface annotation data
        """
        with open(input_2d_annotation_dir / "surface_ann.json") as f:
            surface_ann: list[dict[str, str]] = json.load(f)

        for surface in surface_ann:
            if surface["category_token"] in category_in_out_token_map.keys():
                surface["category_token"] = category_in_out_token_map[surface["category_token"]]
        return surface_ann
