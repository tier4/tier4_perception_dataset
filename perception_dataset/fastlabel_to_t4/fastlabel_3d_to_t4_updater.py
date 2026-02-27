from __future__ import annotations

from collections import defaultdict
import os.path as osp
from pathlib import Path
import shutil
from typing import Any, Dict, List

from perception_dataset.fastlabel_to_t4.fastlabel_to_t4_converter import (
    FastLabelToT4Converter,
)
from perception_dataset.t4_dataset.annotation_files_updater import AnnotationFilesUpdater
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.transform import rotation_to_quaternion

logger = configure_logger(modname=__name__)


class FastLabel3dToT4Updater(FastLabelToT4Converter):
    """Updates existing T4 datasets with FastLabel 3D annotations (in-place or to another directory).
    Reads existing annotation files, merges new 3D labels, and optionally runs keyframe consistency resolution.
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        input_anno_base: str,
        overwrite_mode: bool,
        description: Dict[str, Dict[str, str]],
        make_t4_dataset_dir: bool = True,
        only_annotation_frame: bool = False,
    ):
        super().__init__(
            input_base,
            output_base,
            input_anno_base,
            overwrite_mode,
            description,
            make_t4_dataset_dir=make_t4_dataset_dir,
            only_annotation_frame=only_annotation_frame,
            input_bag_base=None,
            topic_list=None,
        )
        self._make_t4_dataset_dir = make_t4_dataset_dir
        self._only_annotation_frame = only_annotation_frame

    def convert(self) -> None:
        in_place_update = self._input_base.resolve() == self._output_base.resolve()
        if in_place_update:
            logger.warning(
                "input_base and output_base are the same (in-place update). "
                "Annotation files will be updated without deleting/copying scene data."
            )
        t4_datasets = sorted([d.name for d in self._input_base.iterdir() if d.is_dir()])
        anno_jsons_dict = self._load_annotation_jsons(t4_datasets, ".pcd")
        fl_annotations = self._format_fastlabel_3d_annotation(anno_jsons_dict)

        for t4dataset_name in t4_datasets:
            # Check if annotation exists
            if t4dataset_name not in fl_annotations.keys():
                continue

            # Check if input directory exists
            input_dir = self._input_base / t4dataset_name
            # Some datasets are stored as scene_dir/t4_dataset/{data,annotation}
            # (depending on make_t4_dataset_dir setting used at generation time).
            input_annotation_dir = input_dir / "annotation"
            if not osp.exists(input_annotation_dir):
                alt_input_annotation_dir = input_dir / "t4_dataset" / "annotation"
                if osp.exists(alt_input_annotation_dir):
                    input_annotation_dir = alt_input_annotation_dir
                else:
                    logger.warning(f"input_annotation_dir not exists under {input_dir}.")
                    continue

            # Check if output directory already exists
            output_dir = self._output_base / t4dataset_name
            if self._make_t4_dataset_dir:
                output_dir = output_dir / "t4_dataset"
            if self._input_bag_base is not None:
                input_bag_dir = Path(self._input_bag_base) / t4dataset_name

            in_place = in_place_update and (input_dir == output_dir)
            if osp.exists(output_dir):
                logger.warning(f"{output_dir} already exists.")
                if self._overwrite_mode:
                    if in_place:
                        pass
                    else:
                        shutil.rmtree(output_dir, ignore_errors=True)
                else:
                    continue

            # Copy input data to output directory (no-op when in-place)
            if not in_place:
                self._copy_data(input_dir, output_dir)
            # Make rosbag
            if self._input_bag_base is not None and not osp.exists(
                osp.join(output_dir, "input_bag")
            ):
                self._find_start_end_time(input_dir)
                self._make_rosbag(str(input_bag_dir), str(output_dir))

            # Start updating annotations
            annotation_files_updater = AnnotationFilesUpdater(
                description=self._description,
                surface_categories=self._surface_categories,
                label_coordinates="lidar",
            )
            annotation_files_updater.convert_one_scene(
                input_dir=input_dir,
                output_dir=output_dir,
                scene_anno_dict=fl_annotations[t4dataset_name],
                dataset_name=t4dataset_name,
                only_annotation_frame=self._only_annotation_frame,
            )
            logger.info(f"Finished updating annotations for {t4dataset_name}")

    def _format_fastlabel_3d_annotation(self, annotations: Dict[str, List[Dict[str, Any]]]):
        """
        Update用(3D)は、FastLabel側の欠損/不正データで止まらないようにガードする。
        points想定: [x, y, z, roll, pitch, yaw, length, width, height]
        """
        fl_annotations: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

        for filename, ann_list in sorted(annotations.items()):
            for ann in ann_list:
                # Extract dataset_name from ann["name"] (e.g., "dataset_name/00001.pcd" -> "dataset_name")
                ann_name_parts = ann["name"].split("/")
                if len(ann_name_parts) > 1:
                    dataset_name: str = "/".join(ann_name_parts[:-1])
                else:
                    # Fallback to filename-based extraction if ann["name"] doesn't have path separator
                    dataset_name: str = Path(filename).stem.rsplit("_", 1)[0] if "_" in Path(filename).stem else Path(filename).stem
                
                filename: str = ann_name_parts[-1]
                file_id: int = int(filename.split(".")[0])

                if dataset_name not in fl_annotations:
                    fl_annotations[dataset_name] = defaultdict(list)

                for a in ann["annotations"]:
                    # FastLabelの同一jsonに2D系(Polygon等)が混在するケースがあるため、
                    # 3D updateでは cuboid のみ対象にする（混在していたらデータ不整合としてエラーにする）。
                    anno_type = a.get("type")
                    if anno_type is not None and anno_type != "cuboid":
                        raise ValueError(
                            "Invalid annotation type for 3D update. "
                            f"Expected type='cuboid' but got type={anno_type}. "
                            f"dataset={dataset_name} file_id={file_id} ann_id={a.get('id')}"
                        )

                    visibility: str = "none"
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

                    points = a.get("points")
                    if not isinstance(points, list):
                        raise ValueError(
                            "Invalid points for 3D cuboid. "
                            f"Expected list but got {type(points)}. "
                            f"dataset={dataset_name} file_id={file_id} instance_id={instance_id} type={anno_type} points={points}"
                        )
                    if len(points) < 9:
                        # avoid logging huge polygons/segments
                        points_preview = points
                        if len(points) == 1 and isinstance(points[0], list):
                            points_preview = ["<nested points omitted>"]
                        raise ValueError(
                            "Invalid points length for 3D cuboid. "
                            "Expected at least 9 values: [x,y,z,roll,pitch,yaw,length,width,height]. "
                            f"dataset={dataset_name} file_id={file_id} instance_id={instance_id} type={anno_type} points_len={len(points)} points={points_preview}"
                        )

                    rotation = points[3:6]
                    if len(rotation) != 3 or any(r is None for r in rotation):
                        raise ValueError(
                            "Invalid rotation for 3D cuboid. "
                            f"Expected 3 floats [roll,pitch,yaw] but got rotation={rotation}. "
                            f"dataset={dataset_name} file_id={file_id} instance_id={instance_id} type={anno_type} points={points}"
                        )
                    q = rotation_to_quaternion(rotation)

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

