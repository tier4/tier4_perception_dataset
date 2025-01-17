import base64
from collections import defaultdict
import os.path as osp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from nptyping import NDArray
from nuimages import NuImages
import numpy as np
from nuscenes.nuscenes import NuScenes
from pycocotools import mask as cocomask

from perception_dataset.constants import EXTENSION_ENUM, SENSOR_ENUM, T4_FORMAT_DIRECTORY_NAME
from perception_dataset.t4_dataset.classes.abstract_class import AbstractTable
from perception_dataset.t4_dataset.classes.attribute import AttributeTable
from perception_dataset.t4_dataset.classes.category import CategoryTable
from perception_dataset.t4_dataset.classes.instance import InstanceRecord, InstanceTable
from perception_dataset.t4_dataset.classes.lidarseg import LidarSegTable
from perception_dataset.t4_dataset.classes.object_ann import ObjectAnnTable
from perception_dataset.t4_dataset.classes.sample_annotation import (
    SampleAnnotationRecord,
    SampleAnnotationTable,
)
from perception_dataset.t4_dataset.classes.surface_ann import SurfaceAnnTable
from perception_dataset.t4_dataset.classes.visibility import VisibilityTable
from perception_dataset.utils.calculate_num_points import calculate_num_points


class AnnotationFilesGenerator:
    def __init__(
        self,
        with_camera: bool = True,
        description: Dict[str, Dict[str, str]] = {},
        surface_categories: List[str] = [],
    ):
        # TODO(yukke42): remove the hard coded attribute description
        self._attribute_table = AttributeTable(
            name_to_description={},
            default_value="",
        )
        # TODO(yukke42): remove the hard coded category description
        self._category_table = CategoryTable(
            name_to_description={},
            default_value="",
        )
        self._instance_table = InstanceTable()
        self._visibility_table = VisibilityTable(
            level_to_description=description.get(
                "visibility",
                {
                    "v0-40": "visibility of whole object is between 0 and 40%",
                    "v40-60": "visibility of whole object is between 40 and 60%",
                    "v60-80": "visibility of whole object is between 60 and 80%",
                    "v80-100": "visibility of whole object is between 80 and 100%",
                    "none": "visibility isn't available",
                },
            ),
            default_value="",
        )
        self._sample_annotation_table = SampleAnnotationTable()
        self._object_ann_table = ObjectAnnTable()
        self._surface_ann_table = SurfaceAnnTable()

        self._instance_token_to_annotation_token_list: Dict[str, List[str]] = defaultdict(list)

        if with_camera:
            self._camera2idx = description.get("camera_index")
        else:
            self._camera2idx = None
        self._with_lidar = description.get("with_lidar", True)
        self._surface_categories: List[str] = surface_categories
        self._with_lidarseg = description.get("with_lidarseg", False)

        if self._with_lidarseg:
            assert self._with_lidar, "with_lidar must be set if with_lidarseg is set!"

    def save_tables(self, anno_dir: str):
        for cls_attr in self.__dict__.values():
            if isinstance(cls_attr, AbstractTable):
                print(f"{cls_attr.FILENAME}: #rows {len(cls_attr)}")
                cls_attr.save_json(anno_dir)

    def convert_one_scene(
        self,
        input_dir: str,
        output_dir: str,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        dataset_name: str,
    ):
        anno_dir = osp.join(output_dir, "annotation")
        if not osp.exists(anno_dir):
            raise ValueError(f"Annotations files doesn't exist in {anno_dir}")

        nusc = NuScenes(version="annotation", dataroot=input_dir, verbose=False)
        try:
            if "LIDAR_TOP" in nusc.sample[0]["data"]:
                lidar_sensor_channel = SENSOR_ENUM.LIDAR_TOP.value["channel"]
            else:
                lidar_sensor_channel = SENSOR_ENUM.LIDAR_CONCAT.value["channel"]
        except KeyError as e:
            print(e)

        nuim = NuImages(version="annotation", dataroot=input_dir, verbose=False)
        # TODO (KokSeang): Support lidarseg and other annotations at the same time
        if self._with_lidarseg:
            self._convert_lidarseg_scene_annotations(
                scene_anno_dict=scene_anno_dict,
                dataset_name=dataset_name,
                nusc=nusc,
                anno_dir=anno_dir,
                lidar_sensor_channel=lidar_sensor_channel,
            )
        else:
            self._convert_scene_annotations(
                scene_anno_dict=scene_anno_dict, dataset_name=dataset_name, nuim=nuim, nusc=nusc
            )

        self._attribute_table.save_json(anno_dir)
        self._category_table.save_json(anno_dir)
        self._instance_table.save_json(anno_dir)
        self._sample_annotation_table.save_json(anno_dir)
        self._visibility_table.save_json(anno_dir)
        self._object_ann_table.save_json(anno_dir)
        self._surface_ann_table.save_json(anno_dir)
        if self._with_lidar:
            # Calculate and overwrite number of points in lidar cuboid bounding box in annotations
            calculate_num_points(output_dir, lidar_sensor_channel, self._sample_annotation_table)
            self._sample_annotation_table.save_json(anno_dir)

    def _convert_scene_annotations(
        self,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        dataset_name: str,
        nusc: NuScenes,
        nuim: NuImages,
    ) -> None:
        """
        Convert scene annotations to T4 dataset format.
        :param scene_anno_dict: Scene annotations.
        :param dataset_name: Dataset name.
        :param nusc: NuScenes object.
        :param nuim: Nuimages object.
        """
        frame_index_to_sample_token: Dict[int, str] = {}
        for sample_data in nusc.sample_data:
            frame_index = int((sample_data["filename"].split("/")[2]).split(".")[0])
            frame_index_to_sample_token[frame_index] = sample_data["sample_token"]

        # FIXME: Avoid hard coding the number of cameras
        num_cameras = 6 if self._camera2idx is None else len(self._camera2idx)
        frame_index_to_sample_data_token: List[Dict[int, str]] = [{} for _ in range(num_cameras)]
        mask: List[Dict[int, str]] = [{} for x in range(num_cameras)]

        has_2d_annotation: bool = False
        for frame_index in sorted(scene_anno_dict.keys()):
            anno_list: List[Dict[str, Any]] = scene_anno_dict[frame_index]
            for anno in anno_list:
                if "two_d_box" in anno.keys() or "two_d_segmentation" in anno.keys():
                    has_2d_annotation = True
                    break

        if has_2d_annotation:
            object_mask: NDArray = np.zeros((0, 0), dtype=np.uint8)
            prev_wid_hgt: Tuple = (0, 0)
            # NOTE: num_cameras is always 6, because it is hard coded above.
            for _, sample_nuim in enumerate(nuim.sample_data):
                if sample_nuim["fileformat"] == "png" or sample_nuim["fileformat"] == "jpg":
                    cam = sample_nuim["filename"].split("/")[1]
                    cam_idx = self._camera2idx[cam]

                    frame_index = int((sample_nuim["filename"].split("/")[2]).split(".")[0])
                    frame_index_to_sample_data_token[cam_idx].update(
                        {frame_index: sample_nuim["token"]}
                    )

                    wid_hgt = (sample_nuim["width"], sample_nuim["height"])
                    if wid_hgt != prev_wid_hgt:
                        prev_wid_hgt = wid_hgt
                        object_mask = np.zeros(wid_hgt, dtype=np.uint8)
                        object_mask = cocomask.encode(np.asfortranarray(object_mask))
                        object_mask["counts"] = base64.b64encode(object_mask["counts"]).decode(
                            "ascii"
                        )
                    mask[cam_idx].update({frame_index: object_mask})

        self.convert_annotations(
            scene_anno_dict=scene_anno_dict,
            frame_index_to_sample_token=frame_index_to_sample_token,
            dataset_name=dataset_name,
            frame_index_to_sample_data_token=frame_index_to_sample_data_token,
            mask=mask,
        )

    def convert_annotations(
        self,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        frame_index_to_sample_token: Dict[int, str],
        dataset_name: str,
        frame_index_to_sample_data_token: Optional[List[Dict[int, str]]] = None,
        mask: Optional[List[Dict[int, str]]] = None,
    ):
        self._convert_to_t4_format(
            scene_anno_dict=scene_anno_dict,
            frame_index_to_sample_token=frame_index_to_sample_token,
            dataset_name=dataset_name,
            frame_index_to_sample_data_token=frame_index_to_sample_data_token,
            mask=mask,
        )
        self._connect_annotations_in_scene()

    def _convert_to_t4_format(
        self,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        frame_index_to_sample_token: Dict[int, str],
        dataset_name: str,
        frame_index_to_sample_data_token: List[Dict[int, str]],
        mask: List[Dict[int, str]],
    ):
        """Convert the annotations to the NuScenes format.

        Args:
                scene_anno_dict (Dict[int, List[Dict[str, Any]]]): [description]
            frame_index_to_sample_token (Dict[int, str]): [description]
            frame_index_to_sample_data_token (Dict[int, str]):

        scene_anno_dict:
        {
                0: [
                        {
                            "category_name" (str): category name of object,
                            "instance_id" (str): instance id of object,
                            "attribute_names" (List[str]): list of object attributes,
                            "three_d_bbox": {
                                "translation": {
                                                        "x" (float): x of object location,
                                                        "y" (float): y of object location,
                                                        "z" (float): z of object location,
                                                },
                                                "velocity" (Optional[Dict[str, float]]): {
                                                        "x" (float): x of object velocity,
                                                        "y" (float): y of object velocity,
                                                        "z" (float): z of object velocity,
                                                },
                                                "acceleration" (Optional[Dict[str, float]]): {
                                                        "x" (float): x of object acceleration,
                                                        "y" (float): y of object acceleration,
                                                        "z" (float): z of object acceleration,
                                                },
                                                        "size": {
                                                        "width" (float): width of object size,
                                                        "length" (float): length of object size,
                                                        "height" (float): height of object size,
                                                },
                                                "rotation": {
                                                        "w" (float): w of object quaternion,
                                                        "x" (float): x of object quaternion,
                                                        "y" (float): y of object quaternion.
                                                        "z" (float): z of object quaternion,
                                                },
                                        },
                                        "two_d_box": [
                                                "x" (float): x of left top corner,
                                                "y" (float): y of left top corner,
                                                "w" (float): width of bbox,
                                                "h" (float): height of bbox,
                                        ]
                                        "sensor_id": id of the camera
                                        "num_lidar_pts" (int): the number of lidar points in object,
                                        "num_radar_pts" (int): the number of radar points in object,
                                },
                                        ...
                        ],
                1: []. ...
        }

        """
        for frame_index in sorted(scene_anno_dict.keys()):
            anno_list: List[Dict[str, Any]] = scene_anno_dict[frame_index]
            # in case of the first frame in annotation is not 0
            min_frame_index: int = min(scene_anno_dict.keys())

            # for the case that the frame_index is not in the sample_token
            if frame_index - min_frame_index not in frame_index_to_sample_token:
                print(f"frame_index {frame_index} in annotation.json is not in sample_token")
                continue

            for anno in anno_list:
                # Category
                category_token: str = self._category_table.get_token_from_name(
                    name=anno["category_name"]
                )

                # Instance
                instance_token: str = self._instance_table.get_token_from_id(
                    instance_id=anno["instance_id"],
                    category_token=category_token,
                    dataset_name=dataset_name,
                )

                # Attribute
                attribute_tokens: List[str] = [
                    self._attribute_table.get_token_from_name(name=attr_name)
                    for attr_name in anno["attribute_names"]
                ]

                # Visibility
                visibility_token: str = self._visibility_table.get_token_from_level(
                    level=anno.get("visibility_name", "none")
                )

                # Sample Annotation
                if "three_d_bbox" in anno.keys():
                    anno_three_d_bbox: Dict[str, float] = anno["three_d_bbox"]
                    sample_annotation_token: str = self._sample_annotation_table.insert_into_table(
                        sample_token=frame_index_to_sample_token[frame_index],
                        instance_token=instance_token,
                        attribute_tokens=attribute_tokens,
                        visibility_token=visibility_token,
                        translation=anno_three_d_bbox["translation"],
                        velocity=anno_three_d_bbox["velocity"],
                        acceleration=anno_three_d_bbox["acceleration"],
                        size=anno_three_d_bbox["size"],
                        rotation=anno_three_d_bbox["rotation"],
                        num_lidar_pts=anno["num_lidar_pts"],
                        num_radar_pts=anno["num_radar_pts"],
                        automatic_annotation=False,
                    )
                    self._instance_token_to_annotation_token_list[instance_token].append(
                        sample_annotation_token
                    )

                # Object Annotation
                if ("two_d_box" in anno.keys() or "two_d_segmentation" in anno.keys()) and anno[
                    "category_name"
                ] not in self._surface_categories:
                    sensor_id: int = int(anno["sensor_id"])
                    if frame_index not in frame_index_to_sample_data_token[sensor_id]:
                        continue
                    anno_two_d_box: List[float] = (
                        self._clip_bbox(anno["two_d_box"], mask[sensor_id][frame_index])
                        if "two_d_box" in anno.keys()
                        else None
                    )
                    self._object_ann_table.insert_into_table(
                        sample_data_token=frame_index_to_sample_data_token[sensor_id][frame_index],
                        instance_token=instance_token,
                        category_token=category_token,
                        attribute_tokens=attribute_tokens,
                        bbox=anno_two_d_box,
                        mask=(
                            anno["two_d_segmentation"]
                            if "two_d_segmentation" in anno.keys()
                            else mask[sensor_id][frame_index]
                        ),
                        automatic_annotation=False,
                    )

                # Surface Annotation
                if (
                    "two_d_segmentation" in anno.keys()
                    and anno["category_name"] in self._surface_categories
                ):
                    sensor_id: int = int(anno["sensor_id"])
                    if frame_index not in frame_index_to_sample_data_token[sensor_id]:
                        continue
                    self._surface_ann_table.insert_into_table(
                        category_token=category_token,
                        mask=anno["two_d_segmentation"],
                        sample_data_token=frame_index_to_sample_data_token[sensor_id][frame_index],
                        automatic_annotation=False,
                    )

    def _clip_bbox(self, bbox: List[float], mask: Dict[str, Any]) -> List[float]:
        """Clip the bbox to the image size."""
        try:
            width, height = mask["size"]
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(width, bbox[2])
            bbox[3] = min(height, bbox[3])
        except Exception as e:
            print(e)

        return bbox

    def _connect_annotations_in_scene(self):
        """Annotation for Instance and SampleAnnotation. This function adds the relationship between annotations."""
        for (
            instance_token,
            annotation_token_list,
        ) in self._instance_token_to_annotation_token_list.items():
            # set info in instance
            inst_rec: InstanceRecord = self._instance_table.select_record_from_token(
                instance_token
            )
            inst_rec.set_annotation_info(
                nbr_annotations=len(annotation_token_list),
                first_annotation_token=annotation_token_list[0],
                last_annotation_token=annotation_token_list[-1],
            )
            self._instance_table.set_record_to_table(inst_rec)

            # set next/prev of sample_annotation
            for token_i in range(1, len(annotation_token_list)):
                prev_token: str = annotation_token_list[token_i - 1]
                cur_token: str = annotation_token_list[token_i]

                prev_rec: SampleAnnotationRecord = (
                    self._sample_annotation_table.select_record_from_token(prev_token)
                )
                prev_rec.next_token = cur_token
                self._sample_annotation_table.set_record_to_table(prev_rec)

                cur_rec: SampleAnnotationRecord = (
                    self._sample_annotation_table.select_record_from_token(cur_token)
                )
                cur_rec.prev_token = prev_token
                self._sample_annotation_table.set_record_to_table(cur_rec)

    def _convert_lidarseg_scene_annotations(
        self,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        lidar_sensor_channel: str,
        nusc: NuScenes,
        anno_dir: str,
    ) -> None:
        """
        Convert annotations for scenes in lidarseg to t4 dataset format.
        Specifically, it creates lidarseg.json and update category.json.
        :param scene_anno_dict: Dict of annotations for every scenes.
        :param lidar_sensor_channel: Lidar sensor channel name.
        :param nusc: Nuscene object for this dataset.
        :param anno_dir: Annotation directory.
        """
        lidarseg_table = LidarSegTable()

        frame_index_to_sample_data_token: Dict[int, str] = {}
        frame_index_to_sample_token: Dict[int, str] = {}
        for sample_data in nusc.sample_data:
            frame_index = int((sample_data["filename"].split("/")[2]).split(".")[0])
            frame_index_to_sample_token[frame_index] = sample_data["sample_token"]
            if lidar_sensor_channel in sample_data["filename"]:
                frame_index_to_sample_data_token[frame_index] = sample_data["token"]

        # Create lidarseg folder in
        lidarseg_anno_path = Path(anno_dir) / T4_FORMAT_DIRECTORY_NAME.LIDARSEG_ANNO_FOLDER.value
        lidarseg_anno_path.mkdir(parents=True, exist_ok=True)

        for frame_index in sorted(scene_anno_dict.keys()):
            anno_list: List[Dict[str, Any]] = scene_anno_dict[frame_index]
            # in case of the first frame in annotation is not 0
            min_frame_index: int = min(scene_anno_dict.keys())

            # for the case that the frame_index is not in the sample_token
            if frame_index - min_frame_index not in frame_index_to_sample_token:
                print(f"frame_index {frame_index} in annotation.json is not in sample_token")
                continue

            sample_data_token = frame_index_to_sample_data_token.get(frame_index, None)
            if sample_data_token is None:
                raise ValueError(f"sample_data doesn't have {lidar_sensor_channel}!")

            # All tmp lidarseg folders before moving
            for anno in anno_list:
                # Get a LidarSeg token
                lidarseg_token = lidarseg_table.insert_into_table(
                    filename=anno["lidarseg_anno_file"], sample_data_token=sample_data_token
                )

                # Move lidarseg_anno_file to lidarseg_anno_path with token name
                new_lidarseg_anno_filename = str(
                    lidarseg_anno_path / (lidarseg_token + EXTENSION_ENUM.BIN.value)
                )
                shutil.move(anno["lidarseg_anno_file"], new_lidarseg_anno_filename)

                # Update the lidarseg record with the new filename
                lidarseg_record = lidarseg_table.select_record_from_token(token=lidarseg_token)
                lidarseg_record.filename = new_lidarseg_anno_filename
                lidarseg_table.set_record_to_table(
                    record=lidarseg_record,
                )

                # Category
                for category in anno["paint_categories"]:
                    self._category_table.get_token_from_name(name=category)

        lidarseg_table.save_json(anno_dir)

        # Remove the older lidarseg
        shutil.rmtree(T4_FORMAT_DIRECTORY_NAME.LIDARSEG_ANNO_FOLDER.value, ignore_errors=True)
