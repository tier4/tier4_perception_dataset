import base64
from collections import defaultdict
import os.path as osp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from nptyping import NDArray
import numpy as np
from pycocotools import mask as cocomask
from scipy.spatial.transform import Rotation
from t4_devkit import Tier4
from t4_devkit.schema.tables import (
    Attribute,
    Category,
    Instance,
    LidarSeg,
    ObjectAnn,
    SampleAnnotation,
    SurfaceAnn,
    Visibility,
)

from perception_dataset.constants import EXTENSION_ENUM, SENSOR_ENUM, T4_FORMAT_DIRECTORY_NAME
from perception_dataset.t4_dataset.table_handler import TableHandler
from perception_dataset.utils.calculate_num_points import calculate_num_points
from perception_dataset.utils.transform import compose_transform


class AnnotationFilesGenerator:
    def __init__(
        self,
        with_camera: bool = True,
        description: Dict[str, Dict[str, str]] = {},
        surface_categories: List[str] = [],
        label_coordinates: str = "map",
    ):
        """
        Args:
            with_camera (bool): Whether to use camera data.
            description (Dict[str, Dict[str, str]]): Description of the dataset.
            surface_categories (List[str]): List of surface categories.
            label_coordinates (str): Coordinate system for the labels. Can be "map" or "lidar".
        """
        self._with_lidarseg = description.get("with_lidarseg", False)
        assert label_coordinates in {
            "map",
            "lidar",
        }, "label_coordinates must be either 'map' or 'lidar'"
        self._label_coordinates = label_coordinates
        self._attribute_table = TableHandler(Attribute)
        self._category_table = TableHandler(Category)
        self._instance_table = TableHandler(Instance)
        self._visibility_table = TableHandler(Visibility)
        for level, desc in description.get(
            "visibility",
            {
                "v0-40": "visibility of whole object is between 0 and 40%",
                "unavailable": "visibility is not available",
                "v40-60": "visibility of whole object is between 40 and 60%",
                "v60-80": "visibility of whole object is between 60 and 80%",
                "v80-100": "visibility of whole object is between 80 and 100%",
            },
        ).items():
            self._visibility_table.insert_into_table(
                level=level,
                description=desc,
            )
        self._sample_annotation_table = TableHandler(SampleAnnotation)
        self._object_ann_table = TableHandler(ObjectAnn)
        self._surface_ann_table = TableHandler(SurfaceAnn)

        self._instance_token_to_annotation_token_list: Dict[str, List[str]] = defaultdict(list)

        if with_camera:
            self._camera2idx = description.get("camera_index")
        else:
            self._camera2idx = None
        self._with_lidar = description.get("with_lidar", True)
        self._surface_categories: List[str] = surface_categories

        if self._with_lidarseg:
            assert self._with_lidar, "with_lidar must be set if with_lidarseg is set!"

    def save_tables(self, anno_dir: str):
        for cls_attr in self.__dict__.values():
            if isinstance(cls_attr, TableHandler):
                print(f"{cls_attr.schema_name}: #rows {len(cls_attr)}")
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

        t4_dataset = Tier4(data_root=input_dir, verbose=False)
        try:
            if "LIDAR_TOP" in t4_dataset.sample[0].data:
                lidar_sensor_channel = SENSOR_ENUM.LIDAR_TOP.value["channel"]
            else:
                lidar_sensor_channel = SENSOR_ENUM.LIDAR_CONCAT.value["channel"]
        except KeyError as e:
            print(e)

        # TODO (KokSeang): Support lidarseg and other annotations at the same time
        if self._with_lidarseg:
            self._convert_lidarseg_scene_annotations(
                scene_anno_dict=scene_anno_dict,
                t4_dataset=t4_dataset,
                anno_dir=anno_dir,
                lidar_sensor_channel=lidar_sensor_channel,
            )
        else:
            self._convert_scene_annotations(
                scene_anno_dict=scene_anno_dict, dataset_name=dataset_name, t4_dataset=t4_dataset
            )

        self._attribute_table.save_json(anno_dir)
        self._category_table.save_json(anno_dir)
        self._instance_table.save_json(anno_dir)
        self._sample_annotation_table.save_json(anno_dir)
        self._visibility_table.save_json(anno_dir)
        self._object_ann_table.save_json(anno_dir)
        self._surface_ann_table.save_json(anno_dir)
        # Skip if lidarseg is enabled since it doesn't have any bounding box annotations yet
        if self._with_lidar and not self._with_lidarseg:
            # Calculate and overwrite number of points in lidar cuboid bounding box in annotations
            calculate_num_points(output_dir, lidar_sensor_channel, self._sample_annotation_table)
            self._sample_annotation_table.save_json(anno_dir)

    def _convert_scene_annotations(
        self,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        dataset_name: str,
        t4_dataset: Tier4,
    ) -> None:
        """
        Convert scene annotations to T4 dataset format.
        :param scene_anno_dict: Scene annotations.
        :param dataset_name: Dataset name.
        :param t4_dataset: Tier4 object.
        """
        frame_index_to_sample_token: Dict[int, str] = {}
        for sample_data in t4_dataset.sample_data:
            frame_index = int((sample_data.filename.split("/")[2]).split(".")[0])
            frame_index_to_sample_token[frame_index] = sample_data.sample_token

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
            for _, sample_data in enumerate(t4_dataset.sample_data):
                if sample_data.fileformat == "png" or sample_data.fileformat == "jpg":
                    cam = sample_data.filename.split("/")[1]
                    cam_idx = self._camera2idx[cam]

                    frame_index = int((sample_data.filename.split("/")[2]).split(".")[0])
                    frame_index_to_sample_data_token[cam_idx].update(
                        {frame_index: sample_data.token}
                    )

                    hgt_wid = (sample_data.height, sample_data.width)
                    if hgt_wid != prev_wid_hgt:
                        prev_wid_hgt = hgt_wid
                        object_mask = np.zeros(hgt_wid, dtype=np.uint8)
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
            t4_dataset=t4_dataset,
        )

    def convert_annotations(
        self,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        frame_index_to_sample_token: Dict[int, str],
        dataset_name: str,
        frame_index_to_sample_data_token: Optional[List[Dict[int, str]]] = None,
        mask: Optional[List[Dict[int, str]]] = None,
        t4_dataset: Tier4 = None,
    ):
        self._convert_to_t4_format(
            scene_anno_dict=scene_anno_dict,
            frame_index_to_sample_token=frame_index_to_sample_token,
            dataset_name=dataset_name,
            frame_index_to_sample_data_token=frame_index_to_sample_data_token,
            mask=mask,
            t4_dataset=t4_dataset,
        )
        self._connect_annotations_in_scene()

    def _convert_to_t4_format(
        self,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        frame_index_to_sample_token: Dict[int, str],
        dataset_name: str,
        frame_index_to_sample_data_token: List[Dict[int, str]],
        mask: List[Dict[int, str]],
        t4_dataset: Tier4 = None,
    ):
        """Convert the annotations to the Tier4 format.

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
                category_token: str = self._category_table.get_token_from_field(
                    field_name="name", field_value=anno["category_name"]
                )
                if category_token is None:
                    category_token = self._category_table.insert_into_table(
                        name=anno["category_name"],
                        description="",
                    )
                # Instance
                instance_name = dataset_name + "::" + str(anno["instance_id"])
                instance_token: str = self._instance_table.get_token_from_field(
                    field_name="instance_name",
                    field_value=instance_name,
                )
                if not instance_token:
                    instance_token = self._instance_table.insert_into_table(
                        instance_name=instance_name,
                        category_token=category_token,
                        nbr_annotations=0,
                        first_annotation_token="",
                        last_annotation_token="",
                    )

                # Attribute
                attribute_tokens: List[str] = []
                for attr_name in anno["attribute_names"]:
                    attr_token = self._attribute_table.get_token_from_field(
                        field_name="name",
                        field_value=attr_name,
                    )
                    if not attr_token:
                        attr_token = self._attribute_table.insert_into_table(
                            name=attr_name,
                            description="",
                        )
                    attribute_tokens.append(attr_token)

                # Visibility
                visibility_token: str = self._visibility_table.get_token_from_field(
                    field_name="level", field_value=anno.get("visibility_name", "unavailable")
                )
                if not visibility_token:
                    visibility_token = self._visibility_table.insert_into_table(
                        level=anno.get("visibility_name", "unavailable"),
                        description="",
                    )

                # Sample Annotation
                if "three_d_bbox" in anno.keys():
                    sample_token: str = frame_index_to_sample_token[frame_index]
                    anno_three_d_bbox: Dict[str, float] = self._transform_cuboid(
                        anno["three_d_bbox"], sample_token=sample_token, t4_dataset=t4_dataset
                    )
                    sample_annotation_token: str = self._sample_annotation_table.insert_into_table(
                        sample_token=sample_token,
                        instance_token=instance_token,
                        attribute_tokens=attribute_tokens,
                        visibility_token=visibility_token,
                        translation=tuple(anno_three_d_bbox["translation"].values()),
                        velocity=(
                            tuple(anno_three_d_bbox["velocity"].values())
                            if anno_three_d_bbox.get("velocity")
                            else None
                        ),
                        acceleration=(
                            tuple(anno_three_d_bbox["acceleration"].values())
                            if anno_three_d_bbox.get("acceleration")
                            else None
                        ),
                        size=tuple(anno_three_d_bbox["size"].values()),
                        rotation=tuple(anno_three_d_bbox["rotation"].values()),
                        num_lidar_pts=anno["num_lidar_pts"],
                        num_radar_pts=anno["num_radar_pts"],
                        automatic_annotation=False,
                        next="",
                        prev="",
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

    def _transform_cuboid(
        self,
        three_d_bbox: Dict[str, float],
        sample_token: str,
        t4_dataset: Tier4,
    ) -> Dict[str, float]:
        if self._label_coordinates == "map":
            return three_d_bbox
        elif self._label_coordinates == "lidar":
            assert (
                "translation" in three_d_bbox.keys() or "rotation" in three_d_bbox.keys()
            ), "translation and rotation must be in three_d_bbox"
            assert t4_dataset is not None, "t4_dataset must be set in _transform_cuboid"
            sample = t4_dataset.get("sample", sample_token)
            lidar_token: str = sample.data[SENSOR_ENUM.LIDAR_CONCAT.value["channel"]]

            sd_record = t4_dataset.get("sample_data", lidar_token)
            cs_record = t4_dataset.get("calibrated_sensor", sd_record.calibrated_sensor_token)
            ep_record = t4_dataset.get("ego_pose", sd_record.ego_pose_token)

            lidar_to_map_translation, lidar_to_map_rotation = compose_transform(
                trans1=cs_record.translation,
                rot1=cs_record.rotation,
                trans2=ep_record.translation,
                rot2=ep_record.rotation,
            )
            lidar_to_map_rotation_quaternion = Rotation.from_quat(
                lidar_to_map_rotation[1:] + [lidar_to_map_rotation[0]]  # [x, y, z, w]
            )

            # Transform the lidar-based-cuboid to the map coordinate system
            translation = list(three_d_bbox["translation"].values())
            r = three_d_bbox["rotation"]
            rotation = [r["x"], r["y"], r["z"], r["w"]]  # [x, y, z, w] format
            bbox_rotation_quaternion = Rotation.from_quat(rotation)

            translation_map = (
                lidar_to_map_rotation_quaternion.apply(translation) + lidar_to_map_translation
            )
            rotation_map = (
                lidar_to_map_rotation_quaternion * bbox_rotation_quaternion
            ).as_quat()  # [x, y, z, w] format

            # apply back to the three_d_bbox
            three_d_bbox["translation"] = {
                "x": translation_map[0],
                "y": translation_map[1],
                "z": translation_map[2],
            }
            three_d_bbox["rotation"] = {
                "x": rotation_map[0],
                "y": rotation_map[1],
                "z": rotation_map[2],
                "w": rotation_map[3],
            }
            return three_d_bbox

    def _clip_bbox(self, bbox: List[float], mask: Dict[str, Any]) -> List[float]:
        """Clip the bbox to the image size."""
        try:
            height, width = mask["size"]
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
            inst_rec: Instance = self._instance_table.select_record_from_token(instance_token)
            inst_rec.nbr_annotations = len(annotation_token_list)
            inst_rec.first_annotation_token = annotation_token_list[0]
            inst_rec.last_annotation_token = annotation_token_list[-1]
            self._instance_table.set_record_to_table(inst_rec)

            # set next/prev of sample_annotation
            for token_i in range(1, len(annotation_token_list)):
                prev_token: str = annotation_token_list[token_i - 1]
                cur_token: str = annotation_token_list[token_i]

                prev_rec: SampleAnnotation = (
                    self._sample_annotation_table.select_record_from_token(prev_token)
                )
                prev_rec.next = cur_token
                self._sample_annotation_table.set_record_to_table(prev_rec)

                cur_rec: SampleAnnotation = self._sample_annotation_table.select_record_from_token(
                    cur_token
                )
                cur_rec.prev = prev_token
                self._sample_annotation_table.set_record_to_table(cur_rec)

    def _convert_lidarseg_scene_annotations(
        self,
        scene_anno_dict: Dict[int, List[Dict[str, Any]]],
        lidar_sensor_channel: str,
        t4_dataset: Tier4,
        anno_dir: str,
    ) -> None:
        """
        Convert annotations for scenes in lidarseg to t4 dataset format.
        Specifically, it creates lidarseg.json and update category.json.
        :param scene_anno_dict: Dict of annotations for every scenes.
        :param lidar_sensor_channel: Lidar sensor channel name.
        :param t4_dataset: Tier4 object for this dataset.
        :param anno_dir: Annotation directory.
        """
        lidarseg_table = TableHandler(LidarSeg)

        frame_index_to_sample_data_token: Dict[int, str] = {}
        frame_index_to_sample_token: Dict[int, str] = {}
        for sample_data in t4_dataset.sample_data:
            frame_index = int((sample_data.filename.split("/")[2]).split(".")[0])
            frame_index_to_sample_token[frame_index] = sample_data.sample_token
            if lidar_sensor_channel in sample_data.filename:
                frame_index_to_sample_data_token[frame_index] = sample_data.token

        # Create lidarseg folder in
        anno_path = Path(anno_dir)
        anno_path.mkdir(parents=True, exist_ok=True)
        version_name = anno_path.name
        lidarseg_relative_path = osp.join(
            T4_FORMAT_DIRECTORY_NAME.LIDARSEG_ANNO_FOLDER.value, version_name
        )
        # We need to move the level same as anno_dir, and create "lidarseg/<version_name>" because of the design in Tier4 dataset
        lidarseg_anno_path = anno_path.parents[0] / lidarseg_relative_path
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
                # Category
                for category_name in anno["paint_categories"]:
                    self._category_table.insert_into_table(
                        name=category_name.lower(),
                        description="",
                    )

                if not self._visibility_table.get_token_from_field(
                    field_name="level", field_value=anno.get("visibility_name", "unavailable")
                ):
                    self._visibility_table.insert_into_table(
                        level=anno.get("visibility_name", "unavailable"),
                        description="",
                    )

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
                lidarseg_record: LidarSeg = lidarseg_table.select_record_from_token(
                    token=lidarseg_token
                )
                lidarseg_record.filename = osp.join(
                    lidarseg_relative_path, (lidarseg_token + EXTENSION_ENUM.BIN.value)
                )
                lidarseg_table.set_record_to_table(
                    record=lidarseg_record,
                )

        # Add a case for unpainted point cloud
        self._category_table.insert_into_table(
            name="unpainted",
            index=0,
            description="unpainted points",
        )

        lidarseg_table.save_json(anno_dir)

        # Remove the older lidarseg folder
        shutil.rmtree(T4_FORMAT_DIRECTORY_NAME.LIDARSEG_ANNO_FOLDER.value, ignore_errors=True)
