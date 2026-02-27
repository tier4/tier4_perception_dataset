import base64
from collections import defaultdict
import json
import os.path as osp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from nptyping import NDArray
import numpy as np
from pycocotools import mask as cocomask
from scipy.spatial.transform import Rotation
from t4_devkit import Tier4

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
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.transform import compose_transform

logger = configure_logger(modname=__name__)


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
        # TODO(yukke42): remove the hard coded attribute description
        self._attribute_table = AttributeTable(
            name_to_description={},
            default_value="",
        )
        # TODO(yukke42): remove the hard coded category description
        self._category_table = CategoryTable(
            name_to_description={}, default_value="", lidarseg=self._with_lidarseg
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
            # Create reverse mapping from sensor_id to camera channel name
            self._idx2camera = {v: k for k, v in self._camera2idx.items()} if self._camera2idx else None
        else:
            self._camera2idx = None
            self._idx2camera = None
        self._with_lidar = description.get("with_lidar", True)
        self._surface_categories: List[str] = surface_categories

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
        only_annotation_frame: bool = False,
    ):
        anno_dir = osp.join(output_dir, "annotation")
        if not osp.exists(anno_dir):
            raise ValueError(f"Annotations files doesn't exist in {anno_dir}")

        # Fix sample_data.json to handle entries with None sample_token
        # Keep is_key_frame: false data even if sample_token is None to maintain token consistency
        def fix_sample_data_json(json_path: str, location: str, keep_keyframe_false: bool = True, for_tier4_read: bool = False):
            stripped_unknown_keys = False
            if osp.exists(json_path):
                with open(json_path, "r") as f:
                    sample_data_list = json.load(f)
                # For Tier4 compatibility when reading: strip keys t4_devkit SampleData doesn't accept (e.g. uncertainty)
                if for_tier4_read:
                    for sample_data in sample_data_list:
                        if "uncertainty" in sample_data:
                            sample_data.pop("uncertainty")
                            stripped_unknown_keys = True
                # For Tier4 compatibility when reading: convert None sample_token to empty string for is_key_frame: false data
                # For output: keep None sample_token as null for is_key_frame: false data
                # Remove entries with None sample_token only if they are not is_key_frame: false
                # IMPORTANT: Do not modify is_key_frame: false entries that already have a sample_token value
                # to preserve their original state, especially when order is incorrect
                filtered_sample_data_list = []
                removed_count = 0
                modified_count = 0
                for sample_data in sample_data_list:
                    sample_token = sample_data.get("sample_token")
                    is_key_frame = sample_data.get("is_key_frame", True)
                    
                    # Ensure is_key_frame is boolean (handle string "false" or "False")
                    if isinstance(is_key_frame, str):
                        is_key_frame = is_key_frame.lower() == "true"
                    
                    # Do not modify is_key_frame: false entries that already have a sample_token
                    # This preserves their original state when order is incorrect
                    if not is_key_frame and sample_token is not None:
                        filtered_sample_data_list.append(sample_data)
                        continue
                    
                    if sample_token is None:
                        if keep_keyframe_false and not is_key_frame:
                            # For Tier4 class reading, convert None to empty string temporarily
                            # For output, keep None as null
                            if for_tier4_read:
                                sample_data["sample_token"] = ""
                            else:
                                sample_data["sample_token"] = None
                            modified_count += 1
                            filtered_sample_data_list.append(sample_data)
                        else:
                            # Remove entries with None sample_token that are not is_key_frame: false
                            removed_count += 1
                    else:
                        filtered_sample_data_list.append(sample_data)
                
                if removed_count > 0:
                    logger.info(
                        f"Removed {removed_count} entries with None sample_token from {location} sample_data.json"
                    )
                if modified_count > 0:
                    if for_tier4_read:
                        logger.info(
                            f"Temporarily converted {modified_count} is_key_frame: false entries with null sample_token to empty string for Tier4 reading in {location} sample_data.json"
                        )
                    else:
                        logger.info(
                            f"Kept {modified_count} is_key_frame: false entries with null sample_token in {location} sample_data.json"
                        )
                if len(filtered_sample_data_list) != len(sample_data_list) or modified_count > 0 or stripped_unknown_keys:
                    with open(json_path, "w") as f:
                        json.dump(filtered_sample_data_list, f, indent=4, ensure_ascii=False)

        # Fix input_dir's sample_data.json temporarily for Tier4 to read (will restore later)
        # Convert None sample_token to empty string temporarily for Tier4 class compatibility
        # Save original content and restore it after Tier4 class reads the file
        # Always do this temporary fix to allow Tier4 class to read the file, regardless of only_annotation_frame
        input_anno_dir = osp.join(input_dir, "annotation")
        input_sample_data_json_path = osp.join(input_anno_dir, "sample_data.json")
        input_sample_data_original = None
        input_sample_data_modified = False
        if osp.exists(input_sample_data_json_path):
            # Read original content as bytes to preserve exact format
            with open(input_sample_data_json_path, "rb") as f:
                input_sample_data_original = f.read()
            # Temporarily fix for Tier4 to read - convert None to empty string for Tier4 class compatibility
            # This ensures Tier4 can read the file (Tier4 class requires sample_token to be str, not None)
            # We will restore the original file in the finally block
            try:
                fix_sample_data_json(input_sample_data_json_path, "input", keep_keyframe_false=True, for_tier4_read=True)
                input_sample_data_modified = True
            except Exception as e:
                # If fixing fails, restore original immediately
                if input_sample_data_original is not None:
                    with open(input_sample_data_json_path, "wb") as f:
                        f.write(input_sample_data_original)
                raise

        # Fix output_dir's sample_data.json
        # Keep is_key_frame: false entries with null sample_token
        # If only_annotation_frame is False, do not modify sample_data.json to preserve original state
        # The output sample_data.json is already copied from input, so we keep it as-is when only_annotation_frame is False
        if only_annotation_frame:
            output_sample_data_json_path = osp.join(anno_dir, "sample_data.json")
            fix_sample_data_json(output_sample_data_json_path, "output", keep_keyframe_false=True, for_tier4_read=False)

        # Fix sample_annotation.json to remove entries with NaN translation values
        def fix_sample_annotation_json(json_path: str, location: str):
            if osp.exists(json_path):
                with open(json_path, "r") as f:
                    sample_annotation_list = json.load(f)
                # Filter out entries with NaN translation values
                filtered_sample_annotation_list = []
                removed_count = 0
                for ann in sample_annotation_list:
                    translation = ann.get("translation")
                    if translation is None:
                        removed_count += 1
                        continue
                    if isinstance(translation, list) and len(translation) == 3:
                        if any(
                            not isinstance(val, (int, float)) or np.isnan(val) for val in translation
                        ):
                            removed_count += 1
                            continue
                    filtered_sample_annotation_list.append(ann)
                if removed_count > 0:
                    logger.warning(
                        f"Removed {removed_count} entries with NaN translation values from {location} sample_annotation.json"
                    )
                    with open(json_path, "w") as f:
                        json.dump(filtered_sample_annotation_list, f, indent=4)

        output_sample_annotation_json_path = osp.join(anno_dir, "sample_annotation.json")
        fix_sample_annotation_json(output_sample_annotation_json_path, "output")

        # Fix log.json to convert date_captured to data_captured (temporarily)
        def fix_log_json_temporarily(json_path: str):
            """Temporarily fix log.json and return original content to restore later"""
            if not osp.exists(json_path):
                return None
            with open(json_path, "r") as f:
                original_content = f.read()
                log_list = json.loads(original_content)
            # Convert date_captured to data_captured
            modified = False
            for log_entry in log_list:
                if "date_captured" in log_entry and "data_captured" not in log_entry:
                    log_entry["data_captured"] = log_entry.pop("date_captured")
                    modified = True
            if modified:
                with open(json_path, "w") as f:
                    json.dump(log_list, f, indent=2)
                return original_content
            return None

        # Don't modify input_dir's log.json to avoid modifying source files
        # input_dir files are already copied to output_dir by _copy_data
        input_log_json_path = osp.join(input_anno_dir, "log.json")
        input_log_original = None  # Don't modify input files

        output_log_json_path = osp.join(anno_dir, "log.json")
        output_log_original = fix_log_json_temporarily(output_log_json_path)

        # Fix object_ann.json to convert autolabel_metadata format from {"models": [...]} to [...]
        # t4_devkit expects autolabel_metadata to be a list, not a dict with "models" key.
        # Also normalize uncertainty to [0, 1] for t4_devkit (spec expects 0.0-1.0); original is restored in finally.
        # Ensure each model has required fields (e.g. score) for t4_devkit AutolabelModel.
        def _normalize_uncertainty_for_t4(models: list) -> None:
            """Clamp each model's uncertainty to [0.0, 1.0] in-place (t4_devkit expects >= 0 and <= 1)."""
            for model in models:
                if "uncertainty" not in model or model["uncertainty"] is None:
                    continue
                try:
                    u = float(model["uncertainty"])
                    if u < 0.0:
                        model["uncertainty"] = 0.0
                    elif u > 1.0:
                        model["uncertainty"] = 1.0
                except (TypeError, ValueError):
                    pass

        def _ensure_autolabel_model_required_fields(models: list) -> None:
            """Add required fields for t4_devkit AutolabelModel when missing (e.g. score)."""
            for model in models:
                if not isinstance(model, dict):
                    continue
                if "score" not in model or model["score"] is None:
                    model["score"] = 0.0
                if "name" not in model or model["name"] is None:
                    model["name"] = ""

        def fix_autolabel_metadata_for_t4(record: dict) -> bool:
            """Apply autolabel_metadata format/range fixes for t4_devkit. Returns True if modified."""
            autolabel_metadata = record.get("autolabel_metadata")
            if autolabel_metadata is None:
                return False
            modified = False
            if isinstance(autolabel_metadata, dict) and "models" in autolabel_metadata:
                record["autolabel_metadata"] = autolabel_metadata["models"]
                _normalize_uncertainty_for_t4(record["autolabel_metadata"])
                _ensure_autolabel_model_required_fields(record["autolabel_metadata"])
                modified = True
            elif isinstance(autolabel_metadata, list):
                _normalize_uncertainty_for_t4(autolabel_metadata)
                _ensure_autolabel_model_required_fields(autolabel_metadata)
                modified = True
            return modified

        def fix_object_ann_json(json_path: str):
            """Fix autolabel_metadata format and normalize for t4_devkit."""
            if not osp.exists(json_path):
                return None
            with open(json_path, "r") as f:
                original_content = f.read()
                object_ann_list = json.loads(original_content)
            modified = False
            if isinstance(object_ann_list, list):
                for record in object_ann_list:
                    if fix_autolabel_metadata_for_t4(record):
                        modified = True
            if modified:
                with open(json_path, "w") as f:
                    json.dump(object_ann_list, f, indent=4)
                return original_content
            return None

        input_object_ann_json_path = osp.join(input_anno_dir, "object_ann.json")
        input_object_ann_original = fix_object_ann_json(input_object_ann_json_path)

        # Keys t4_devkit SurfaceAnn does not accept (strip before Tier4 reads, restore in finally)
        _SURFACE_ANN_STRIP_KEYS_FOR_T4 = ("instance_token", "attribute_tokens")

        def fix_surface_ann_json(json_path: str):
            """Fix autolabel_metadata and strip unknown keys for t4_devkit SurfaceAnn."""
            if not osp.exists(json_path):
                return None
            with open(json_path, "r") as f:
                original_content = f.read()
                surface_ann_list = json.loads(original_content)
            modified = False
            if isinstance(surface_ann_list, list):
                for surface_ann in surface_ann_list:
                    if not isinstance(surface_ann, dict):
                        continue
                    if fix_autolabel_metadata_for_t4(surface_ann):
                        modified = True
                    for key in _SURFACE_ANN_STRIP_KEYS_FOR_T4:
                        if key in surface_ann:
                            surface_ann.pop(key)
                            modified = True
            if modified:
                with open(json_path, "w") as f:
                    json.dump(surface_ann_list, f, indent=4)
                return original_content
            return None

        input_surface_ann_json_path = osp.join(input_anno_dir, "surface_ann.json")
        input_surface_ann_original = fix_surface_ann_json(input_surface_ann_json_path)

        try:
            t4_dataset = Tier4(data_root=input_dir, verbose=False)
        finally:
            # Restore original files - always restore to ensure original file is not modified
            if input_sample_data_original is not None and input_sample_data_modified:
                with open(input_sample_data_json_path, "wb") as f:
                    f.write(input_sample_data_original)
            if input_log_original is not None:
                with open(input_log_json_path, "w") as f:
                    f.write(input_log_original)
            if output_log_original is not None:
                with open(output_log_json_path, "w") as f:
                    f.write(output_log_original)
            if input_object_ann_original is not None:
                with open(input_object_ann_json_path, "w") as f:
                    f.write(input_object_ann_original)
            if input_surface_ann_original is not None:
                with open(input_surface_ann_json_path, "w") as f:
                    f.write(input_surface_ann_original)
        try:
            if "LIDAR_TOP" in t4_dataset.sample[0].data:
                lidar_sensor_channel = SENSOR_ENUM.LIDAR_TOP.value["channel"]
            else:
                lidar_sensor_channel = SENSOR_ENUM.LIDAR_CONCAT.value["channel"]
        except (KeyError, IndexError) as e:
            print(e)
            # Default to LIDAR_CONCAT if detection fails
            lidar_sensor_channel = SENSOR_ENUM.LIDAR_CONCAT.value["channel"]

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
                scene_anno_dict=scene_anno_dict,
                dataset_name=dataset_name,
                t4_dataset=t4_dataset,
                lidar_sensor_channel=lidar_sensor_channel,
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
        lidar_sensor_channel: str,
    ) -> None:
        """
        Convert scene annotations to T4 dataset format.
        :param scene_anno_dict: Scene annotations.
        :param dataset_name: Dataset name.
        :param t4_dataset: Tier4 object.
        :param lidar_sensor_channel: LiDAR sensor channel name (e.g., "LIDAR_TOP" or "LIDAR_CONCAT").
        """
        # Use only LiDAR sample_data to create frame_index_to_sample_token mapping
        # This ensures that segmentation annotations are correctly mapped to LiDAR frames
        frame_index_to_sample_token: Dict[int, str] = {}
        for sample_data in t4_dataset.sample_data:
            # Skip is_key_frame: false data as they have non-integer frame indices (e.g., '00000-01')
            if not sample_data.is_key_frame:
                continue
            # Only use LiDAR sample_data to avoid frame synchronization issues
            if lidar_sensor_channel in sample_data.filename:
                frame_index = int((sample_data.filename.split("/")[2]).split(".")[0])
                # Skip if sample_token is None (e.g., when KeyFrameConsistencyResolver has removed the sample)
                if sample_data.sample_token is not None:
                    frame_index_to_sample_token[frame_index] = sample_data.sample_token
        
        # Debug: print frame_index_to_sample_token for frames with annotations
        if scene_anno_dict:
            for frame_index in sorted(scene_anno_dict.keys()):
                if frame_index not in frame_index_to_sample_token:
                    print(f"WARNING: frame_index {frame_index} in annotations but not in frame_index_to_sample_token")

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
                # Skip is_key_frame: false data as they have non-integer frame indices (e.g., '00000-01')
                if not sample_data.is_key_frame:
                    continue
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
            # Use frame_index directly since frame_index_to_sample_token uses frame_index as key
            if frame_index not in frame_index_to_sample_token:
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
                    # Skip if frame_index is not in frame_index_to_sample_token or sample_token is None
                    if frame_index not in frame_index_to_sample_token:
                        continue
                    sample_token: str = frame_index_to_sample_token[frame_index]
                    # Skip if sample_token is None
                    if sample_token is None:
                        continue
                    # Validate three_d_bbox before processing
                    three_d_bbox = anno["three_d_bbox"]
                    if three_d_bbox is None:
                        continue
                    if "translation" not in three_d_bbox or three_d_bbox["translation"] is None:
                        continue
                    translation = three_d_bbox["translation"]
                    if not isinstance(translation, dict) or not all(
                        key in translation for key in ["x", "y", "z"]
                    ):
                        continue
                    # Check for NaN values
                    if any(
                        not isinstance(translation[key], (int, float))
                        or np.isnan(translation[key])
                        for key in ["x", "y", "z"]
                    ):
                        continue
                    # print(anno["three_d_bbox"])
                    try:
                        anno_three_d_bbox: Dict[str, float] = self._transform_cuboid(
                            three_d_bbox, sample_token=sample_token, t4_dataset=t4_dataset
                        )
                        # Validate transformed values
                        if anno_three_d_bbox is None:
                            continue
                        transformed_translation = anno_three_d_bbox.get("translation")
                        if transformed_translation is None:
                            continue
                        if not isinstance(transformed_translation, dict) or not all(
                            key in transformed_translation for key in ["x", "y", "z"]
                        ):
                            continue
                        # Check for NaN values after transformation
                        if any(
                            not isinstance(transformed_translation[key], (int, float))
                            or np.isnan(transformed_translation[key])
                            for key in ["x", "y", "z"]
                        ):
                            logger.warning(
                                f"Skipping annotation with NaN translation values for instance {instance_token}"
                            )
                            continue
                    except (ValueError, KeyError, AttributeError) as e:
                        logger.warning(
                            f"Skipping annotation due to transformation error for instance {instance_token}: {e}"
                        )
                        continue

                    sample_annotation_token: str = self._sample_annotation_table.insert_into_table(
                        sample_token=sample_token,
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
                    # Get sample_token from frame_index (LiDAR-based)
                    # Use frame_index directly (not frame_index - min_frame_index) since frame_index_to_sample_token uses frame_index as key
                    sample_token: str = frame_index_to_sample_token.get(frame_index)
                    if sample_token is None:
                        continue
                    # Get camera channel name from sensor_id
                    if self._idx2camera is None or sensor_id not in self._idx2camera:
                        continue
                    camera_channel = self._idx2camera[sensor_id]
                    # Get sample record and find corresponding camera sample_data_token
                    sample_record = t4_dataset.get("sample", sample_token)
                    if camera_channel not in sample_record.data:
                        continue
                    camera_sample_data_token = sample_record.data[camera_channel]
                    # Get camera sample_data record for mask dimensions
                    camera_sample_data = t4_dataset.get("sample_data", camera_sample_data_token)
                    # Create mask if needed
                    if frame_index not in mask[sensor_id]:
                        object_mask = np.zeros((camera_sample_data.height, camera_sample_data.width), dtype=np.uint8)
                        object_mask = cocomask.encode(np.asfortranarray(object_mask))
                        object_mask["counts"] = base64.b64encode(object_mask["counts"]).decode("ascii")
                        mask[sensor_id][frame_index] = object_mask
                    anno_two_d_box: List[float] = (
                        self._clip_bbox(anno["two_d_box"], mask[sensor_id][frame_index])
                        if "two_d_box" in anno.keys()
                        else None
                    )
                    self._object_ann_table.insert_into_table(
                        sample_data_token=camera_sample_data_token,
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
                    # Get sample_token from frame_index (LiDAR-based)
                    # Use frame_index directly (not frame_index - min_frame_index) since frame_index_to_sample_token uses frame_index as key
                    sample_token: str = frame_index_to_sample_token.get(frame_index)
                    if sample_token is None:
                        continue
                    # Get camera channel name from sensor_id
                    if self._idx2camera is None or sensor_id not in self._idx2camera:
                        continue
                    camera_channel = self._idx2camera[sensor_id]
                    # Get sample record and find corresponding camera sample_data_token
                    sample_record = t4_dataset.get("sample", sample_token)
                    if camera_channel not in sample_record.data:
                        continue
                    camera_sample_data_token = sample_record.data[camera_channel]
                    self._surface_ann_table.insert_into_table(
                        category_token=category_token,
                        mask=anno["two_d_segmentation"],
                        sample_data_token=camera_sample_data_token,
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
                "translation" in three_d_bbox.keys() and "rotation" in three_d_bbox.keys()
            ), "translation and rotation must be in three_d_bbox"
            assert t4_dataset is not None, "t4_dataset must be set in _transform_cuboid"
            
            # Validate translation values
            translation = three_d_bbox.get("translation")
            if translation is None or not isinstance(translation, dict):
                raise ValueError("translation must be a dictionary")
            if not all(key in translation for key in ["x", "y", "z"]):
                raise ValueError("translation must contain x, y, z keys")
            if any(
                not isinstance(translation[key], (int, float)) or np.isnan(translation[key])
                for key in ["x", "y", "z"]
            ):
                raise ValueError("translation values must be valid numbers (not NaN)")
            
            # Validate rotation values
            rotation = three_d_bbox.get("rotation")
            if rotation is None or not isinstance(rotation, dict):
                raise ValueError("rotation must be a dictionary")
            if not all(key in rotation for key in ["x", "y", "z", "w"]):
                raise ValueError("rotation must contain x, y, z, w keys")
            if any(
                not isinstance(rotation[key], (int, float)) or np.isnan(rotation[key])
                for key in ["x", "y", "z", "w"]
            ):
                raise ValueError("rotation values must be valid numbers (not NaN)")
            
            sample = t4_dataset.get("sample", sample_token)
            lidar_token: str = sample.data[SENSOR_ENUM.LIDAR_CONCAT.value["channel"]]

            sd_record = t4_dataset.get("sample_data", lidar_token)
            cs_record = t4_dataset.get("calibrated_sensor", sd_record.calibrated_sensor_token)
            ep_record = t4_dataset.get("ego_pose", sd_record.ego_pose_token)
            # print(f"cs_record: {cs_record}")
            # print(f"cs_record.rotation: {list(cs_record.rotation)}")
            # print(f"ep_record.rotation: {list(ep_record.rotation)}")
            lidar_to_map_translation, lidar_to_map_rotation = compose_transform(
                trans1=cs_record.translation,
                rot1=list(cs_record.rotation),
                trans2=ep_record.translation,
                rot2=list(ep_record.rotation),
            )
            lidar_to_map_rotation_quaternion = Rotation.from_quat(
                lidar_to_map_rotation[1:] + [lidar_to_map_rotation[0]]  # [x, y, z, w]
            )

            # Transform the lidar-based-cuboid to the map coordinate system
            translation_list = list(translation.values())
            r = rotation
            rotation_list = [r["x"], r["y"], r["z"], r["w"]]  # [x, y, z, w] format
            bbox_rotation_quaternion = Rotation.from_quat(rotation_list)

            translation_map = (
                lidar_to_map_rotation_quaternion.apply(translation_list) + lidar_to_map_translation
            )
            rotation_map = (
                lidar_to_map_rotation_quaternion * bbox_rotation_quaternion
            ).as_quat()  # [x, y, z, w] format

            # Check for NaN values after transformation
            if np.any(np.isnan(translation_map)) or np.any(np.isnan(rotation_map)):
                raise ValueError("Transformation resulted in NaN values")

            # apply back to the three_d_bbox
            three_d_bbox["translation"] = {
                "x": float(translation_map[0]),
                "y": float(translation_map[1]),
                "z": float(translation_map[2]),
            }
            three_d_bbox["rotation"] = {
                "x": float(rotation_map[0]),
                "y": float(rotation_map[1]),
                "z": float(rotation_map[2]),
                "w": float(rotation_map[3]),
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
        lidarseg_table = LidarSegTable()

        frame_index_to_sample_data_token: Dict[int, str] = {}
        frame_index_to_sample_token: Dict[int, str] = {}
        for sample_data in t4_dataset.sample_data:
            # Skip is_key_frame: false data as they have non-integer frame indices (e.g., '00000-01')
            if not sample_data.is_key_frame:
                continue
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
            # Use frame_index directly since frame_index_to_sample_token uses frame_index as key
            if frame_index not in frame_index_to_sample_token:
                print(f"frame_index {frame_index} in annotation.json is not in sample_token")
                continue

            sample_data_token = frame_index_to_sample_data_token.get(frame_index, None)
            if sample_data_token is None:
                raise ValueError(f"sample_data doesn't have {lidar_sensor_channel}!")

            # All tmp lidarseg folders before moving
            for anno in anno_list:
                # Category
                for category_name in anno["paint_categories"]:
                    self._category_table.get_token_from_name(
                        name=category_name.lower()
                    )  # Make category name lowercase

                # Visibility
                self._visibility_table.get_token_from_level(
                    level=anno.get("visibility_name", "none")
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
                lidarseg_record = lidarseg_table.select_record_from_token(token=lidarseg_token)
                lidarseg_record.filename = osp.join(
                    lidarseg_relative_path, (lidarseg_token + EXTENSION_ENUM.BIN.value)
                )
                lidarseg_table.set_record_to_table(
                    record=lidarseg_record,
                )

        # Add a case for unpainted point cloud
        self._category_table.add_category_to_record(
            name="unpainted", index=0, description="unpainted points"
        )

        lidarseg_table.save_json(anno_dir)

        # Remove the older lidarseg folder
        shutil.rmtree(T4_FORMAT_DIRECTORY_NAME.LIDARSEG_ANNO_FOLDER.value, ignore_errors=True)
