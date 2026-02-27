import copy
import json
import os.path as osp
import numpy as np
from t4_devkit import Tier4

from perception_dataset.t4_dataset.classes.sample_annotation import (
    SampleAnnotationRecord,
    SampleAnnotationTable,
)
from perception_dataset.utils import box_np_ops
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


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


def calculate_num_points(
    dataroot: str, lidar_sensor_channel: str, annotation_table: SampleAnnotationTable
):
    """Calcluate number of points in each box and overwrite the annotation table"""
    # Fix sample_data.json to remove entries with None sample_token
    # IMPORTANT: Keep is_key_frame: false entries even if sample_token is None to preserve token consistency
    # Temporarily convert None to empty string for Tier4 class compatibility, then restore after processing
    anno_dir = osp.join(dataroot, "annotation")
    sample_data_json_path = osp.join(anno_dir, "sample_data.json")
    sample_data_original = None
    sample_data_modified = False
    if osp.exists(sample_data_json_path):
        # Read original content as bytes to preserve exact format
        with open(sample_data_json_path, "rb") as f:
            sample_data_original = f.read()
        
        with open(sample_data_json_path, "r") as f:
            sample_data_list = json.load(f)
        
        # Track which entries need to be restored
        entries_to_restore = []
        
        # Filter out entries with None sample_token, but keep is_key_frame: false entries
        # Temporarily convert None to empty string for is_key_frame: false entries for Tier4 compatibility
        filtered_sample_data_list = []
        removed_count = 0
        for sample_data in sample_data_list:
            sample_token = sample_data.get("sample_token")
            is_key_frame = sample_data.get("is_key_frame", True)
            
            # Ensure is_key_frame is boolean (handle string "false" or "False")
            if isinstance(is_key_frame, str):
                is_key_frame = is_key_frame.lower() == "true"
            
            # Keep is_key_frame: false entries even if sample_token is None
            if not is_key_frame:
                # Temporarily convert None to empty string for Tier4 class compatibility
                if sample_token is None:
                    sample_data["sample_token"] = ""
                    entries_to_restore.append(sample_data)
                filtered_sample_data_list.append(sample_data)
            elif sample_token is not None:
                filtered_sample_data_list.append(sample_data)
            else:
                removed_count += 1
        
        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} entries with None sample_token (excluding is_key_frame: false) from sample_data.json"
            )
        
        # Strip keys t4_devkit SampleData doesn't accept (e.g. uncertainty) before Tier4 reads
        for sd in filtered_sample_data_list:
            sd.pop("uncertainty", None)
        # Write modified list for Tier4 class to read
        with open(sample_data_json_path, "w") as f:
            json.dump(filtered_sample_data_list, f, indent=4)
        sample_data_modified = True

    # Fix object_ann.json to remove entries with None bbox and for t4_devkit
    # Note: object_ann.json can be empty initially (2D bbox is generated from polygon/segmentation)
    # but t4_devkit cannot handle entries with bbox=None. Uncertainty is normalized to [0,1] for t4
    # and restored to original values in finally.
    object_ann_json_path = osp.join(anno_dir, "object_ann.json")
    object_ann_output_to_restore = None  # list to write back in finally (original uncertainty)
    if osp.exists(object_ann_json_path):
        with open(object_ann_json_path, "r") as f:
            object_ann_list = json.load(f)
        # Filter out entries with None bbox (t4_devkit expects roi to be a 4-element tuple)
        # Also fix autolabel_metadata format: convert {"models": [...]} to [...] (t4_devkit expects list format)
        if isinstance(object_ann_list, list) and len(object_ann_list) > 0:
            filtered_object_ann_list = []
            for object_ann in object_ann_list:
                if object_ann.get("bbox") is not None:
                    # Fix autolabel_metadata format if it exists
                    autolabel_metadata = object_ann.get("autolabel_metadata")
                    if autolabel_metadata is not None and isinstance(autolabel_metadata, dict):
                        # Convert {"models": [...]} to [...]
                        if "models" in autolabel_metadata:
                            object_ann["autolabel_metadata"] = autolabel_metadata["models"]
                    filtered_object_ann_list.append(object_ann)
            
            if len(filtered_object_ann_list) != len(object_ann_list):
                logger.info(
                    f"Removed {len(object_ann_list) - len(filtered_object_ann_list)} entries with None bbox from object_ann.json"
                )
            
            # Keep version with original uncertainty for restoration in finally
            object_ann_output_to_restore = filtered_object_ann_list
            # For t4_devkit: write copy with uncertainty clamped to [0, 1] and required fields (score, name)
            t4_object_ann_list = copy.deepcopy(filtered_object_ann_list)
            for object_ann in t4_object_ann_list:
                am = object_ann.get("autolabel_metadata")
                if isinstance(am, list):
                    _normalize_uncertainty_for_t4(am)
                    _ensure_autolabel_model_required_fields(am)
            with open(object_ann_json_path, "w") as f:
                json.dump(t4_object_ann_list, f, indent=4)

    # Fix log.json to convert date_captured to data_captured (temporarily)
    log_json_path = osp.join(anno_dir, "log.json")
    log_original_content = None
    if osp.exists(log_json_path):
        with open(log_json_path, "r") as f:
            log_original_content = f.read()
            log_list = json.loads(log_original_content)
        # Convert date_captured to data_captured
        modified = False
        for log_entry in log_list:
            if "date_captured" in log_entry and "data_captured" not in log_entry:
                log_entry["data_captured"] = log_entry.pop("date_captured")
                modified = True
        if modified:
            with open(log_json_path, "w") as f:
                json.dump(log_list, f, indent=2)

    try:
        t4_dataset = Tier4(data_root=dataroot, verbose=False)
        for sample in t4_dataset.sample:
            if lidar_sensor_channel not in sample.data:
                continue
            lidar_token = sample.data[lidar_sensor_channel]

            # Get the annotation tokens from the sample
            ann_tokens = sample.ann_3ds
            if not ann_tokens:
                continue

            # Get sample data with specific annotation tokens to maintain token mapping
            lidar_path, boxes, _ = t4_dataset.get_sample_data(
                lidar_token, selected_ann_tokens=ann_tokens
            )

            points = np.fromfile(lidar_path, dtype=np.float32)
            points = points.reshape(-1, 5)

            # taken from awml_det3d/dataset_converter/t4dataset_converter.py
            locs = np.array([b.position for b in boxes]).reshape(-1, 3)
            dims = np.array([b.size for b in boxes]).reshape(-1, 3)
            rots = np.array([b.rotation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

            indices = box_np_ops.points_in_rbbox(
                points[:, :3],
                gt_boxes[:, :7],
            )
            num_points = indices.sum(0)

            for ann_token, box, num in zip(ann_tokens, boxes, num_points):
                # Create new record with num_lidar_pts and overwrite the original one
                record: SampleAnnotationRecord = annotation_table._token_to_record[ann_token]
                new_record = SampleAnnotationRecord(
                    sample_token=record._sample_token,
                    instance_token=record._instance_token,
                    attribute_tokens=record._attribute_tokens,
                    visibility_token=record._visibility_token,
                    translation=record._translation,
                    velocity=record._velocity,
                    acceleration=record._acceleration,
                    size=record._size,
                    rotation=record._rotation,
                    num_lidar_pts=int(num),
                    num_radar_pts=record._num_radar_pts,
                )
                new_record._token = ann_token  # overwrite record token with old one
                annotation_table._token_to_record[ann_token] = new_record

        # connect next/prev tokens
        for instance in t4_dataset.instance:
            if instance.nbr_annotations == 0:
                continue
            try:
                prev_sample_data: str = annotation_table._token_to_record[
                    instance.first_annotation_token
                ]
                annotation_data_list = [
                    v
                    for v in annotation_table._token_to_record.values()
                    if v._instance_token == instance.token
                ]
                annotation_data_list[0].prev = ""
                for sample_data_i in range(1, len(annotation_data_list)):
                    cur_sample_data: str = annotation_data_list[sample_data_i]
                    if prev_sample_data._instance_token != cur_sample_data._instance_token:
                        prev_sample_data.next_token = ""
                        cur_sample_data.prev_token = ""
                    else:
                        prev_sample_data.next_token = cur_sample_data.token
                        cur_sample_data.prev_token = prev_sample_data.token
                    prev_sample_data: str = cur_sample_data
            except KeyError as e:
                logger.error(f"no key {e} in annotation table")
    finally:
        # Restore original sample_data.json file to preserve is_key_frame: false entries with None sample_token
        if sample_data_original is not None and sample_data_modified:
            with open(sample_data_json_path, "wb") as f:
                f.write(sample_data_original)
        # Restore original log.json file
        if log_original_content is not None:
            with open(log_json_path, "w") as f:
                f.write(log_original_content)
        # Restore object_ann.json with original uncertainty (t4_devkit was given normalized [0,1] values)
        if object_ann_output_to_restore is not None:
            with open(object_ann_json_path, "w") as f:
                json.dump(object_ann_output_to_restore, f, indent=4)
