import argparse
import json
import os.path as osp
from pathlib import Path


class KeyFrameConsistencyResolver:
    def inspect_and_fix_t4_segment(self, segment_path: Path, only_annotation_frame: bool = False):
        print(f"Inspecting: {str(segment_path)}")
        if not only_annotation_frame:
            # When only_annotation_frame is False, keep frames even if they have no annotations.
            # Skipping keyframe filtering preserves `sample.json` / `scene.json.nbr_samples`.
            # Also preserve is_key_frame: false entries in sample_data.json
            return

        with open(segment_path / "annotation/sample_annotation.json", "r") as f:
            sample_annotation_list = json.load(f)
        with open(segment_path / "annotation/sample_data.json", "r") as f:
            sample_data_list = json.load(f)
        with open(segment_path / "annotation/object_ann.json", "r") as f:
            object_ann_list = json.load(f)
        with open(segment_path / "annotation/sample.json", "r") as f:
            sample_list = json.load(f)
        with open(segment_path / "annotation/scene.json", "r") as f:
            scene_list = json.load(f)
        with open(segment_path / "annotation/instance.json", "r") as f:
            instance_list = json.load(f)

        lidarseg_list = []
        lidarseg_path = Path(segment_path) / "annotation/lidarseg.json"
        if lidarseg_path.exists():
            with open(lidarseg_path, "r") as f:
                lidarseg_list = json.load(f)

        # Get set of sample_tokens that have annotations (before any modifications)
        # This is needed to preserve samples with annotations
        annotated_sample_tokens = set(
            [ann["sample_token"] for ann in sample_annotation_list]
        )
        
        # Also add sample_tokens from 2D annotations (object_ann)
        # 2D annotations use sample_data_token, so we need to find the corresponding sample_token
        for object_ann in object_ann_list:
            sample_data_token = object_ann["sample_data_token"]
            # Find the sample_data with this token
            for sample_data in sample_data_list:
                if sample_data["token"] == sample_data_token:
                    sample_token = sample_data.get("sample_token")
                    if sample_token is not None:
                        annotated_sample_tokens.add(sample_token)
                    break
        
        # Also add sample_tokens from lidarseg annotations
        for lidarseg in lidarseg_list:
            sample_data_token = lidarseg["sample_data_token"]
            # Find the sample_data with this token
            for sample_data in sample_data_list:
                if sample_data["token"] == sample_data_token:
                    sample_token = sample_data.get("sample_token")
                    if sample_token is not None:
                        annotated_sample_tokens.add(sample_token)
                    break
        
        # remove sample that sample_data is not a keyframe
        self._remove_sample_of_non_keyframe(
            sample_data_list,
            sample_list,
            sample_annotation_list,
            object_ann_list,
            lidarseg_list=lidarseg_list,
        )

        # change non-keyframe sample_token in sample_data to next closest keyframe
        # BUT: Do not change sample_token for sample_data that already have the correct sample_token
        # (i.e., keyframes that have annotations should keep their original sample_token)
        self._change_sample_token_to_next_closest_keyframe(
            sample_data_list, annotated_sample_tokens=annotated_sample_tokens
        )

        self.connect_sample_data_next_prev_tokens(sample_data_list)

        # remove samples and annotations that are in conflict with the annotations.
        self._cleanup_sample_data_and_annotations(
            sample_list, sample_data_list, sample_annotation_list, object_ann_list, lidarseg_list
        )
        self._fix_instance_according_to_sample_annotation(instance_list, sample_annotation_list)
        self.connect_sample_next_prev_tokens(sample_list)

        # fix scene
        if len(scene_list) > 0 and len(sample_list) > 0:
            scene_list[0]["first_sample_token"] = sample_list[0]["token"]
            scene_list[0]["last_sample_token"] = sample_list[-1]["token"]
            scene_list[0]["nbr_samples"] = len(sample_list)

        with open(segment_path / "annotation/scene.json", "w") as f:
            json.dump(scene_list, f, indent=4)
        with open(segment_path / "annotation/sample.json", "w") as f:
            json.dump(sample_list, f, indent=4)
        with open(segment_path / "annotation/sample_data.json", "w") as f:
            json.dump(sample_data_list, f, indent=4)
        with open(segment_path / "annotation/sample_annotation.json", "w") as f:
            json.dump(sample_annotation_list, f, indent=4)
        with open(segment_path / "annotation/instance.json", "w") as f:
            json.dump(instance_list, f, indent=4)

    def _remove_sample_of_non_keyframe(
        self,
        sample_data_list: list,
        sample_list: list,
        sample_annotation_list: list,
        object_ann_list: list,
        lidarseg_list: list,
    ):
        for sample_data in sample_data_list:
            # Skip entries that are already marked as is_key_frame: false
            # These should not be modified to preserve their original state
            if not sample_data.get("is_key_frame", True):
                continue
            
            # Check for 2D annotations using sample_data token directly
            # This is important for 2D-only annotations where sample_token might be None
            corresponding_2d_annotation_direct = [
                object_ann
                for object_ann in object_ann_list
                if object_ann["sample_data_token"] == sample_data["token"]
            ]
            
            # Check for 3D annotations using sample_token
            sample_token = sample_data.get("sample_token")
            corresponding_annotation = []
            same_sample_sample_data_token_list = []
            corresponding_2d_annotation = []
            if sample_token is not None:
                corresponding_annotation = [
                    sample_anno
                    for sample_anno in sample_annotation_list
                    if sample_anno["sample_token"] == sample_token
                ]
                same_sample_sample_data_token_list = [
                    data["token"]
                    for data in sample_data_list
                    if data.get("sample_token") == sample_token
                ]
                corresponding_2d_annotation = [
                    object_ann
                    for object_ann in object_ann_list
                    if object_ann["sample_data_token"] in same_sample_sample_data_token_list
                ]
            
            # Combine direct 2D annotation check and sample-based 2D annotation check
            all_2d_annotations = corresponding_2d_annotation_direct + corresponding_2d_annotation
            # Remove duplicates based on token
            seen_tokens = set()
            unique_2d_annotations = []
            for ann in all_2d_annotations:
                if ann["token"] not in seen_tokens:
                    seen_tokens.add(ann["token"])
                    unique_2d_annotations.append(ann)
            
            corresponding_sample_list = []
            if sample_token is not None:
                corresponding_sample_list = [
                    sample for sample in sample_list if sample["token"] == sample_token
                ]
            
            corresponding_lidarseg = []
            if same_sample_sample_data_token_list:
                corresponding_lidarseg = [
                    lidarseg
                    for lidarseg in lidarseg_list
                    if lidarseg["sample_data_token"] in same_sample_sample_data_token_list
                ]
            # Also check direct lidarseg match
            corresponding_lidarseg_direct = [
                lidarseg
                for lidarseg in lidarseg_list
                if lidarseg["sample_data_token"] == sample_data["token"]
            ]
            all_lidarseg = corresponding_lidarseg + corresponding_lidarseg_direct
            seen_lidarseg_tokens = set()
            unique_lidarseg = []
            for lidarseg in all_lidarseg:
                if lidarseg["token"] not in seen_lidarseg_tokens:
                    seen_lidarseg_tokens.add(lidarseg["token"])
                    unique_lidarseg.append(lidarseg)

            # If there is no corresponding annotation, then it is not a keyframe
            if (
                len(corresponding_annotation) == 0
                and len(unique_2d_annotations) == 0
                and len(unique_lidarseg) == 0
            ):
                # Mark as non-keyframe but keep the sample_data to maintain token consistency
                # Removing sample_data would break token references in other parts of the dataset
                sample_data["is_key_frame"] = False

                # change sample_token to null for non-keyframe
                sample_data["sample_token"] = None

                # remove from sample (but keep sample_data in sample_data_list)
                if len(corresponding_sample_list) == 0:
                    continue
                corresponding_sample = corresponding_sample_list[0]
                sample_list.remove(corresponding_sample)

    def _change_sample_token_to_next_closest_keyframe(
        self, sample_data_list: list, annotated_sample_tokens: set = None
    ):
        if annotated_sample_tokens is None:
            annotated_sample_tokens = set()
        
        for sample_data in sample_data_list[:]:
            # Skip entries that are already marked as is_key_frame: false
            # These should not be modified to preserve their original state, especially when order is incorrect
            if not sample_data.get("is_key_frame", True) or not sample_data.get("is_valid", True):
                continue
            
            # If this sample_data's sample_token has annotations, keep it as-is
            # This ensures that frames with annotations are preserved correctly
            current_sample_token = sample_data.get("sample_token")
            if current_sample_token in annotated_sample_tokens:
                # This is a keyframe with annotations, keep its sample_token unchanged
                continue
            
            # For other keyframes (without annotations), change to next closest keyframe
            sample_data["sample_token"] = self._get_next_closest_keyframe(
                sample_data, sample_data_list
            )["sample_token"]
            # Keep is_key_frame: false data even if sample_token is null to maintain token consistency
            # Removing these would break token references in other parts of the dataset
            # if sample_data["sample_token"] is None:
            #     sample_data_list.remove(sample_data)

    def _cleanup_sample_data_and_annotations(
        self, sample_list, sample_data_list, sample_annotation_list, object_ann_list, lidarseg_list
    ):
        # Get set of sample_tokens that have annotations
        # When only_annotation_frame is True, we must keep samples that have annotations
        annotated_sample_tokens = set(
            [ann["sample_token"] for ann in sample_annotation_list]
        )
        
        # Also add sample_tokens from 2D annotations (object_ann)
        # 2D annotations use sample_data_token, so we need to find the corresponding sample_token
        for object_ann in object_ann_list:
            sample_data_token = object_ann["sample_data_token"]
            # Find the sample_data with this token
            for sample_data in sample_data_list:
                if sample_data["token"] == sample_data_token:
                    sample_token = sample_data.get("sample_token")
                    if sample_token is not None:
                        annotated_sample_tokens.add(sample_token)
                    break
        
        # Also add sample_tokens from lidarseg annotations
        for lidarseg in lidarseg_list:
            sample_data_token = lidarseg["sample_data_token"]
            # Find the sample_data with this token
            for sample_data in sample_data_list:
                if sample_data["token"] == sample_data_token:
                    sample_token = sample_data.get("sample_token")
                    if sample_token is not None:
                        annotated_sample_tokens.add(sample_token)
                    break
        
        # remove sample that has no corresponding sample_data
        # BUT: Do not remove samples that have annotations (when only_annotation_frame is True)
        # this is not supposed to happen since we have removed sample_data that has no corresponding annotation
        # if this happens, then it means that the sample_annotation is not consistent with sample_data
        unexpected_sample_token_list = []
        for sample_i in sample_list[:]:
            refered_sample_data_list = [
                sample_data
                for sample_data in sample_data_list
                if sample_data["sample_token"] == sample_i["token"]
            ]
            if len(refered_sample_data_list) == 0:
                # Only remove if this sample has no annotations
                # If it has annotations, we must keep it even if there's no sample_data
                # (This can happen if sample_data was incorrectly removed or not created)
                if sample_i["token"] not in annotated_sample_tokens:
                    print(f"Sample {sample_i['token']} has no corresponding sample_data and no annotations")
                    unexpected_sample_token_list.append(sample_i["token"])
                    sample_list.remove(sample_i)
                else:
                    print(f"WARNING: Sample {sample_i['token']} has annotations but no corresponding sample_data. Keeping sample to preserve annotations.")

        # remove non keyframe sample_annotation
        for cur_annotation in sample_annotation_list[:]:
            if cur_annotation["sample_token"] in unexpected_sample_token_list:
                if cur_annotation["prev"] != "":
                    prev_annotation = [
                        anno
                        for anno in sample_annotation_list
                        if anno["token"] == cur_annotation["prev"]
                    ][0]
                    prev_annotation["next"] = cur_annotation["next"]
                if cur_annotation["next"] != "":
                    next_annotation = [
                        anno
                        for anno in sample_annotation_list
                        if anno["token"] == cur_annotation["next"]
                    ][0]
                    next_annotation["prev"] = cur_annotation["prev"]

                print(f"Sample annotation {cur_annotation['token']} is removed")
                sample_annotation_list.remove(cur_annotation)

    def _fix_instance_according_to_sample_annotation(self, instance_list, sample_annotation_list):
        # fix instance according to sample_annotation
        for instance in instance_list:
            instance_annotation_list = [
                anno
                for anno in sample_annotation_list
                if anno["instance_token"] == instance["token"]
            ]
            if len(instance_annotation_list) == 0:
                instance["first_annotation_token"] = ""
                instance["last_annotation_token"] = ""
                instance["nbr_annotations"] = 0
            else:
                instance["first_annotation_token"] = instance_annotation_list[0]["token"]
                instance["last_annotation_token"] = instance_annotation_list[-1]["token"]
                instance["nbr_annotations"] = len(instance_annotation_list)

    def connect_sample_next_prev_tokens(self, sample_list: list):
        if len(sample_list) == 0:
            return
        sample_list = sorted(sample_list, key=lambda x: x["timestamp"])
        sample_list[0]["prev"] = ""
        sample_list[-1]["next"] = ""

        for sample_i in range(1, len(sample_list)):
            cur_sample = sample_list[sample_i]
            prev_sample = sample_list[sample_i - 1]

            prev_sample["next"] = cur_sample["token"]
            cur_sample["prev"] = prev_sample["token"]

    def connect_sample_data_next_prev_tokens(self, sample_data_list: list):
        if len(sample_data_list) == 0:
            return
        sensor_list = []
        [
            sensor_list.append(data["calibrated_sensor_token"])
            for data in sample_data_list
            if not data["calibrated_sensor_token"] in sensor_list
        ]

        new_sample_data_list = []
        for sensor in sensor_list:
            # Keep all sample_data including is_key_frame: false to maintain token consistency
            # is_key_frame: false data should always be kept regardless of is_valid
            # is_key_frame: true data should be kept only if is_valid is True
            sensor_sample_data_list = [
                data
                for data in sample_data_list
                if data["calibrated_sensor_token"] == sensor
                and (not data.get("is_key_frame", True) or data.get("is_valid", True))
            ]
            self.connect_sample_next_prev_tokens(sensor_sample_data_list)
            new_sample_data_list.extend(sensor_sample_data_list)

        # Update the original list in place to keep is_key_frame: false data
        sample_data_list[:] = new_sample_data_list

    def _get_next_closest_keyframe(self, current_sample_data: dict, sample_data_list: list):
        timestamp = current_sample_data["timestamp"]

        next_closest_keyframe = sample_data_list[-1]
        sample_data_keyframe_list = [
            sample_data
            for sample_data in sample_data_list
            if sample_data["is_key_frame"]
            and sample_data["timestamp"] > timestamp
            and sample_data["fileformat"] == current_sample_data["fileformat"]
            and osp.dirname(sample_data["filename"])
            == osp.dirname(current_sample_data["filename"])
        ]
        for sample_data in sample_data_keyframe_list:
            if sample_data["timestamp"] < next_closest_keyframe["timestamp"]:
                next_closest_keyframe = sample_data
        return next_closest_keyframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", help="Root of the database to fix")
    args = parser.parse_args()

    fixer = KeyFrameConsistencyResolver()
    dataset_path = Path(args.database_root)
    for item in sorted(dataset_path.iterdir()):
        if not item.is_dir() or not (item / "annotation/sample_data.json").exists():
            continue
        fixer.inspect_and_fix_t4_segment(item)
