import argparse
import json
from pathlib import Path


class KeyFrameFixer:
    def inspect_and_fix_t4_segment(self, segment_path: Path):
        print(f"Inspecting: {str(segment_path)}")

        with open(segment_path / "annotation/sample_annotation.json", "r") as f:
            sample_annotation_list = json.load(f)
        with open(segment_path / "annotation/sample_data.json", "r") as f:
            sample_data_list = json.load(f)
        with open(segment_path / "annotation/sample.json", "r") as f:
            sample_list = json.load(f)
        with open(segment_path / "annotation/scene.json", "r") as f:
            scene_list = json.load(f)

        # remove sample that sample_data is not a keyframe
        self._remove_sample_of_non_keyframe(sample_data_list, sample_list, sample_annotation_list)

        # change non-keyframe sample_token in sample_data to next closest keyframe
        self._change_sample_token_to_next_closest_keyframe(sample_data_list)

        self.connect_sample_data_next_prev_tokens(sample_data_list)

        self.connect_sample_next_prev_tokens(sample_list)

        scene_list[0]["first_sample_token"] = sample_list[0]["token"]
        scene_list[0]["last_sample_token"] = sample_list[-1]["token"]

        with open(segment_path / "annotation/scene.json", "w") as f:
            json.dump(scene_list, f, indent=4)
        with open(segment_path / "annotation/sample.json", "w") as f:
            json.dump(sample_list, f, indent=4)
        with open(segment_path / "annotation/sample_data.json", "w") as f:
            json.dump(sample_data_list, f, indent=4)

    def _remove_sample_of_non_keyframe(
        self, sample_data_list: list, sample_list: list, sample_annotation_list: list
    ):
        for sample_data in sample_data_list:
            corresponding_annotation = [
                sample_anno
                for sample_anno in sample_annotation_list
                if sample_anno["sample_token"] == sample_data["sample_token"]
            ]
            corresponding_sample_list = [
                sample for sample in sample_list if sample["token"] == sample_data["sample_token"]
            ]

            # If there is no corresponding annotation, then it is not a keyframe
            if len(corresponding_annotation) == 0:
                print(f"Sample data {sample_data['token']} has no corresponding annotation")
                # change is_keyframe to False
                sample_data["is_key_frame"] = False

                # change sample_token to "" (temporarily)
                sample_data["sample_token"] = ""

                # remove from sample
                if len(corresponding_sample_list) == 0:
                    continue
                corresponding_sample = corresponding_sample_list[0]
                sample_list.remove(corresponding_sample)

    def _change_sample_token_to_next_closest_keyframe(self, sample_data_list: list):
        for sample_data in sample_data_list[:]:
            if sample_data["is_key_frame"]:
                continue
            sample_data["sample_token"] = self._get_next_closest_keyframe(
                sample_data, sample_data_list
            )["sample_token"]
            if sample_data["sample_token"] == "":
                sample_data_list.remove(sample_data)

    def connect_sample_next_prev_tokens(self, sample_list: list):
        sample_list = sorted(sample_list, key=lambda x: x["timestamp"])
        sample_list[0]["prev"] = ""
        sample_list[-1]["next"] = ""

        for sample_i in range(1, len(sample_list)):
            cur_sample = sample_list[sample_i]
            prev_sample = sample_list[sample_i - 1]

            prev_sample["next"] = cur_sample["token"]
            cur_sample["prev"] = prev_sample["token"]

    def connect_sample_data_next_prev_tokens(self, sample_data_list: list):
        sensor_list = []
        [
            sensor_list.append(data["calibrated_sensor_token"])
            for data in sample_data_list
            if not data["calibrated_sensor_token"] in sensor_list
        ]

        new_sample_data_list = []
        for sensor in sensor_list:
            sensor_sample_data_list = [
                data for data in sample_data_list if data["calibrated_sensor_token"] == sensor
            ]
            self.connect_sample_next_prev_tokens(sensor_sample_data_list)
            new_sample_data_list.extend(sensor_sample_data_list)

        sample_data_list = new_sample_data_list

    def _get_next_closest_keyframe(self, current_sample_data: dict, sample_data_list: list):
        timestamp = current_sample_data["timestamp"]

        next_closest_keyframe = sample_data_list[-1]
        sample_data_keyframe_list = [
            sample_data
            for sample_data in sample_data_list
            if sample_data["is_key_frame"]
            and sample_data["timestamp"] > timestamp
            and sample_data["fileformat"] == current_sample_data["fileformat"]
        ]
        for sample_data in sample_data_keyframe_list:
            if sample_data["timestamp"] < next_closest_keyframe["timestamp"]:
                next_closest_keyframe = sample_data
        return next_closest_keyframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", help="Root of the database to fix")
    args = parser.parse_args()

    fixer = KeyFrameFixer()
    dataset_path = Path(args.database_root)
    for item in sorted(dataset_path.iterdir()):
        if not item.is_dir() or not (item / "annotation/sample_data.json").exists():
            continue
        fixer.inspect_and_fix_t4_segment(item)
