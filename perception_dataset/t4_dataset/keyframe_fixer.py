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
                # change is_keyframe to False
                sample_data["is_key_frame"] = False

                # change sample_token to ""
                sample_data["sample_token"] = ""

                # remove from sample
                if len(corresponding_sample_list) == 0:
                    continue
                corresponding_sample = corresponding_sample_list[0]
                sample_list.remove(corresponding_sample)

        for sample_data in sample_data_list[:]:
            if sample_data["is_key_frame"]:
                continue
            print(f"{sample_data['filename']}@{sample_data['timestamp']/1e6}")
            sample_data["sample_token"] = self._get_next_closest_keyframe(
                sample_data, sample_data_list
            )["sample_token"]
            if sample_data["sample_token"] == "":
                sample_data_list.remove(sample_data)

        with open(segment_path / "annotation/sample_changed.json", "w") as f:
            json.dump(sample_list, f, indent=4)
        with open(segment_path / "annotation/sample_data_changed.json", "w") as f:
            json.dump(sample_data_list, f, indent=4)

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
        print(f"    {next_closest_keyframe['sample_token']}")
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
