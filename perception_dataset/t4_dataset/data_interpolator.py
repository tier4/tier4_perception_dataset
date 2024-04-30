# cspell: ignore nuscenes, slerp, interp, fileformat, modname, dataroot, nusc, anns, arange, linalg, ncols, nrows

from __future__ import annotations

from functools import partial
from glob import glob
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
from secrets import token_hex
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import numpy as np
from nuscenes import NuScenes
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger
import subprocess


class DataInterpolator(AbstractConverter):
    """A class to interpolate annotations."""

    SAMPLE_KEYS = ("token", "timestamp", "scene_token", "next", "prev")

    SAMPLE_DATA_KEYS = (
        "token",
        "sample_token",
        "ego_pose_token",
        "calibrated_sensor_token",
        "filename",
        "fileformat",
        "width",
        "height",
        "timestamp",
        "is_key_frame",
        "next",
        "prev",
        "is_valid",
    )

    SAMPLE_ANN_KEYS = (
        "token",
        "sample_token",
        "instance_token",
        "attribute_tokens",
        "visibility_token",
        "translation",
        "velocity",
        "acceleration",
        "size",
        "rotation",
        "num_lidar_pts",
        "num_radar_pts",
        "next",
        "prev",
    )

    def __init__(
        self,
        input_base: str,
        output_base: str,
        copy_excludes: Optional[List[str]] = None,
        logger: Optional[logging.RootLogger] = None,
    ) -> None:
        super().__init__(input_base, output_base)
        self._dataset_paths = glob(osp.join(input_base, "*"))
        self.copy_excludes = copy_excludes
        self.logger = configure_logger(modname=__name__) if logger is None else logger

    def convert(self) -> None:
        """Interpolate the following annotation files.
        * `sample.json`
        * `sample_annotation.json`
        * `instance.json`
        * `scene.json`
        """
        func = partial(self._convert_single)
        with mp.Pool(mp.cpu_count()) as p:
            p.map(func, self._dataset_paths)
        # for data_root in self._dataset_paths:
        #     self._convert_single(data_root)

    def _convert_single(self, data_root: str) -> None:
        self.logger.info(f"Start processing: {data_root}")

        dataset_id = osp.basename(data_root)
        output_path = osp.join(self._output_base, dataset_id)
        os.makedirs(output_path, exist_ok=True)

        command: list[str] = ["rsync", "-av"]
        if self.copy_excludes is not None:
            assert isinstance(self.copy_excludes, list), f"Unexpected type of excludes: {type(self.copy_excludes)}"
            for e in self.copy_excludes:
                command.extend(["--exclude", e])
        command.extend([f"{data_root}", f"{self._output_base}"])
        subprocess.run(command)

        nusc = NuScenes(version="annotation", dataroot=data_root, verbose=False)

        all_samples, all_sample_data = self.interpolate_sample(nusc)
        self.logger.info("Finish interpolating sample and sample data")

        all_sample_anns = self.interpolate_sample_annotation(nusc, all_samples)
        self.logger.info("Finish interpolating sample annotation")

        all_instances = self.update_instance_record(nusc, all_sample_anns)
        self.logger.info("Finish updating instance")

        all_scenes = self.update_scene_record(nusc, all_samples)
        self.logger.info("Finish updating scene")

        # save
        annotation_root = osp.join(output_path, nusc.version)
        self._save_json(all_samples, osp.join(annotation_root, "sample.json"))
        self._save_json(all_sample_data, osp.join(annotation_root, "sample_data.json"))
        self._save_json(all_sample_anns, osp.join(annotation_root, "sample_annotation.json"))
        self._save_json(all_instances, osp.join(annotation_root, "instance.json"))
        self._save_json(all_scenes, osp.join(annotation_root, "scene.json"))

    def interpolate_sample(
        self, nusc: NuScenes
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Interpolate sample with non-key frame sample data.

        Args:
            nusc (NuScenes): NuScenes instance.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
                All sample records containing interpolated ones,
                and all sample data that `is_key_frame` is converted to `True`.
        """
        # interpolate sample record based on lidar timestamp
        scene_token: str = nusc.sample[0]["scene_token"]
        interpolated_samples_dict: Dict[str, Dict[str, Any]] = {}
        for sd_record in nusc.sample_data:
            if sd_record["is_key_frame"] or sd_record["sensor_modality"] != "lidar":
                continue

            # [filename, pcd, bin]
            filename: str = osp.basename(sd_record["filename"]).split(".")[0]
            interpolated_samples_dict[filename] = {
                "token": token_hex(16),
                "timestamp": sd_record["timestamp"],
                "scene_token": scene_token,
                "prev": None,
                "next": None,
            }

        # update prev/next token in sample record
        all_sorted_samples = sorted(
            [{key: s[key] for key in self.SAMPLE_KEYS} for s in nusc.sample]
            + list(interpolated_samples_dict.values()),
            key=lambda s: s["timestamp"],
        )
        prev_token: str = ""
        for i, sample_record in enumerate(all_sorted_samples[:-1]):
            next_sample_token = all_sorted_samples[i + 1]["token"]
            sample_record["prev"] = prev_token
            sample_record["next"] = next_sample_token
            prev_token = sample_record["token"]
        all_sorted_samples[-1]["prev"] = prev_token
        all_sorted_samples[-1]["next"] = ""

        # update sample_token in sample_data
        all_sample_data = [
            {name: record[name] for name in self.SAMPLE_DATA_KEYS if name in record.keys()}
            for record in nusc.sample_data
        ]
        for sd_record in all_sample_data:
            if sd_record["is_key_frame"]:
                continue
            filename: str = osp.splitext(osp.basename(sd_record["filename"].replace(".pcd.bin", ".pcd")))[0]
            if filename in interpolated_samples_dict.keys():
                sample_token: str = interpolated_samples_dict[filename]["token"]
                sd_record["sample_token"] = sample_token
                sd_record["is_key_frame"] = True

        return all_sorted_samples, all_sample_data

    def interpolate_sample_annotation(
        self,
        nusc: NuScenes,
        all_samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Interpolate sample annotation records.

        Args:
            nusc (NuScenes): NuScenes instance.
            all_samples (List[Dict[str, Any]]): All sample records containing interpolated ones.

        Returns:
            List[Dict[str, Any]]: All sample annotation records containing interpolated ones.
        """
        # extract interpolated sample timestamps
        original_timestamps_info = {s["token"]: s["timestamp"] for s in nusc.sample}
        interpolated_timestamps: List[int] = []
        for sample in all_samples:
            # if sample timestamp is not contained in original timestamps, it is interpolated
            is_interpolated = sample["timestamp"] not in original_timestamps_info.values()
            if is_interpolated:
                interpolated_timestamps.append(sample["timestamp"])

        # separate original sample annotations by instance token
        all_instance_anns = {ins["token"]: [] for ins in nusc.instance}
        for ann in nusc.sample_annotation:
            tmp_ann = {key: ann.get(key, None) for key in self.SAMPLE_ANN_KEYS}
            all_instance_anns[ann["instance_token"]].append(tmp_ann)

        for ins_token, sample_anns in all_instance_anns.items():
            translations = []
            rotations = []
            velocities = []
            accelerations = []
            original_timestamps = []
            for ann in sample_anns:
                translations.append(ann["translation"])
                rotations.append(ann["rotation"])
                velocities.append(ann["velocity"])
                accelerations.append(ann["acceleration"])
                original_timestamps.append(original_timestamps_info[ann["sample_token"]])

            # skip if there is only one annotation because interpolation is unavailable
            if len(original_timestamps) < 2:
                continue

            sample_anns.sort(key=lambda s: original_timestamps_info[s["sample_token"]])

            original_timestamps = np.array(sorted(original_timestamps))
            translations = np.array([t for _, t in sorted(zip(original_timestamps, translations))])
            func_xyz = CubicSpline(original_timestamps, translations)

            rotations = Rotation.from_quat(
                np.array([r for _, r in sorted(zip(original_timestamps, rotations))])
            )
            func_rot = Slerp(original_timestamps, rotations)

            if all(velocities):
                velocities = [v for _, v in sorted(zip(original_timestamps, velocities))]
                func_vel = CubicSpline(original_timestamps, velocities)
            else:
                func_vel = None

            if all(accelerations):
                accelerations = [a for _, a in sorted(zip(original_timestamps, accelerations))]
                func_acc = CubicSpline(original_timestamps, accelerations)
            else:
                func_acc = None

            new_sample_anns = []
            for i, curr_time in enumerate(original_timestamps[:-1]):
                next_time: int = original_timestamps[i + 1]
                curr_interp_times = [
                    t for t in interpolated_timestamps if curr_time <= t <= next_time
                ]

                num_interpolation: int = len(curr_interp_times)

                # skip if there is no interpolated timestamp
                if num_interpolation == 0:
                    continue

                interp_xyz = func_xyz(curr_interp_times)
                interp_quat = func_rot(curr_interp_times).as_quat()

                interp_vel = (
                    func_vel(curr_interp_times)
                    if func_vel is not None
                    else [None] * num_interpolation
                )

                interp_acc = (
                    func_acc(curr_interp_times)
                    if func_acc is not None
                    else [None] * num_interpolation
                )

                # update next token in current sample annotation to the first interpolated token
                # note that, keep original next token to set to next token for the last interpolated sample annotation
                inter_token = token_hex(16)
                original_next_token = sample_anns[i]["next"]
                sample_anns[i]["next"] = inter_token
                inter_prev_token = sample_anns[i]["token"]
                for j, (timestamp, xyz, q, vel, acc) in enumerate(
                    zip(
                        curr_interp_times,
                        interp_xyz,
                        interp_quat,
                        interp_vel,
                        interp_acc,
                    )
                ):
                    closest_sample = self._get_closest_timestamp(all_samples, timestamp)
                    # other than the last interpolation, generate a new next token
                    # at last, set original next token
                    inter_next_token = (
                        token_hex(16) if j != num_interpolation - 1 else original_next_token
                    )
                    new_sample_anns.append(
                        {
                            "token": inter_token,
                            "sample_token": closest_sample["token"],
                            "instance_token": ins_token,
                            "attribute_tokens": sample_anns[i]["attribute_tokens"],
                            "visibility_token": sample_anns[i]["visibility_token"],
                            "translation": xyz.tolist(),
                            "velocity": vel,
                            "acceleration": acc,
                            "size": sample_anns[i]["size"],
                            "rotation": q.tolist(),
                            "num_lidar_pts": sample_anns[i]["num_lidar_pts"],
                            "num_radar_pts": sample_anns[i]["num_radar_pts"],
                            "next": inter_next_token,
                            "prev": inter_prev_token,
                        }
                    )
                    # set the latest prev token as current token, and
                    # next token as current token
                    inter_prev_token = inter_token
                    inter_token = inter_next_token
                sample_anns[i + 1]["prev"] = inter_prev_token
            # extend original sample annotations with interpolated
            sample_anns += new_sample_anns

        return [ann for ann_list in all_instance_anns.values() for ann in ann_list]

    @staticmethod
    def update_instance_record(
        nusc: NuScenes,
        all_sample_annotations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        all_instances = nusc.instance.copy()
        for instance_record in all_instances:
            token = instance_record["token"]
            num_anns = len(
                [ann for ann in all_sample_annotations if ann["instance_token"] == token]
            )
            instance_record["nbr_annotations"] = num_anns
        return all_instances

    @staticmethod
    def update_scene_record(
        nusc: NuScenes,
        all_samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        all_scenes = nusc.scene.copy()
        all_scenes[0]["nbr_samples"] = len(all_samples)
        description = all_scenes[0]["description"]
        if len(description) > 0:
            description += ", interpolate"
        else:
            description = "interpolate"
        all_scenes[0]["description"] = description
        return all_scenes

    def _get_closest_timestamp(
        self,
        records: List[Dict[str, Any]],
        timestamp: int,
    ) -> Dict[str, Any]:
        """Get the closest element to 'timestamp' from the input list."""
        assert isinstance(
            timestamp, int
        ), f"Expected integer timestamp, but got: {type(timestamp)}"
        res = min(records, key=lambda r: abs(r["timestamp"] - timestamp))
        return res

    def _save_json(self, records: Any, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(records, f, indent=4)


# ============================== DEBUG ==============================
def plot_sample_annotation(
    ax: Axes,
    nusc: NuScenes,
    sample_annotations: list[dict],
    instance_token: str,
) -> Axes:
    sample_timestamps = {s["token"]: s["timestamp"] for s in nusc.sample}
    sample_annotations = sorted(
        sample_annotations, key=lambda s: sample_timestamps[s["sample_token"]]
    )

    timestamps_msec = []
    translations = []
    rotations = []
    for ann in sample_annotations:
        timestamps_msec.append(sample_timestamps[ann["sample_token"]] * 1e-3)
        translations.append(ann["translation"])
        rotations.append(ann["rotation"])

    timestamps_msec = sorted(timestamps_msec)
    translations = [t for _, t in sorted(zip(timestamps_msec, translations))]
    rotations = [r for _, r in sorted(zip(timestamps_msec, rotations))]

    timestamps_msec = np.array(timestamps_msec) - timestamps_msec[0]
    translations = np.array(translations)

    ax.plot(translations[:, 0], translations[:, 1])

    if len(timestamps_msec) > 1:
        mean_hz: float = np.diff(timestamps_msec).mean() * 0.1
        print(f"[{instance_token}]: Mean Hz={mean_hz}")

        diff = np.diff(translations, axis=0)
        arrow_pos = translations[:-1] + 0.5 * diff
        arrow_norm = np.linalg.norm(diff[:, :2], axis=1)
        ax.quiver(
            arrow_pos[:, 0],
            arrow_pos[:, 1],
            diff[:, 0] / (arrow_norm + 1e-6),
            diff[:, 1] / (arrow_norm + 1e-6),
            angles="xy",
        )
    else:
        print(f"[{instance_token}]: Single annotation only!!")

    return ax


def test_with_plot():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot interpolated paths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_root", type=str, help="Path to interpolated output base root")
    parser.add_argument("-o", "--output", type=str, help="Output path, skip if not specified")
    parser.add_argument("-n", "--num_max_plot", type=int, help="Number of instances to plot")
    parser.add_argument("--show", action="store_true", help="")
    args = parser.parse_args()

    data_paths = glob(osp.join(args.data_root, "*"))
    save_dir: str | None = args.output
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for data_root in data_paths:
        try:
            print(f"Start plotting >> {data_root}")
            nusc = NuScenes("annotation", data_root, verbose=False)

            sample_annotations: Dict[str, List[Any]] = {}
            instances: List[Dict[str, Any]] = (
                nusc.instance if args.num_max_plot is None else nusc.instance[: args.num_max_plot]
            )
            for record in instances:
                token: str = record["token"]
                next_ann_token: str = record["first_annotation_token"]
                prev_ann_token: str = ""
                last_ann_token: str = record["last_annotation_token"]
                sample_annotations[token] = []
                while next_ann_token != "":
                    sample_ann = nusc.get("sample_annotation", next_ann_token)
                    sample_annotations[token].append(sample_ann)
                    assert (
                        prev_ann_token == sample_ann["prev"]
                    ), f"Invalid prev token>> Expect: {prev_ann_token}, Result: {sample_ann['prev']}"
                    prev_ann_token = next_ann_token
                    next_ann_token = sample_ann["next"]
                assert (
                    sample_annotations[token][-1]["token"] == last_ann_token
                ), f"Invalid last ann token>>: Expect: {last_ann_token}, Result: {sample_annotations[token][-1]['token']}"

            num_instances: int = len(sample_annotations.keys())
            num_cols = 5 if num_instances > 5 else num_instances
            num_rows = (
                num_instances // num_cols
                if num_instances % num_cols == 0
                else num_instances // num_cols + 1
            )
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols)

            for i, (ins_token, ann) in enumerate(sample_annotations.items()):
                ax: Axes = (
                    axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i % num_cols]
                )
                ax = plot_sample_annotation(ax, nusc, ann, ins_token)
                instance_record = nusc.get("instance", ins_token)
                category_record = nusc.get("category", instance_record["category_token"])
                ax.set_title(category_record["name"])

            if save_dir is not None:
                scenario_name: str = osp.basename(data_root)
                fig.savefig(osp.join(save_dir, f"{scenario_name}.png"))

            if args.show:
                plt.show()

            plt.close()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    test_with_plot()
