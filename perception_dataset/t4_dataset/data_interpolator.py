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
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import numpy as np
from nuscenes import NuScenes
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger


class DataInterpolator(AbstractConverter):
    """A class to interpolate annotations."""

    EGO_POSE_KEYS = ("token", "translation", "rotation", "timestamp")
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
        target_hz: float = 10.0,
        logger: Optional[logging.RootLogger] = None,
    ) -> None:
        super().__init__(input_base, output_base)
        self._dataset_paths = glob(osp.join(input_base, "*"))
        self._target_hz = target_hz
        self._interpolate_step_msec = 1000.0 / self._target_hz  # [msec]
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

    def _convert_single(self, data_root: str) -> None:
        self.logger.info(f"Start processing: {data_root}")

        dataset_id = osp.basename(data_root)
        output_path = osp.join(self._output_base, dataset_id)
        os.makedirs(output_path, exist_ok=True)
        os.system(f"cp -r {data_root} {self._output_base}")

        nusc = NuScenes(version="annotation", dataroot=data_root, verbose=False)

        all_samples = self.interpolate_sample(nusc)
        self.logger.info("Finish interpolating sample")

        all_sample_anns = self.interpolate_sample_annotation(nusc, all_samples)
        self.logger.info("Finish interpolating sample annotation")

        all_instances = self.update_instance_record(nusc, all_sample_anns)
        self.logger.info("Finish updating instance")

        all_scenes = self.update_scene_record(nusc, all_samples)
        self.logger.info("Finish updating scene")

        # save
        annotation_root = osp.join(output_path, nusc.version)
        self._save_json(all_samples, osp.join(annotation_root, "sample.json"))
        self._save_json(all_sample_anns, osp.join(annotation_root, "sample_annotation.json"))
        self._save_json(all_instances, osp.join(annotation_root, "instance.json"))
        self._save_json(all_scenes, osp.join(annotation_root, "scene.json"))

    def interpolate_sample(self, nusc: NuScenes) -> List[Dict[str, Any]]:
        """
        Extend sample records with interpolation.

        The keys of sample are as follows.
        * token (str)
        * timestamp (int)
        * scene_token (str)
        * next (str)
        * prev (str)

        Args:
            nusc (NuScenes): _description_

        Returns:
            List[Dict[str, Any]]: _description_
        """
        original_samples: List[Dict[str, Any]] = sorted(
            [{name: record[name] for name in self.SAMPLE_KEYS} for record in nusc.sample],
            key=lambda s: s["timestamp"],
        )
        prev_token: str = ""
        all_samples: List[Dict[str, Any]] = []
        for sample in original_samples:
            sample["prev"] = prev_token
            next_token = sample["next"]
            if next_token == "":
                all_samples.append(sample)
                continue
            next_sample = nusc.get("sample", next_token)
            curr_msec = sample["timestamp"] * 1e-3
            next_msec = next_sample["timestamp"] * 1e-3
            msec_list = np.arange(
                curr_msec + self._interpolate_step_msec,
                next_msec,
                self._interpolate_step_msec,
            )

            inter_token = token_hex(16)
            inter_prev_token: str = sample["token"]
            if len(msec_list) != 0:
                sample["next"] = inter_token
            for i, msec in enumerate(msec_list):
                inter_next_token = token_hex(16) if i != len(msec_list) - 1 else next_token
                all_samples.append(
                    {
                        "token": inter_token,
                        "timestamp": int(msec * 1e3),
                        "scene_token": sample["scene_token"],
                        "next": inter_next_token,
                        "prev": inter_prev_token,
                    }
                )
                inter_prev_token = inter_token
                inter_token = inter_next_token
            prev_token = inter_prev_token
            all_samples.append(sample)
        all_samples = sorted(all_samples, key=lambda s: s["timestamp"])
        return all_samples

    def interpolate_sample_annotation(
        self,
        nusc: NuScenes,
        interpolated_samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extend sample annotation records with interpolation.

        The keys of sample annotation are as follows.
        * token (str)
        * sample_token (str)
        * instance_token (str)
        * attribute_tokens (list[str])
        * visibility_token (str)
        * translation (list[float])
        * velocity (list[float])
        * acceleration (list[float])
        * size (list[float])
        * rotation (list[float])
        * num_lidar_pts (int)
        * num_radar_pts (int)
        * next (str)
        * prev (str)
        """
        all_original_sample_annotations = [
            {key: s.get(key) for key in self.SAMPLE_ANN_KEYS} for s in nusc.sample_annotation
        ]

        all_original_sample_timestamps = {s["token"]: s["timestamp"] for s in nusc.sample}
        interpolated_sample_timestamps: List[Dict[str, int]] = []
        for sample in interpolated_samples:
            is_interpolated = sample["timestamp"] not in all_original_sample_timestamps.values()
            if is_interpolated:
                interpolated_sample_timestamps.append({sample["token"]: sample["timestamp"]})

        all_instance_anns = {ins["token"]: [] for ins in nusc.instance}
        for ann in all_original_sample_annotations:
            all_instance_anns[ann["instance_token"]].append(ann)

        for ins_token, sample_anns in all_instance_anns.items():
            translations = []
            rotations = []
            velocities = []
            accelerations = []
            original_timestamps_msec = []
            for ann in sample_anns:
                translations.append(ann["translation"])
                rotations.append(ann["rotation"])
                velocities.append(ann["velocity"])
                accelerations.append(ann["acceleration"])
                original_timestamps_msec.append(
                    all_original_sample_timestamps[ann["sample_token"]] * 1e-3
                )

            # skip if there is only one annotation because interpolation is unavailable
            if len(original_timestamps_msec) < 2:
                continue

            sample_anns.sort(key=lambda s: all_original_sample_timestamps[s["sample_token"]])

            original_timestamps_msec = np.array(sorted(original_timestamps_msec))
            translations = np.array(
                [t for _, t in sorted(zip(original_timestamps_msec, translations))]
            )
            func_xyz = CubicSpline(original_timestamps_msec, translations)

            rotations = Rotation.from_quat(
                np.array([r for _, r in sorted(zip(original_timestamps_msec, rotations))])
            )
            func_rot = Slerp(original_timestamps_msec, rotations)

            if all(velocities):
                velocities = [v for _, v in sorted(zip(original_timestamps_msec, velocities))]
                func_vel = CubicSpline(original_timestamps_msec, velocities)
            else:
                func_vel = None

            if all(accelerations):
                accelerations = [
                    a for _, a in sorted(zip(original_timestamps_msec, accelerations))
                ]
                func_acc = CubicSpline(original_timestamps_msec, accelerations)
            else:
                func_acc = None

            new_sample_anns = []
            for i, curr_msec in enumerate(original_timestamps_msec[:-1]):
                interpolated_timestamps_msec = np.arange(
                    curr_msec + self._interpolate_step_msec,
                    original_timestamps_msec[i + 1],
                    self._interpolate_step_msec,
                )

                num_interpolation: int = len(interpolated_timestamps_msec)

                # skip if there is no interpolated timestamp
                if num_interpolation == 0:
                    continue

                inter_xyz = func_xyz(interpolated_timestamps_msec)
                inter_quat = func_rot(interpolated_timestamps_msec).as_quat()

                inter_vel = (
                    func_vel(interpolated_timestamps_msec)
                    if func_vel is not None
                    else [None] * num_interpolation
                )

                inter_acc = (
                    func_acc(interpolated_timestamps_msec)
                    if func_acc is not None
                    else [None] * num_interpolation
                )

                # update next token in current sample annotation to the first interpolated token
                # note that, keep original next token to set to next token for the last interpolated sample annotation
                inter_token = token_hex(16)
                original_next_token = sample_anns[i]["next"]
                sample_anns[i]["next"] = inter_token
                inter_prev_token = sample_anns[i]["token"]
                for j, (timestamp_msec, xyz, q, vel, acc) in enumerate(
                    zip(
                        interpolated_timestamps_msec,
                        inter_xyz,
                        inter_quat,
                        inter_vel,
                        inter_acc,
                    )
                ):
                    closest_sample = self._get_closest_timestamp(
                        interpolated_samples,
                        int(timestamp_msec * 1e3),
                    )
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
                    # next token as token
                    inter_prev_token = inter_token
                    inter_token = inter_next_token
                sample_anns[i + 1]["prev"] = inter_prev_token
            # extend original sample annotations with interpolated
            sample_anns += new_sample_anns

        all_sample_annotations = [
            ann for _, sample_anns in all_instance_anns.items() for ann in sample_anns
        ]

        return all_sample_annotations

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
def plot_sample_annotation(ax: Axes, nusc: NuScenes, sample_annotations: list[dict]) -> Axes:
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
    return ax


def test_with_plot():
    import argparse

    from nuscenes import NuScenes

    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str, help="Path to interpolated output base root")
    args = parser.parse_args()

    data_paths = glob(osp.join(args.data_root, "*"))
    for data_root in data_paths:
        try:
            print(f"Start plotting >> {data_root}")
            nusc = NuScenes("annotation", data_root, verbose=False)

            sample_annotations: Dict[str, List[Any]] = {}
            for record in nusc.instance:
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

            num_instance: int = len(sample_annotations.keys())
            num_cols = 5
            _, axes = plt.subplots(nrows=num_instance // num_cols + 1, ncols=num_cols)

            for i, (ins_token, ann) in enumerate(sample_annotations.items()):
                ax: Axes = axes[i // num_cols, i % num_cols]
                ax = plot_sample_annotation(ax, nusc, ann)
                instance_record = nusc.get("instance", ins_token)
                category_record = nusc.get("category", instance_record["category_token"])
                ax.set_title(category_record["name"])

            plt.tight_layout()
            plt.show()
            plt.close()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    test_with_plot()
