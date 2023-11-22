# cspell: ignore nuscenes, akima, pchip, slerp, interp, fileformat, modname, dataroot, nusc, anns, arange, linalg

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from glob import glob
import os.path as osp
import os
import json
import multiprocessing as mp
from functools import partial

from secrets import token_hex
from nuscenes import NuScenes
from scipy.interpolate import CubicSpline
from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from scipy.spatial.transform import Rotation, Slerp


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
        * `sample_data.json`
        * `sample_annotation.json`
        * `ego_pose.json`
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

        all_ego_poses = self.interpolate_ego_pose(nusc)
        self.logger.info("Finish interpolating ego pose")

        all_samples = self.interpolate_sample(nusc)
        self.logger.info("Finish interpolating sample")

        all_sample_data = self.interpolate_sample_data(nusc, all_ego_poses, all_samples)
        self.logger.info("Finish interpolating sample data")

        all_sample_anns = self.interpolate_sample_annotation(nusc, all_samples)
        self.logger.info("Finish interpolating sample annotation")

        all_instances = self.update_instance_record(nusc, all_sample_anns)
        self.logger.info("Finish updating instance")

        all_scenes = self.update_scene_record(nusc, all_samples)
        self.logger.info("Finish updating scene")

        # save
        annotation_root = osp.join(output_path, nusc.version)
        self._save_json(all_ego_poses, osp.join(annotation_root, "ego_pose.json"))
        self._save_json(all_samples, osp.join(annotation_root, "sample.json"))
        self._save_json(all_sample_data, osp.join(annotation_root, "sample_data.json"))
        self._save_json(all_sample_anns, osp.join(annotation_root, "sample_annotation.json"))
        self._save_json(all_instances, osp.join(annotation_root, "instance.json"))
        self._save_json(all_scenes, osp.join(annotation_root, "scene.json"))

    def interpolate_ego_pose(self, nusc: NuScenes) -> List[Dict[str, Any]]:
        """
        Extend ego pose records with interpolation.

        The keys of ego pose are as follows.
        * token (str)
        * translation (list[float])
        * rotation (list[float])
        * timestamp (int)

        Args:
            nusc (NuScenes)

        Returns:
            List[Dict[str, Any]]
        """
        timestamps_msec = []
        translations = []
        rotations = []
        all_ego_poses: List[Dict[str, Any]] = sorted(nusc.ego_pose, key=lambda e: e["timestamp"])
        for ego_pose in all_ego_poses:
            timestamps_msec.append(ego_pose["timestamp"] * 1e-3)
            translations.append(ego_pose["translation"])
            rotations.append(ego_pose["rotation"])
        translations = np.array([t for _, t in sorted(zip(timestamps_msec, translations))])
        rotations = np.array([r for _, r in sorted(zip(timestamps_msec, rotations))])

        timestamps_msec, unique_idx = np.unique(timestamps_msec, return_index=True)
        translations = translations[unique_idx]
        rotations = rotations[unique_idx]
        rotations = Rotation.from_quat(rotations)

        func_xyz = CubicSpline(timestamps_msec, translations)
        func_rot = Slerp(timestamps_msec, rotations)

        num_times = len(timestamps_msec)
        inter_times = np.concatenate(
            [
                np.arange(
                    timestamps_msec[i] + self._interpolate_step_msec,
                    timestamps_msec[i + 1],
                    self._interpolate_step_msec,
                )
                for i in range(num_times - 1)
            ]
        )
        inter_xyz = func_xyz(inter_times)
        inter_quat = func_rot(inter_times).as_quat()

        for time, xyz, quat in zip(inter_times, inter_xyz, inter_quat):
            inter_ego_info = {
                "token": token_hex(16),
                "translation": xyz.tolist(),
                "rotation": quat.tolist(),
                "timestamp": int(time * 1e3),
            }
            all_ego_poses.append(inter_ego_info)

        all_ego_poses = sorted(all_ego_poses, key=lambda e: e["timestamp"])

        return all_ego_poses

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

    def interpolate_sample_data(
        self,
        nusc: NuScenes,
        interpolated_ego_poses: List[Dict[str, Any]],
        interpolated_samples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extend sample data records with interpolation.

        The keys of sample data are as follows.
        * token (str)
        * sample_token (str)
        * ego_pose_token (str)
        * calibrated_sensor_token (str)
        * filename (str)
        * fileformat (str)
        * width (int)
        * height (int)
        * timestamp (int)
        * is_key_frame (bool)
        * next (str)
        * prev (str)
        * is_valid (bool)
        """
        original_sample_data = sorted(
            [{key: s.get(key) for key in self.SAMPLE_DATA_KEYS} for s in nusc.sample_data],
            key=lambda s: s["timestamp"],
        )

        prev_tokens = {cs["token"]: "" for cs in nusc.calibrated_sensor}
        all_sample_data = []
        for sample_data in original_sample_data:
            cs_token: str = sample_data["calibrated_sensor_token"]
            sample_data["prev"] = prev_tokens[cs_token]
            next_token: str = sample_data["next"]
            if next_token == "":
                all_sample_data.append(sample_data)
                continue
            next_sample_data = nusc.get("sample_data", next_token)
            curr_msec = sample_data["timestamp"] * 1e-3
            next_msec = next_sample_data["timestamp"] * 1e-3
            msec_list = np.arange(
                curr_msec + self._interpolate_step_msec,
                next_msec,
                self._interpolate_step_msec,
            )

            inter_token = token_hex(16)
            inter_prev_token: str = sample_data["token"]
            if len(msec_list) != 0:
                sample_data["next"] = inter_token
            for i, msec in enumerate(msec_list):
                inter_next_token = token_hex(16) if i != len(msec_list) - 1 else next_token
                inter_sample_token = self._get_closest_timestamp(
                    interpolated_samples, int(msec * 1e3)
                )["token"]
                inter_ego_token = self._get_closest_timestamp(
                    interpolated_ego_poses, int(msec * 1e3)
                )["token"]
                all_sample_data.append(
                    {
                        "token": inter_token,
                        "sample_token": inter_sample_token,
                        "ego_pose_token": inter_ego_token,
                        "calibrated_sensor_token": cs_token,
                        "filename": "",
                        "fileformat": sample_data["fileformat"],
                        "width": sample_data["width"],
                        "height": sample_data["height"],
                        "timestamp": int(msec * 1e3),
                        "is_key_frame": True,
                        "next": inter_next_token,
                        "prev": inter_prev_token,
                        "is_valid": False,
                    }
                )
                inter_prev_token = inter_token
                inter_token = inter_next_token
            prev_tokens[cs_token] = inter_prev_token
            all_sample_data.append(sample_data)

        all_sample_data = sorted(all_sample_data, key=lambda s: s["timestamp"])
        return all_sample_data

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
        all_sample_annotations = [
            {key: s.get(key) for key in self.SAMPLE_ANN_KEYS} for s in nusc.sample_annotation
        ]

        all_original_sample_timestamps = {s["token"]: s["timestamp"] for s in nusc.sample}
        interpolated_sample_timestamps: List[Dict[str, int]] = []
        for sample in interpolated_samples:
            is_interpolated = sample["timestamp"] not in all_original_sample_timestamps.values()
            if is_interpolated:
                interpolated_sample_timestamps.append({sample["token"]: sample["timestamp"]})

        all_instance_anns = {ins["token"]: [] for ins in nusc.instance}
        for ann in all_sample_annotations:
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

            for i, curr_msec in enumerate(original_timestamps_msec[:-1]):
                interpolated_timestamps_msec = np.arange(
                    curr_msec + self._interpolate_step_msec,
                    original_timestamps_msec[i + 1],
                    self._interpolate_step_msec,
                )

                inter_xyz = func_xyz(interpolated_timestamps_msec)
                inter_quat = func_rot(interpolated_timestamps_msec).as_quat()

                if func_vel is not None:
                    inter_vel = func_vel(interpolated_timestamps_msec)
                else:
                    inter_vel = [None] * len(interpolated_timestamps_msec)

                if func_acc is not None:
                    inter_acc = func_acc(interpolated_timestamps_msec)
                else:
                    inter_acc = [None] * len(interpolated_timestamps_msec)

                inter_prev_token = sample_anns[i]["token"]
                inter_token = token_hex(16)
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
                    inter_next_token = (
                        token_hex(16)
                        if j != len(interpolated_sample_timestamps) - 1
                        else sample_anns[i]["next"]
                    )
                    all_sample_annotations.append(
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
                    inter_prev_token = inter_token
                    inter_token = inter_next_token
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
        res = min(records, key=lambda r: abs(r["timestamp"] - timestamp))
        return res

    def _save_json(self, records: Any, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(records, f, indent=4)


# ============================== DEBUG ==============================
def plot_ego_pose(ax: Axes, nusc: NuScenes) -> Axes:
    timestamps_msec = []
    translations = []
    rotations = []
    for record in nusc.ego_pose:
        timestamps_msec.append(record["timestamp"] * 1e-3)
        translations.append(record["translation"])
        rotations.append(record["rotation"])

    timestamps_msec = sorted(timestamps_msec)
    translations = [t for _, t in sorted(zip(timestamps_msec, translations))]
    rotations = [r for _, r in sorted(zip(timestamps_msec, rotations))]

    timestamps_msec = np.array(timestamps_msec) - timestamps_msec[0]
    translations = np.array(translations)
    rotations = Rotation(rotations)

    ax.plot(translations[:, 0], translations[:, 1])
    diff = np.diff(translations, axis=0)
    arrow_pos = translations[:-1] + 0.5 * diff
    arrow_norm = np.linalg.norm(diff[:, :2], axis=1)
    ax.quiver(
        arrow_pos[:, 0],
        arrow_pos[:, 1],
        diff[:, 0] / arrow_norm,
        diff[:, 1] / arrow_norm,
        angles="xy",
    )
    return ax


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
        diff[:, 0] / arrow_norm,
        diff[:, 1] / arrow_norm,
        angles="xy",
    )
    return ax


def test_with_plot():
    import argparse
    from nuscenes import NuScenes

    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str, help="root directory path")
    args = parser.parse_args()

    nusc = NuScenes("annotation", args.data_root, verbose=False)
    ego_ax = plt.subplot()
    plot_ego_pose(ego_ax, nusc)
    ego_ax.set_title("Ego")

    instance_tokens = [record["token"] for record in nusc.instance]
    sample_annotations = {ins_token: [] for ins_token in instance_tokens}
    for ann in nusc.sample_annotation:
        sample_annotations[ann["instance_token"]].append(ann)

    num_instance: int = len(instance_tokens)
    num_cols = 5
    _, axes = plt.subplots(nrows=num_instance // num_cols + 1, ncols=num_cols)

    for i, (ins_token, ann) in enumerate(sample_annotations.items()):
        ax: Axes = axes[i // num_cols, i % num_cols]
        ax = plot_sample_annotation(ax, nusc, ann)
        instance_record = nusc.get("instance", ins_token)
        category_record = nusc.get("category", instance_record["category_token"])
        ax.set_title(category_record["name"])

    plt.show()
    plt.close()


if __name__ == "__main__":
    test_with_plot()
