from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Union, List, Dict, Any, Optional
from glob import glob
import os.path as osp
import os
import json

from secrets import token_hex
from nuscenes import NuScenes
from scipy import interpolate
from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp


class InterpolationMethod(Enum):
    CUBIC = "CUBIC"
    QUADRATIC = "QUADRATIC"
    NEAREST = "NEAREST"
    LINEAR = "LINEAR"
    LAGRANGE = "LAGRANGE"
    AKIMA = "AKIMA"

    @classmethod
    def from_str(cls, name: str) -> InterpolationMethod:
        name = name.upper()
        assert name in cls.__members__, f"{name} is not in enum members"
        return cls.__members__[name]

    def __eq__(self, __value: Union[str, InterpolationMethod]) -> bool:
        if isinstance(__value, str):
            return self.value == __value.upper()
        else:
            return super().__eq__(__value)

    def get_func(self) -> Callable:
        if self == "cubic":
            return lambda x, y: interpolate.interp1d(x, y, kind="cubic")
        elif self == "quadratic":
            return lambda x, y: interpolate.interp1d(x, y, kind="quadratic")
        elif self == "nearest":
            return lambda x, y: interpolate.interp1d(x, y, kind="nearest")
        elif self == "linear":
            return lambda x, y: interpolate.interp1d(x, y, kind="linear")
        elif self == "lagrange":
            return interpolate.lagrange
        elif self == "akima":
            return interpolate.Akima1DInterpolator


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
        method: Union[str, InterpolationMethod] = "cubic",
        logger: Optional[logging.RootLogger] = None,
    ) -> None:
        super().__init__(input_base, output_base)
        self._dataset_paths = glob(osp.join(input_base, "*"))
        self._method = InterpolationMethod.from_str(method) if isinstance(method, str) else method
        self._interpolation = self._method.get_func()
        self._target_hz = target_hz
        self._interpolate_step = 100.0 / self._target_hz  # [msec]
        self.logger = configure_logger(modname=__name__) if logger is None else logger

    def convert(self) -> None:
        """Interpolate the following annotation files.
        * `sample.json`
        * `sample_data.json`
        * `sample_annotation.json`
        * `ego_pose.json`
        """
        for data_root in self._dataset_paths:
            dataset_id = osp.basename(data_root)
            output_path = osp.join(self._output_base, dataset_id)
            os.makedirs(output_path, exist_ok=True)
            os.system(f"cp -r {data_root} {self._output_base}")

            nusc = NuScenes(version="annotation", dataroot=data_root, verbose=False)

            all_ego_poses = self.get_interpolated_ego_poses(nusc)
            self.logger.info("Finish interpolating ego pose")

            all_samples = self.get_interpolated_samples(nusc)
            self.logger.info("Finish interpolating sample")

            all_sample_data = self.get_interpolated_sample_data(nusc, all_ego_poses, all_samples)
            self.logger.info("Finish interpolating sample data")

            all_sample_annotations = self.get_interpolated_sample_annotations(nusc, all_samples)
            self.logger.info("Finish interpolating sample annotation")

            # save
            annotation_root = osp.join(output_path, nusc.version)
            self._save_json(all_ego_poses, osp.join(annotation_root, "ego_pose.json"))
            self._save_json(all_samples, osp.join(annotation_root, "sample.json"))
            self._save_json(all_sample_data, osp.join(annotation_root, "sample_data.json"))
            self._save_json(
                all_sample_annotations, osp.join(annotation_root, "sample_annotation.json")
            )

    def get_interpolated_ego_poses(self, nusc: NuScenes) -> List[Dict[str, Any]]:
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

        func_x = self._interpolation(timestamps_msec, translations[:, 0])
        func_y = self._interpolation(timestamps_msec, translations[:, 1])
        func_z = self._interpolation(timestamps_msec, translations[:, 2])
        slerp = Slerp(timestamps_msec, rotations)

        num_times = len(timestamps_msec)
        inter_times = np.concatenate(
            [
                np.arange(
                    timestamps_msec[i] + self._interpolate_step,
                    timestamps_msec[i + 1],
                    self._interpolate_step,
                )
                for i in range(num_times - 1)
            ]
        )
        inter_tx = func_x(inter_times)
        inter_ty = func_y(inter_times)
        inter_tz = func_z(inter_times)
        inter_quat = slerp(inter_times).as_quat()

        for time, tx, ty, tz, quat in zip(inter_times, inter_tx, inter_ty, inter_tz, inter_quat):
            inter_ego_info = {
                "token": token_hex(16),
                "translation": [tx, ty, tz],
                "rotation": quat.tolist(),
                "timestamp": int(time * 1e3),
            }
            all_ego_poses.append(inter_ego_info)

        all_ego_poses = sorted(all_ego_poses, key=lambda e: e["timestamp"])

        return all_ego_poses

    def get_interpolated_samples(self, nusc: NuScenes) -> List[Dict[str, Any]]:
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
        all_samples = original_samples.copy()
        for sample in original_samples:
            next_token = sample["next"]
            if next_token == "":
                continue
            next_sample = nusc.get("sample", next_token)
            curr_msec = sample["timestamp"] * 1e-3
            next_msec = next_sample["timestamp"] * 1e-3
            msec_list = np.arange(
                curr_msec + self._interpolate_step,
                next_msec,
                self._interpolate_step,
            )

            inter_token = token_hex(16)
            inter_prev_token: str = sample["token"]
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
            all_samples = sorted(all_samples, key=lambda s: s["timestamp"])
        return all_samples

    def get_interpolated_sample_data(
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
        all_sample_data = sorted(
            [{key: s.get(key) for key in self.SAMPLE_DATA_KEYS} for s in nusc.sample_data],
            key=lambda s: s["timestamp"],
        )

        for sample_data in all_sample_data:
            next_token: str = sample_data["next"]
            if next_token == "":
                continue
            next_sample_data = nusc.get("sample_data", next_token)
            curr_msec = sample_data["timestamp"] * 1e-3
            next_msec = next_sample_data["timestamp"] * 1e-3
            msec_list = np.arange(
                curr_msec + self._interpolate_step,
                next_msec,
                self._interpolate_step,
            )

            inter_token = token_hex(16)
            inter_prev_token: str = sample_data["token"]
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
                        "calibrated_sensor_token": sample_data["calibrated_sensor_token"],
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

            all_sample_data = sorted(all_sample_data, key=lambda s: s["timestamp"])
            return all_sample_data

    def get_interpolated_sample_annotations(
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

        original_sample_timestamps = {s["token"]: s["timestamp"] for s in nusc.sample}
        interpolated_sample_timestamps: List[Dict[str, int]] = []
        for sample in interpolated_samples:
            is_interpolated = sample["timestamp"] not in original_sample_timestamps.values()
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
                    original_sample_timestamps[ann["sample_token"]] * 1e-3
                )

            original_timestamps_msec = np.array(sorted(original_timestamps_msec))
            translations = np.array(
                [t for _, t in sorted(zip(original_timestamps_msec, translations))]
            )
            func_x = self._interpolation(original_timestamps_msec, translations[:, 0])
            func_y = self._interpolation(original_timestamps_msec, translations[:, 1])
            func_z = self._interpolation(original_timestamps_msec, translations[:, 2])

            rotations = Rotation.from_quat(
                np.array([r for _, r in sorted(zip(original_timestamps_msec, rotations))])
            )
            slerp = Slerp(original_timestamps_msec, rotations)

            if all(velocities):
                velocities = [v for _, v in sorted(zip(original_timestamps_msec, velocities))]
                func_vel = self._interpolation(original_timestamps_msec, velocities)
            else:
                func_vel = None

            if all(accelerations):
                accelerations = [
                    a for _, a in sorted(zip(original_timestamps_msec, accelerations))
                ]
                func_acc = self._interpolation(original_timestamps_msec, accelerations)
            else:
                func_acc = None

            for i, curr_msec in enumerate(original_timestamps_msec[:-1]):
                interpolated_timestamps_msec = np.arange(
                    curr_msec + self._interpolate_step,
                    original_timestamps_msec[i + 1],
                    self._interpolate_step,
                )

                inter_x = func_x(interpolated_timestamps_msec)
                inter_y = func_y(interpolated_timestamps_msec)
                inter_z = func_z(interpolated_timestamps_msec)

                inter_quat = slerp(interpolated_timestamps_msec).as_quat()

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
                for j, (timestamp, x, y, z, q, vel, acc) in enumerate(
                    zip(
                        interpolated_timestamps_msec,
                        inter_x,
                        inter_y,
                        inter_z,
                        inter_quat,
                        inter_vel,
                        inter_acc,
                    )
                ):
                    closest_sample = self._get_closest_timestamp(interpolated_samples, timestamp)
                    inter_next_token = (
                        token_hex(16)
                        if j != len(interpolated_sample_timestamps) - 1
                        else sample_anns[i + 1]["next"]
                    )
                    all_sample_annotations.append(
                        {
                            "token": inter_token,
                            "sample_token": closest_sample["token"],
                            "instance_token": ins_token,
                            "attribute_tokens": sample_anns[i]["attribute_tokens"],
                            "visibility_token": sample_anns[i]["visibility_token"],
                            "translation": [x, y, z],
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

    def _get_closest_timestamp(
        self,
        records: List[Dict[str, Any]],
        timestamp: float,
    ) -> Dict[str, Any]:
        """Get the closest element to 'timestamp' from the input list."""
        res = min(records, key=lambda r: abs(r["timestamp"] - timestamp))
        return res

    def _save_json(self, records: Any, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(records, f, indent=4)


def plot_ego_pose(nusc: NuScenes, show: bool) -> None:
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

    plt.plot(translations[:, 0], translations[:, 1])
    plt.grid()

    if show:
        plt.show()
    plt.close()


def plot_sample_annotation(nusc: NuScenes, sample_annotations: list[dict], show: bool) -> None:
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

    plt.plot(translations[:, 0], translations[:, 1])
    diff = np.diff(translations, axis=0)
    arrow_pos = translations[:-1] + 0.5 * diff
    arrow_norm = np.linalg.norm(diff[:, :2], axis=1)
    plt.quiver(
        arrow_pos[:, 0],
        arrow_pos[:, 1],
        diff[:, 0] / arrow_norm,
        diff[:, 1] / arrow_norm,
        angles="xy",
    )
    plt.grid()

    if show:
        plt.show()
    plt.close()


def test_with_plot():
    import argparse
    from nuscenes import NuScenes

    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str, help="root directory path")
    parser.add_argument("--show", action="store_true", help="whether to show plot")
    args = parser.parse_args()

    nusc = NuScenes("annotation", args.data_root, verbose=False)
    plot_ego_pose(nusc, args.show)

    instance_tokens = [record["token"] for record in nusc.instance]
    sample_annotations = {ins_token: [] for ins_token in instance_tokens}
    for ann in nusc.sample_annotation:
        sample_annotations[ann["instance_token"]].append(ann)

    for ins_token, ann in sample_annotations.items():
        plot_sample_annotation(nusc, ann, args.show)


if __name__ == "__main__":
    test_with_plot()
