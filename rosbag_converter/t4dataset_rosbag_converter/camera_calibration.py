from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraCalibration:
    topic: str
    channel: str
    camera_name: str
    frame_id: str
    width: int
    height: int
    k: np.ndarray
    d: np.ndarray
    r: np.ndarray
    p: np.ndarray
    distortion_model: str

    @property
    def link_frame_id(self) -> str:
        suffix = "/camera_optical_link"
        if self.frame_id.endswith(suffix):
            return self.frame_id[: -len(suffix)] + "/camera_link"
        return self.frame_id

    @property
    def raw_intrinsic(self) -> list[list[float]]:
        return self.k.reshape(3, 3).tolist()

    @property
    def rectified_intrinsic(self) -> list[list[float]]:
        return np.delete(self.p.reshape(3, 4), 3, 1).tolist()

    @property
    def distortion(self) -> list[float]:
        return self.d.tolist()


def load_camera_calibrations(
    vehicle_root: Path,
    cameras: list[dict[str, Any]],
) -> dict[str, CameraCalibration]:
    tier4_c2 = vehicle_root / "tier4-c2"
    if not tier4_c2.exists():
        raise FileNotFoundError(f"Missing camera calibration directory: {tier4_c2}")

    calibrations = {}
    for camera in cameras:
        topic = camera["topic"]
        camera_name = _camera_name_from_topic(topic)
        info = _load_yaml(tier4_c2 / f"{camera_name}_info.yaml")
        frame_id = str(camera.get("frame_id") or _load_camera_frame_id(tier4_c2, camera_name))
        calibrations[topic] = CameraCalibration(
            topic=topic,
            channel=camera["channel"],
            camera_name=camera_name,
            frame_id=frame_id,
            width=int(info["image_width"]),
            height=int(info["image_height"]),
            k=np.asarray(info["camera_matrix"]["data"], dtype=np.float64),
            d=np.asarray(info["distortion_coefficients"]["data"], dtype=np.float64),
            r=np.asarray(
                info.get("rectification_matrix", {}).get("data", np.eye(3).reshape(-1)),
                dtype=np.float64,
            ),
            p=np.asarray(info["projection_matrix"]["data"], dtype=np.float64),
            distortion_model=str(info.get("distortion_model", "")),
        )
    return calibrations


def _camera_name_from_topic(topic: str) -> str:
    for part in topic.strip("/").split("/"):
        if part.startswith("camera") and part[len("camera") :].isdigit():
            return part
    raise ValueError(f"Could not resolve camera name from topic: {topic}")


def _load_camera_frame_id(tier4_c2: Path, camera_name: str) -> str:
    suffix = camera_name.removeprefix("camera")
    candidates = [
        tier4_c2 / f"{camera_name}_sxpf.param.yaml",
        tier4_c2 / f"v4l2_{suffix}.param.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        params = (_load_yaml(path).get("/**", {}) or {}).get("ros__parameters", {})
        frame_id = params.get("frame_id") or params.get("camera_frame_id")
        if frame_id:
            return str(frame_id)
    raise FileNotFoundError(f"Missing camera frame param for {camera_name} under {tier4_c2}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as fp:
        return yaml.safe_load(fp) or {}
