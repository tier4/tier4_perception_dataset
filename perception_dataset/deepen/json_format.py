from abc import ABCMeta, abstractmethod
from collections import defaultdict
import json
import os
import os.path as osp
import shutil
from typing import Any, Dict, List

from nptyping import NDArray
import numpy as np

from perception_dataset.constants import EXTENSION_ENUM


class AbstractData(metaclass=ABCMeta):
    @abstractmethod
    def to_dict(self) -> Dict:
        raise NotImplementedError()


class ImageData(AbstractData):
    """

    e.g.:
    {
        "fx": 1037.37598,
        "fy": 1040.97986,
        "cx": 742.10227,
        "cy": 600.99113,
        "timestamp": 1624164470.899887,
        "image_url": "image/0_cam_0.png",
        "position": {
            "x": 81532.011296,
            "y": 50369.700811,
            "z": 36.520526
        },
        "heading": {
            "x": -0.707906,
            "y": -0.070829,
            "z": 0.04981,
            "w": 0.700979
        },
        "camera_model ": "pinhone",
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
    }

    Args:
        AbstractData ([type]): [description]
    """

    def __init__(
        self,
        frame_index: int,
        channel: str,
        fileformat: str,
        unix_timestamp: float,
        device_position: NDArray = None,
        device_heading: NDArray = None,
        camera_intrinsic_matrix: NDArray = None,
        camera_model: str = "pinhole",
        camera_distortion: NDArray = None,
    ):
        assert frame_index > -1, "frame_index must be positive"
        assert fileformat in [EXTENSION_ENUM.JPG.value[1:], EXTENSION_ENUM.PNG.value[1:]]
        assert camera_model in ["pinhole", "fisheye"]

        if camera_model == "pinhole":
            assert (
                len(camera_distortion) == 5
            ), "For pinhole camera model, camera_distortion must have 5 parameters"
        elif camera_model == "fisheye":
            assert (
                len(camera_distortion) == 4
            ), "For fisheye camera model, camera_distortion must have 4 parameters"

        self._frame_index: int = frame_index
        self._channel: str = channel
        self._fileformat: str = fileformat
        self._unix_timestamp: float = unix_timestamp
        self._camera_model = camera_model
        self._fx: float = 0.0
        self._fy: float = 0.0
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._device_position: Dict[str, float] = defaultdict(float)
        self._device_heading: Dict[str, float] = defaultdict(float)
        self._k1: float = 0.0
        self._k2: float = 0.0
        self._p1: float = 0.0
        self._p2: float = 0.0
        self._k3: float = 0.0
        self._k4: float = 0.0

        if device_position is not None:
            self.add_device_position(device_position)
        if device_heading is not None:
            self.add_device_heading(device_heading)
        if camera_intrinsic_matrix is not None:
            self.add_intrinsic_calibration(camera_intrinsic_matrix)
        if camera_distortion is not None:
            self.add_camera_distortion(camera_distortion)

    @property
    def filepath(self):
        return osp.join("data", self._channel, f"{self._frame_index}.{self._fileformat}")

    def make_directory(self, output_path):
        os.makedirs(osp.join(output_path, osp.dirname(self.filepath)), exist_ok=True)

    def save(self, image_path: str, output_dir: str):
        shutil.copy(
            image_path,
            osp.join(output_dir, self.filepath),
        )

    def add_intrinsic_calibration(self, intrinsic_matrix: NDArray):
        assert intrinsic_matrix.shape == (3, 3), "intrinsic_matrix must be the shape of (3, 3)"
        intrinsic_matrix = intrinsic_matrix.astype(np.float32).tolist()
        self._fx: float = intrinsic_matrix[0][0]
        self._fy: float = intrinsic_matrix[1][1]
        self._cx: float = intrinsic_matrix[0][2]
        self._cy: float = intrinsic_matrix[1][2]

    def add_device_position(self, device_position: NDArray):
        """
        Args:
            device_position (NDArray): [x. y, z]
        """
        assert device_position.shape == (3,), "device_position must be the shape of (3,)"
        device_position = device_position.astype(np.float32).tolist()
        self._device_position = {
            "x": device_position[0],
            "y": device_position[1],
            "z": device_position[2],
        }

    def add_device_heading(self, device_heading: NDArray):
        """
        Args:
            device_heading (NDArray): [w, x, y, z]
        """
        assert device_heading.shape == (4,), "device_heading must be the shape of (4,)"
        device_heading = device_heading.astype(np.float32).tolist()
        self._device_heading = {
            "w": device_heading[0],
            "x": device_heading[1],
            "y": device_heading[2],
            "z": device_heading[3],
        }

    def add_camera_distortion(self, camera_distortion: NDArray):
        """
        Args:
            camera_distortion (NDArray): [k1, k2, p1, p2, k3, k4]
        """
        assert camera_distortion.shape == (6,), "camera_distortion must be the shape of (6,)"
        camera_distortion = camera_distortion.astype(np.float32).tolist()
        self._k1 = camera_distortion[0]
        self._k2 = camera_distortion[1]
        self._p1 = camera_distortion[2]
        self._p2 = camera_distortion[3]
        self._k3 = camera_distortion[4]
        self._k4 = camera_distortion[5]

    def to_dict(self):
        value = {
            "fx": self._fx,
            "fy": self._fy,
            "cx": self._cx,
            "cy": self._cy,
            "timestamp": self._unix_timestamp,
            "image_url": self.filepath,
            "position": self._device_position,
            "heading": self._device_heading,
            "camera_model": self._camera_model,
            "k1": self._k1,
            "k2": self._k2,
            "p1": self._p1,
            "p2": self._p2,
            "k3": self._k3,
            "k4": self._k4,
        }
        return value


class ConfigData(AbstractData):
    """

    Directory structure
    data_root
        |-  {frame_index}.json|
        |-  data
            |-  CAM_FRONT
                |-   {frame_index}.json
            |-  CAM_FRONT_RIGHT
                |-   {frame_index}.json
            ...
    """

    def __init__(
        self,
        frame_index: int,
        unix_timestamp: float,
        points: NDArray = None,
        device_position: NDArray = None,
        device_heading: NDArray = None,
        save_intensity: bool = False,
    ):
        self._save_intensity = save_intensity
        self._frame_index: int = frame_index
        self._image_data_list: List[ImageData] = []
        self._unix_timestamp: float = unix_timestamp
        self._points: List[Dict[str, float]] = []
        self._device_position: Dict[str, float] = defaultdict(float)
        self._device_heading: Dict[str, float] = defaultdict(float)

        if points is not None:
            self.add_points(points)
        if device_position is not None:
            self.add_device_position(device_position)
        if device_heading is not None:
            self.add_device_heading(device_heading)

    @property
    def filename(self):
        return str(self._frame_index) + EXTENSION_ENUM.JSON.value

    def to_dict(self) -> Dict[str, Any]:
        value = {
            "images": [image.to_dict() for image in self._image_data_list],
            "timestamp": self._unix_timestamp,
            "device_position": self._device_position,
            "device_heading": self._device_heading,
            "points": self._points,
        }
        return value

    def save_json(self, output_path: str):
        with open(osp.join(output_path, self.filename), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def add_image_data(self, image_data: ImageData):
        self._image_data_list.append(image_data)

    def add_points(self, points: NDArray):
        """
        Adds points in the point cloud, with optional intensity values.
        "x": X coordinate of the point,
        "y": Y coordinate of the point,
        "z": Z coordinate of the point,
        "i": (optional) Intensity of the point
        """
        assert points.ndim == 2 and points.shape[1] >= (
            4 if self._save_intensity else 3
        ), f"invalid points shape: {points.shape}"
        points = points.tolist()
        self._points = (
            [{"x": p[0], "y": p[1], "z": p[2], "i": p[3]} for p in points]
            if self._save_intensity
            else [{"x": p[0], "y": p[1], "z": p[2]} for p in points]
        )

    def add_device_position(self, device_position: NDArray):
        assert device_position.shape == (3,), "device_position must be the shape of (3,)"
        device_position = device_position.astype(np.float32).tolist()
        self._device_position = {
            "x": device_position[0],
            "y": device_position[1],
            "z": device_position[2],
        }

    def add_device_heading(self, device_heading: NDArray):
        assert device_heading.shape == (4,), "device_heading must be the shape of (4,)"
        device_heading = device_heading.astype(np.float32).tolist()
        self._device_heading = {
            "w": device_heading[0],
            "x": device_heading[1],
            "y": device_heading[2],
            "z": device_heading[3],
        }
