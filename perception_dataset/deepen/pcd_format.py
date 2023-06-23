from abc import ABCMeta, abstractmethod
from collections import defaultdict
import json
import os
import os.path as osp
from typing import Any, Dict, List

from nptyping import NDArray
import numpy as np

from perception_dataset.constants import EXTENSION_ENUM, SENSOR_ENUM, SENSOR_MODALITY_ENUM


class AbstractData(metaclass=ABCMeta):
    @abstractmethod
    def to_dict(self) -> Dict:
        raise NotImplementedError()

    def save_json(self, output_path: str):
        """
        Args:
            output_path (str): [description]
        """
        assert self.FILENAME is not None, f"This instance cannot be saved: {__class__.__name__}"

        with open(osp.join(output_path, self.FILENAME), "w") as f:
            json.dump(self.to_dict(), f, indent=4)


class Sensor(AbstractData):
    FILENAME = None

    def __init__(self, sensor_type: SENSOR_ENUM, extension: EXTENSION_ENUM = None):
        """an object containing the details of all the sensors (lidar and cameras) present in the dataset.

        Args:
            data_dir (str): the relative path of the folder where the data for the sensor is present.
                for image, {pcf_file_name}.pcd.jpg or {pcd_file_name}.pcd.png
            sensor_type (SENSOR_ENUM): lidar or camera
            extension (EXTENSION_ENUM, optional): Valid only for sensor_type ‘camera’.
                Supported formats are ‘jpg’ and ‘png’. Defaults to None.
        """
        if sensor_type.value["modality"] == SENSOR_MODALITY_ENUM.CAMERA.value:
            assert extension is not None, "for camera, image extension must be used."
            assert extension in [
                EXTENSION_ENUM.JPG,
                EXTENSION_ENUM.PNG,
            ], "image extension must be jpg or png"

        self._sensor_type: SENSOR_ENUM = sensor_type
        self._sensor_fusion: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
        self._extension: EXTENSION_ENUM = extension

    @property
    def channel(self):
        return self._sensor_type.value["channel"]

    @property
    def modality(self):
        return self._sensor_type.value["modality"]

    @property
    def filepath(self) -> str:
        """this is `content` value of each sensor in Config.json"""
        return osp.join("data", self.channel)

    def get_filename(self, frame_index: int):
        """
        the relative path of the folder where the data for the sensor is present.
        for image, {pcf_file_name}.pcd.jpg or {pcd_file_name}.pcd.png

        Args:
            frame_index (int): [description]

        Returns:
            filename (str): [description]
        """
        filename = osp.join(self.filepath, f"{frame_index}.pcd")
        if self.modality == SENSOR_MODALITY_ENUM.CAMERA.value:
            filename += self._extension.value
        return filename

    def make_directory(self, output_path: str):
        os.makedirs(osp.join(output_path, self.filepath), exist_ok=True)

    def add_sensor_fusion(self, camera_sensor_type: SENSOR_ENUM):
        camera_channel = camera_sensor_type.value["channel"]
        self._sensor_fusion[camera_channel]["view_matrix"] = np.eye(4).tolist()

    def to_dict(self) -> Dict[str, Any]:
        value = {
            "content": self.filepath,
            "sensor_type": self.modality,
        }
        if self.modality == SENSOR_MODALITY_ENUM.LIDAR.value:
            # value["sensor_fusion"] = self._sensor_fusion
            pass
        else:
            value["extension"] = self._extension.value[1:]

        return value


class Config(AbstractData):
    """This is the configuration file for the dataset. This contains the information about primary sensor, cameras,
    the view matrix for each camera and the directory the images and lidar files are present in. A sample
    config.json data is given below:

    {
        "primary_sensor_id": "lidar",
        "sensors": {
            "lidar": {
                "content": "lidar",
                "sensor_type": "lidar",
                "sensor_fusion": {
                    "camera_0": {
                        "view_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ]
                    },
                    "camera_1": {
                        "view_matrix": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ]
                    }
                }
            },
            "camera_0": {
                "content": "images/camera_0_img",
                "sensor_type": "camera",
                "extension": "jpg",
            },
            "camera_1": {
                "content": "images/camera_1_img",
                "sensor_type": "camera",
                "extension": "jpg",
            },
        },
    }

    """

    FILENAME = "config" + EXTENSION_ENUM.JSON.value

    def __init__(self, lidar_sensor: Sensor):
        """
        Args:
            primary_sensor_id (int): the name/id of the lidar sensor
        """
        self._lidar_sensor: Sensor = lidar_sensor
        self._camera_sensors: Dict[str, Sensor] = {}

    @property
    def camera_sensors(self) -> List[Sensor]:
        return list(self._camera_sensors.values())

    def make_directories(self, output_path: str):
        self._lidar_sensor.make_directory(output_path)
        for camera_sensor in self._camera_sensors.values():
            camera_sensor.make_directory(output_path)

    def add_camera(self, camera_sensor_type: SENSOR_ENUM, extension: EXTENSION_ENUM):
        camera_sensor = Sensor(
            sensor_type=camera_sensor_type,
            extension=extension,
        )
        self._camera_sensors[camera_sensor.channel] = camera_sensor
        self._lidar_sensor.add_sensor_fusion(camera_sensor_type=camera_sensor_type)

    def to_dict(self) -> Dict[str, Any]:
        lidar_modality = self._lidar_sensor.modality
        # sensor_id of LiDAR must be lidar, if not upload error
        value = {
            "primary_sensor_id": lidar_modality,
            "sensors": {
                lidar_modality: self._lidar_sensor.to_dict(),
            },
        }

        for camera_channel, camera_sensor in self._camera_sensors.items():
            value["sensors"][camera_channel] = camera_sensor.to_dict()

        return value


class LidarTransforms(AbstractData):
    """This file contains the transformation matrices used to convert the lidar points in the lidar (local) frame of
    reference to the world (global) frame of reference.
    """

    FILENAME = "lidar_transforms" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        self._forward_transforms: Dict[int, List[List[float]]] = {}
        self._inverse_transforms: Dict[int, List[List[float]]] = {}

    def add_transform(self, frame_index: int, forward_transform: NDArray):
        """[summary]

        Args:
            frame_index (int): the index of frame in a scene
            forward_transform (NDArray): transform the points in the lidar frame to world frame
        """
        assert forward_transform.shape == (4, 4), "forward_transform must be (4, 4) matrix."
        inverse_transform = np.linalg.inv(forward_transform)

        self._forward_transforms[frame_index] = forward_transform.tolist()
        self._inverse_transforms[frame_index] = inverse_transform.tolist()

    def to_dict(self) -> Dict[str, Dict[int, List[List[float]]]]:
        value = {
            "forward_transforms": self._forward_transforms,
            "inverse_transforms": self._inverse_transforms,
        }
        return value


class ViewMatrices(AbstractData):
    """This file contains the view_matrices of each camera for each frame. The json object has frame index as the
        key and the value similar to sensor_fusion object as mentioned in the config.json

    {
        "0": {
            "camera_0": {
                "view_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            },
            "camera_1": {
                "view_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            }
        },
        "1": {
            "camera_0": {
                "view_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            },
            "camera_1": {
                "view_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            }
        },
        ...
    }

    """

    FILENAME = "view_matrices" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        self._view_matrices: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = defaultdict(dict)

    def add_view_matrix(self, frame_index: int, sensor_channel: str, view_matrix: NDArray):
        """

        Args:
            frame_index (int): the index of frame in a scene
            sensor_channel (str): the channel of sensor
            view_matrix (NDArray): : the transformation matrix used to transform the points from lidar
                coordinates to image coordinates
        """
        assert view_matrix.shape == (4, 4), "view_matrix must be (4, 4) matrix."
        self._view_matrices[frame_index][sensor_channel] = {"view_matrix": view_matrix.tolist()}

    def to_dict(self) -> Dict:
        return self._view_matrices
