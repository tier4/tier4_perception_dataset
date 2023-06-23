from enum import Enum
from typing import Dict, List


class T4_FORMAT_DIRECTORY_NAME(Enum):
    ANNOTATION = "annotation"
    DATA = "data"


class SENSOR_MODALITY_ENUM(Enum):
    LIDAR = "lidar"
    CAMERA = "camera"
    RADAR = "radar"


class SENSOR_ENUM(Enum):
    CAM_BACK_LEFT = {
        "channel": "CAM_BACK_LEFT",
        "modality": SENSOR_MODALITY_ENUM.CAMERA.value,
    }
    CAM_FRONT = {
        "channel": "CAM_FRONT",
        "modality": SENSOR_MODALITY_ENUM.CAMERA.value,
    }
    CAM_FRONT_RIGHT = {
        "channel": "CAM_FRONT_RIGHT",
        "modality": SENSOR_MODALITY_ENUM.CAMERA.value,
    }
    CAM_BACK_RIGHT = {
        "channel": "CAM_BACK_RIGHT",
        "modality": SENSOR_MODALITY_ENUM.CAMERA.value,
    }
    CAM_BACK = {
        "channel": "CAM_BACK",
        "modality": SENSOR_MODALITY_ENUM.CAMERA.value,
    }
    CAM_FRONT_LEFT = {
        "channel": "CAM_FRONT_LEFT",
        "modality": SENSOR_MODALITY_ENUM.CAMERA.value,
    }
    CAM_TRAFFIC_LIGHT_NEAR = {
        "channel": "CAM_TRAFFIC_LIGHT_NEAR",
        "modality": SENSOR_MODALITY_ENUM.CAMERA.value,
    }
    CAM_TRAFFIC_LIGHT_FAR = {
        "channel": "CAM_TRAFFIC_LIGHT_FAR",
        "modality": SENSOR_MODALITY_ENUM.CAMERA.value,
    }
    LIDAR_TOP = {
        "channel": "LIDAR_TOP",
        "modality": SENSOR_MODALITY_ENUM.LIDAR.value,
    }
    LIDAR_CONCAT = {
        "channel": "LIDAR_CONCAT",
        "modality": SENSOR_MODALITY_ENUM.LIDAR.value,
    }
    RADAR_FRONT = {
        "channel": "RADAR_FRONT",
        "modality": SENSOR_MODALITY_ENUM.RADAR.value,
    }
    RADAR_FRONT_RIGHT = {
        "channel": "RADAR_FRONT_RIGHT",
        "modality": SENSOR_MODALITY_ENUM.RADAR.value,
    }
    RADAR_FRONT_LEFT = {
        "channel": "RADAR_FRONT_LEFT",
        "modality": SENSOR_MODALITY_ENUM.RADAR.value,
    }
    RADAR_BACK_LEFT = {
        "channel": "RADAR_BACK_LEFT",
        "modality": SENSOR_MODALITY_ENUM.RADAR.value,
    }
    RADAR_BACK_RIGHT = {
        "channel": "RADAR_BACK_RIGHT",
        "modality": SENSOR_MODALITY_ENUM.RADAR.value,
    }

    @staticmethod
    def has_value(item) -> bool:
        return item in [v.value for v in SENSOR_ENUM.__members__.values()]

    @staticmethod
    def has_channel(item) -> bool:
        return item in [v.value["channel"] for v in SENSOR_ENUM.__members__.values()]

    @staticmethod
    def values() -> List[Dict[str, str]]:
        return [v.value for v in SENSOR_ENUM.__members__.values()]

    def get_sensor_modality(sensor_channel: str) -> str:
        for sensor in SENSOR_ENUM.__members__.values():
            if sensor.value["channel"] == sensor_channel:
                return sensor.value["modality"]
        raise ValueError(f"No sensor_channel {sensor_channel}")


class EXTENSION_ENUM(Enum):
    JPG = ".jpg"
    XML = ".xml"
    JSON = ".json"
    PNG = ".png"
    TXT = ".txt"
    CSV = ".csv"
    ONNX = ".onnx"
    PCD = ".pcd"
    BIN = ".bin"
    PCDBIN = ".pcd.bin"

    @staticmethod
    def has_value(item) -> bool:
        return item in [v.value for v in EXTENSION_ENUM.__members__.values()]

    @staticmethod
    def values() -> List[str]:
        return [v.value for v in EXTENSION_ENUM.__members__.values()]


def constant(f):
    def fset(self, value):
        raise TypeError

    def fget(self):
        return f()

    return property(fget, fset)
