import numpy as np
import pytest

from perception_dataset.constants import EXTENSION_ENUM, SENSOR_ENUM
from perception_dataset.deepen.json_format import ConfigData, ImageData


def test_image_data():
    frame_index = 100
    channel = SENSOR_ENUM.CAM_FRONT.value["channel"]
    fileformat = EXTENSION_ENUM.PNG.value[1:]
    unix_timestamp = 1635353737.10000
    device_position = np.array([1, 2, 3], dtype=np.float32)
    device_heading = np.array([10, 20, 30, 40], dtype=np.float32)
    fx = 100.0
    fy = 200.0
    cx = 300.0
    cy = 400.0
    camera_intrinsic_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    filepath = f"data/{channel}/{frame_index}.{fileformat}"

    image_data = ImageData(
        frame_index=frame_index,
        channel=channel,
        fileformat=fileformat,
        unix_timestamp=unix_timestamp,
        device_position=device_position,
        device_heading=device_heading,
        camera_intrinsic_matrix=camera_intrinsic_matrix,
    )

    assert image_data.filepath == filepath

    image_data_dict = image_data.to_dict()
    assert isinstance(image_data_dict["fx"], float)
    assert isinstance(image_data_dict["fy"], float)
    assert isinstance(image_data_dict["cx"], float)
    assert isinstance(image_data_dict["cy"], float)
    assert isinstance(image_data_dict["timestamp"], float)
    assert isinstance(image_data_dict["image_url"], str)
    assert isinstance(image_data_dict["position"], dict)
    assert isinstance(image_data_dict["position"]["x"], float)
    assert isinstance(image_data_dict["position"]["y"], float)
    assert isinstance(image_data_dict["position"]["z"], float)
    assert isinstance(image_data_dict["heading"], dict)
    assert isinstance(image_data_dict["heading"]["w"], float)
    assert isinstance(image_data_dict["heading"]["x"], float)
    assert isinstance(image_data_dict["heading"]["y"], float)
    assert isinstance(image_data_dict["heading"]["z"], float)
    assert isinstance(image_data_dict["camera_model"], str)
    assert isinstance(image_data_dict["k1"], float)
    assert isinstance(image_data_dict["k2"], float)
    assert isinstance(image_data_dict["p1"], float)
    assert isinstance(image_data_dict["p2"], float)
    assert isinstance(image_data_dict["k3"], float)
    assert isinstance(image_data_dict["k4"], float)
    assert image_data_dict["fx"] == pytest.approx(fx)
    assert image_data_dict["fy"] == pytest.approx(fy)
    assert image_data_dict["cx"] == pytest.approx(cx)
    assert image_data_dict["cy"] == pytest.approx(cy)
    assert image_data_dict["timestamp"] == pytest.approx(unix_timestamp)
    assert image_data_dict["image_url"] == filepath
    assert image_data_dict["camera_model"] == "pinhole"
    assert image_data_dict["position"]["x"] == pytest.approx(device_position[0])
    assert image_data_dict["position"]["y"] == pytest.approx(device_position[1])
    assert image_data_dict["position"]["z"] == pytest.approx(device_position[2])
    assert image_data_dict["heading"]["w"] == pytest.approx(device_heading[0])
    assert image_data_dict["heading"]["x"] == pytest.approx(device_heading[1])
    assert image_data_dict["heading"]["y"] == pytest.approx(device_heading[2])
    assert image_data_dict["heading"]["z"] == pytest.approx(device_heading[3])


def test_config_data():
    frame_index = 100
    unix_timestamp = 1635353737.10000
    device_position = np.array([1, 2, 3], dtype=np.float32)
    device_heading = np.array([10, 20, 30, 40], dtype=np.float32)
    points = np.arange(6, dtype=np.float32).reshape(2, 3)

    config_data = ConfigData(
        frame_index=frame_index,
        unix_timestamp=unix_timestamp,
        points=points,
        device_position=device_position,
        device_heading=device_heading,
    )

    assert config_data.filename == f"{frame_index}.json"

    config_data_dict = config_data.to_dict()
    assert isinstance(config_data_dict["images"], list)
    assert isinstance(config_data_dict["timestamp"], float)
    assert isinstance(config_data_dict["device_position"], dict)
    assert isinstance(config_data_dict["device_position"]["x"], float)
    assert isinstance(config_data_dict["device_position"]["y"], float)
    assert isinstance(config_data_dict["device_position"]["z"], float)
    assert isinstance(config_data_dict["device_heading"], dict)
    assert isinstance(config_data_dict["device_heading"]["w"], float)
    assert isinstance(config_data_dict["device_heading"]["x"], float)
    assert isinstance(config_data_dict["device_heading"]["y"], float)
    assert isinstance(config_data_dict["device_heading"]["z"], float)
    assert isinstance(config_data_dict["points"], list)
    assert isinstance(config_data_dict["points"][0], dict)
    assert isinstance(config_data_dict["points"][0]["x"], float)
    assert isinstance(config_data_dict["points"][0]["y"], float)
    assert isinstance(config_data_dict["points"][0]["z"], float)
    assert config_data_dict["timestamp"] == pytest.approx(unix_timestamp)
    assert config_data_dict["device_position"]["x"] == pytest.approx(device_position[0])
    assert config_data_dict["device_position"]["y"] == pytest.approx(device_position[1])
    assert config_data_dict["device_position"]["z"] == pytest.approx(device_position[2])
    assert config_data_dict["device_heading"]["w"] == pytest.approx(device_heading[0])
    assert config_data_dict["device_heading"]["x"] == pytest.approx(device_heading[1])
    assert config_data_dict["device_heading"]["y"] == pytest.approx(device_heading[2])
    assert config_data_dict["device_heading"]["z"] == pytest.approx(device_heading[3])
    for i in range(points.shape[0]):
        assert config_data_dict["points"][i]["x"] == pytest.approx(points[i, 0])
        assert config_data_dict["points"][i]["y"] == pytest.approx(points[i, 1])
        assert config_data_dict["points"][i]["z"] == pytest.approx(points[i, 2])
