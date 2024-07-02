from t4_devkit.schema import SensorChannel, SensorModality


def test_sensor_modality() -> None:
    """Test SensorModality enum."""

    modalities = ("lidar", "camera", "radar")

    # check all enum members are covered by above names
    members: list[str] = [m.value for m in SensorModality]
    assert set(members) == set(modalities)

    # check each member can construct
    for value in modalities:
        _ = SensorModality(value)


def test_sensor_channel() -> None:
    """Test SensorChannel enum."""

    # {channel: modality}
    sensor_channels = {
        "CAM_FRONT": SensorModality.CAMERA,
        "CAM_FRONT_RIGHT": SensorModality.CAMERA,
        "CAM_FRONT_LEFT": SensorModality.CAMERA,
        "CAM_BACK": SensorModality.CAMERA,
        "CAM_BACK_RIGHT": SensorModality.CAMERA,
        "CAM_BACK_LEFT": SensorModality.CAMERA,
        "CAM_TRAFFIC_LIGHT_NEAR": SensorModality.CAMERA,
        "CAM_TRAFFIC_LIGHT_FAR": SensorModality.CAMERA,
        "LIDAR_TOP": SensorModality.LIDAR,
        "LIDAR_CONCAT": SensorModality.LIDAR,
        "RADAR_FRONT": SensorModality.RADAR,
        "RADAR_FRONT_RIGHT": SensorModality.RADAR,
        "RADAR_FRONT_LEFT": SensorModality.RADAR,
        "RADAR_BACK": SensorModality.RADAR,
        "RADAR_BACK_RIGHT": SensorModality.RADAR,
        "RADAR_BACK_LEFT": SensorModality.RADAR,
    }

    # check all enum members are covered by above names
    members: list[str] = [m.value for m in SensorChannel]
    assert set(members) == set(sensor_channels.keys())

    # check each member can construct and its method is valid
    for channel, modality in sensor_channels.items():
        member = SensorChannel(channel)

        # check modality
        assert member.modality == modality
