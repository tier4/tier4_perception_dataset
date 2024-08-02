from t4_devkit.schema import SensorModality


def test_sensor_modality() -> None:
    """Test SensorModality enum."""

    modalities = ("lidar", "camera", "radar")

    # check all enum members are covered by above names
    members: list[str] = [m.value for m in SensorModality]
    assert set(members) == set(modalities)

    # check each member can construct
    for value in modalities:
        _ = SensorModality(value)
