from pathlib import Path

import pytest
import yaml

from perception_dataset.t4_v1.vehicle_table import VehicleTable

SAMPLE_TABLE = Path(__file__).parents[2] / "config" / "vehicle_info" / "vehicles_sample.yaml"


def test_sample_table():
    model = VehicleTable.load(SAMPLE_TABLE).find("sample_vehicle_01")

    assert model.vehicle_fields() == {
        "model": "sample_vehicle",
        "sensor_rig_id": "sample_sensor_kit",
        "length": 4.89,
        "width": 1.896,
        "height": 2.5,
        "wheel_base": 2.79,
        "rear_axle_to_center": 1.345,
    }
    assert model.camera_hardware("camera0").model == "Sample Wide Camera"
    assert model.camera_hardware("camera1").model == "Sample Camera"


def test_unknown_and_partial_models(tmp_path):
    path = tmp_path / "vehicles.yaml"
    entries = {
        "no_info": {"sensor_kit": "kit", "vehicle_names": ["bus_*"]},
        "other": {"vehicle_names": ["bus_0*"]},
    }
    path.write_text(yaml.safe_dump(entries))
    table = VehicleTable.load(path)

    assert table.find("taxi_01") is None
    assert table.find("bus_10").vehicle_fields() == {"model": "no_info", "sensor_rig_id": "kit"}
    assert table.find("bus_10").camera_hardware("camera0") is None
    with pytest.raises(ValueError):
        table.find("bus_01")
