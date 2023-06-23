from typing import Any, Dict

import numpy as np
import pytest

from perception_dataset.t4_dataset.classes.calibrated_sensor import (
    CalibratedSensorRecord,
    CalibratedSensorTable,
)


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "sensor_token": "sensor_token_xxx",
        "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
        "rotation": {"w": 10.0, "x": 20.0, "y": 30.0, "z": 40.0},
        "camera_intrinsic": [],
        "camera_distortion": [],
    }
    return d


class TestCalibratedSensorRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return CalibratedSensorRecord(**record_data)

    def test_to_dict(self, record_for_test: CalibratedSensorRecord, record_data: Dict[str, Any]):
        translation_list = [
            record_data["translation"]["x"],
            record_data["translation"]["y"],
            record_data["translation"]["z"],
        ]
        rotation_list = [
            record_data["rotation"]["w"],
            record_data["rotation"]["x"],
            record_data["rotation"]["y"],
            record_data["rotation"]["z"],
        ]

        rec_dict = record_for_test.to_dict()
        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["sensor_token"], str)
        assert isinstance(rec_dict["translation"], list)
        assert any(isinstance(t, float) for t in rec_dict["translation"])
        assert isinstance(rec_dict["rotation"], list)
        assert any(isinstance(t, float) for t in rec_dict["rotation"])
        assert isinstance(rec_dict["camera_intrinsic"], list)
        assert isinstance(rec_dict["camera_distortion"], list)

        assert rec_dict["sensor_token"] == record_data["sensor_token"]
        assert rec_dict["translation"] == translation_list
        assert rec_dict["rotation"] == rotation_list
        assert len(rec_dict["camera_intrinsic"]) == 0 or np.array(
            rec_dict["camera_intrinsic"]
        ).shape == (3, 3)
        assert len(rec_dict["camera_distortion"]) == 0 or len(rec_dict["camera_distortion"]) == 5
        # TODO(yukke42): add test value of camera_intrinsic and camera_distortion


class TestCalibratedSensorTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return CalibratedSensorTable()

    def test_filename(self, table_for_test: CalibratedSensorTable):
        assert table_for_test.FILENAME == "calibrated_sensor.json"

    def test__to_record(self, table_for_test: CalibratedSensorTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, CalibratedSensorRecord)
