from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.sensor import SensorRecord, SensorTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "channel": "LIDAR_TOP",
        "modality": "lidar",
    }
    return d


class TestSensorRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return SensorRecord(**record_data)

    def test_to_dict(self, record_for_test: SensorRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()

        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["channel"], str)
        assert isinstance(rec_dict["modality"], str)

        for key, value in record_data.items():
            assert rec_dict[key] == value, key


class TestSceneTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return SensorTable(channel_to_modality={"LIDAR_TOP": "lidar"})

    def test_filename(self, table_for_test: SensorTable):
        assert table_for_test.FILENAME == "sensor.json"

    def test__to_record(self, table_for_test: SensorTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, SensorRecord)

    def test_get_token_from_channel(self, table_for_test: SensorTable):
        # TODO(yukke42): impl test_get_token_from_channel
        pass
