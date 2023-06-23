from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.sample_data import SampleDataRecord, SampleDataTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "sample_token": "sample_token_xxx",
        "ego_pose_token": "ego_pose_token_xxx",
        "calibrated_sensor_token": "calibrated_sensor_token_xxx",
        "filename": "data/LIDAR_TOP/0.pcd.bin",
        "fileformat": "pcd.bin",
        "width": 100,
        "height": 200,
        "timestamp": 123456789,
        "is_key_frame": True,
        "next_token": "next_token_xxx",
        "prev_token": "prev_token_xxx",
        "is_valid": True,
    }
    return d


class TestSampleDataRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return SampleDataRecord(**record_data)

    def test_to_dict(self, record_for_test: SampleDataRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()

        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["sample_token"], str)
        assert isinstance(rec_dict["ego_pose_token"], str)
        assert isinstance(rec_dict["calibrated_sensor_token"], str)
        assert isinstance(rec_dict["filename"], str)
        assert isinstance(rec_dict["fileformat"], str)
        assert isinstance(rec_dict["width"], int)
        assert isinstance(rec_dict["height"], int)
        assert isinstance(rec_dict["timestamp"], int)
        assert isinstance(rec_dict["is_key_frame"], bool)
        assert isinstance(rec_dict["next"], str)
        assert isinstance(rec_dict["prev"], str)

        for key, value in record_data.items():
            if key == "next_token" or key == "prev_token":
                key = key.replace("_token", "")
                continue
            assert rec_dict[key] == value, key


class TestSampleDataTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return SampleDataTable()

    def test_filename(self, table_for_test: SampleDataTable):
        assert table_for_test.FILENAME == "sample_data.json"

    def test__to_record(self, table_for_test: SampleDataTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, SampleDataRecord)
