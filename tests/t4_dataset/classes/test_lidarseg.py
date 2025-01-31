from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.lidarseg import LidarSegRecord, LidarSegTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "sample_data_token": "test_sample_data_token",
        "filename": "tmp/test_file.bin",
    }
    return d


class TestLidarSegRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return LidarSegRecord(**record_data)

    def test_to_dict(self, record_for_test: LidarSegRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()
        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["sample_data_token"], str)
        assert isinstance(rec_dict["filename"], str)
        assert rec_dict["sample_data_token"] == record_data["sample_data_token"]
        assert rec_dict["filename"] == record_data["filename"]


class TestLidarSegTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return LidarSegTable()

    def test_filename(self, table_for_test: LidarSegTable):
        assert table_for_test.FILENAME == "lidarseg.json"

    def test__to_record(self, table_for_test: LidarSegTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, LidarSegRecord)
