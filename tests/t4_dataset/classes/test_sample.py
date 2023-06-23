from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.sample import SampleRecord, SampleTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "timestamp": 123456789,
        "scene_token": "scene_token_xxx",
        "next_token": "next_token_xxx",
        "prev_token": "prev_token_xxx",
    }
    return d


class TestSampleRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return SampleRecord(**record_data)

    def test_to_dict(self, record_for_test: SampleRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()

        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["timestamp"], int)
        assert isinstance(rec_dict["scene_token"], str)
        assert isinstance(rec_dict["next"], str)
        assert isinstance(rec_dict["prev"], str)

        for key, value in record_data.items():
            if key == "next_token" or key == "prev_token":
                key = key.replace("_token", "")
                continue
            assert rec_dict[key] == value, key


class TestSampleTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return SampleTable()

    def test_filename(self, table_for_test: SampleTable):
        assert table_for_test.FILENAME == "sample.json"

    def test__to_record(self, table_for_test: SampleTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, SampleRecord)
