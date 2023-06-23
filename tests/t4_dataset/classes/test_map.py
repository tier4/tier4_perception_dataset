from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.map import MapRecord, MapTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "log_tokens": ["log_token_xxx"],
        "category": "map category",
        "filename": "map_filename",
    }
    return d


class TestMapRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return MapRecord(**record_data)

    def test_to_dict(self, record_for_test: MapRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()

        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["log_tokens"], list)
        assert any(isinstance(log, str) for log in rec_dict["log_tokens"])
        assert isinstance(rec_dict["category"], str)
        assert isinstance(rec_dict["category"], str)

        for key, value in record_data.items():
            assert rec_dict[key] == value, key


class TestMapTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return MapTable()

    def test_filename(self, table_for_test: MapTable):
        assert table_for_test.FILENAME == "map.json"

    def test__to_record(self, table_for_test: MapTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, MapRecord)
