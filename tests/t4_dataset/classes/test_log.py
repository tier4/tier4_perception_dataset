from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.log import LogRecord, LogTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "logfile": "path_to_logfile",
        "vehicle": "vehicle_name",
        "data_captured": "2020-01-01-00-00-00",
        "location": "log_location",
    }
    return d


class TestLogRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return LogRecord(**record_data)

    def test_to_dict(self, record_for_test: LogRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()

        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["logfile"], str)
        assert isinstance(rec_dict["vehicle"], str)
        assert isinstance(rec_dict["data_captured"], str)
        assert isinstance(rec_dict["location"], str)

        for key, value in record_data.items():
            assert rec_dict[key] == value


class TestLogTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return LogTable()

    def test_filename(self, table_for_test: LogTable):
        assert table_for_test.FILENAME == "log.json"

    def test__to_record(self, table_for_test: LogTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, LogRecord)
