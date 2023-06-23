from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.ego_pose import EgoPoseRecord, EgoPoseTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
        "rotation": {"w": 10.0, "x": 20.0, "y": 30.0, "z": 40.0},
        "timestamp": 123456789,
    }
    return d


class TestEgoPoseRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return EgoPoseRecord(**record_data)

    def test_to_dict(self, record_for_test: EgoPoseRecord, record_data: Dict[str, Any]):
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
        assert isinstance(rec_dict["translation"], list)
        assert any(isinstance(t, float) for t in rec_dict["translation"])
        assert isinstance(rec_dict["rotation"], list)
        assert any(isinstance(t, float) for t in rec_dict["rotation"])
        assert isinstance(rec_dict["timestamp"], int)

        assert rec_dict["translation"] == translation_list
        assert rec_dict["rotation"] == rotation_list
        assert rec_dict["timestamp"] == record_data["timestamp"]


class TestEgoPoseTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return EgoPoseTable()

    def test_filename(self, table_for_test: EgoPoseTable):
        assert table_for_test.FILENAME == "ego_pose.json"

    def test__to_record(self, table_for_test: EgoPoseTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, EgoPoseRecord)
