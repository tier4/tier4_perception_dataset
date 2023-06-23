from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.scene import SceneRecord, SceneTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "name": "the name of scene",
        "description": "the description of scene",
        "log_token": "log_token_xxx",
        "nbr_samples": 10,
        "first_sample_token": "first_sample_token_xxx",
        "last_sample_token": "last_sample_token_xxx",
    }
    return d


class TestSceneRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return SceneRecord(**record_data)

    def test_to_dict(self, record_for_test: SceneRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()

        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["name"], str)
        assert isinstance(rec_dict["description"], str)
        assert isinstance(rec_dict["log_token"], str)
        assert isinstance(rec_dict["nbr_samples"], int)
        assert isinstance(rec_dict["first_sample_token"], str)
        assert isinstance(rec_dict["last_sample_token"], str)

        for key, value in record_data.items():
            assert rec_dict[key] == value, key


class TestSceneTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return SceneTable()

    def test_filename(self, table_for_test: SceneTable):
        assert table_for_test.FILENAME == "scene.json"

    def test__to_record(self, table_for_test: SceneTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, SceneRecord)
