from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.visibility import VisibilityRecord, VisibilityTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "level": "test_visibility_level",
        "description": "the description of the visibility",
    }
    return d


class TestAttributeRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return VisibilityRecord(**record_data)

    def test_to_dict(self, record_for_test: VisibilityRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()
        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["level"], str)
        assert isinstance(rec_dict["description"], str)
        assert rec_dict["level"] == record_data["level"]
        assert rec_dict["description"] == record_data["description"]


class TestAttributeTable:
    # TODO(yukke42): impl TestAttributeTable with level_to_description
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return VisibilityTable(level_to_description={}, default_value="")

    def test_filename(self, table_for_test: VisibilityTable):
        assert table_for_test.FILENAME == "visibility.json"

    def test__to_record(self, table_for_test: VisibilityTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, VisibilityRecord)

    def test_get_token_from_level(self, table_for_test: VisibilityTable):
        # TODO(yukke42): impl test_get_token_from_level with level_to_description
        token1 = table_for_test.get_token_from_level(level="v0-40")
        assert isinstance(token1, str)
        assert token1 != ""

        # same token
        token2 = table_for_test.get_token_from_level(level="v0-40")
        assert isinstance(token2, str)
        assert token2 != ""
        assert token2 == token1

        # different token
        token3 = table_for_test.get_token_from_level(level="v40-60")
        assert isinstance(token3, str)
        assert token3 != ""
        assert token3 != token1
