from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.category import CategoryRecord, CategoryTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "name": "test_category_name",
        "description": "the description of the category",
        "index": 1,
    }
    return d


class TestCategoryRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return CategoryRecord(**record_data)

    def test_to_dict(self, record_for_test: CategoryRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()
        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["name"], str)
        assert isinstance(rec_dict["description"], str)
        assert rec_dict["name"] == record_data["name"]
        assert rec_dict["description"] == record_data["description"]


class TestCategoryTable:
    # TODO(yukke42): impl TestCategoryTable with name_to_description
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return CategoryTable(name_to_description={}, default_value="", lidarseg=True)

    def test_filename(self, table_for_test: CategoryTable):
        assert table_for_test.FILENAME == "category.json"

    def test__to_record(self, table_for_test: CategoryTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, CategoryRecord)

    def test_get_index_from_token(self, table_for_test: CategoryTable) -> int:
        """Test retrieving index from a token."""
        token1 = table_for_test.get_token_from_name(name="test_category_name")
        index1 = table_for_test.get_index_from_token(token=token1)
        assert index1 == 1

        token2 = table_for_test.get_token_from_name(name="test_category_name_2")
        index2 = table_for_test.get_index_from_token(token=token2)
        assert index2 == 2

    def test_get_token_from_name(self, table_for_test: CategoryTable):
        # TODO(yukke42): impl test_get_token_from_name with description
        token1 = table_for_test.get_token_from_name(name="car")
        assert isinstance(token1, str)
        assert token1 != ""

        # same token
        token2 = table_for_test.get_token_from_name(name="car")
        assert isinstance(token2, str)
        assert token2 != ""
        assert token2 == token1

        # different token
        token3 = table_for_test.get_token_from_name(name="pedestrian")
        assert isinstance(token3, str)
        assert token3 != ""
        assert token3 != token1
