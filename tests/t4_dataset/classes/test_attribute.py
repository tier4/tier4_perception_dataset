from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.attribute import AttributeRecord, AttributeTable


@pytest.fixture(scope="function", autouse=True)
def record_data():
    d = {
        "name": "test_attribute_name",
        "description": "the description of the attribute",
    }
    return d


class TestAttributeRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return AttributeRecord(**record_data)

    def test_to_dict(self, record_for_test: AttributeRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()
        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["name"], str)
        assert isinstance(rec_dict["description"], str)
        assert rec_dict["name"] == record_data["name"]
        assert rec_dict["description"] == record_data["description"]


class TestAttributeTable:
    # TODO(yukke42): impl TestAttributeTable with name_to_description
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return AttributeTable(name_to_description={}, default_value="")

    def test_filename(self, table_for_test: AttributeTable):
        assert table_for_test.FILENAME == "attribute.json"

    def test__to_record(self, table_for_test: AttributeTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, AttributeRecord)

    def test_get_token_from_name(self, table_for_test: AttributeTable):
        # TODO(yukke42): impl test_get_token_from_name with name_to_description
        token1 = table_for_test.get_token_from_name(name="car.moving")
        assert isinstance(token1, str)
        assert token1 != ""

        # same token
        token2 = table_for_test.get_token_from_name(name="car.moving")
        assert isinstance(token2, str)
        assert token2 != ""
        assert token2 == token1

        # different token
        token3 = table_for_test.get_token_from_name(name="car.parked")
        assert isinstance(token3, str)
        assert token3 != ""
        assert token3 != token1
