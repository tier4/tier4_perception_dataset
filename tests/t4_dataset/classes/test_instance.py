from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.instance import InstanceRecord, InstanceTable


@pytest.fixture(scope="function")
def record_data():
    d = {
        "category_token": "category_token_xxxxx",
    }
    return d


class TestInstanceRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return InstanceRecord(**record_data)

    def test_to_dict(self, record_for_test: InstanceRecord, record_data: Dict[str, Any]):
        rec_dict = record_for_test.to_dict()
        assert isinstance(rec_dict, dict)
        assert isinstance(rec_dict["token"], str)
        assert isinstance(rec_dict["category_token"], str)
        assert rec_dict["category_token"] == record_data["category_token"]

    def test_set_annotation_info(self, record_for_test: InstanceRecord):
        nbr_annotations = 100
        first_annotation_token = "first_annotation_token_xxxxx"
        last_annotation_token = "last_annotation_token_xxxxx"

        record_for_test.set_annotation_info(
            nbr_annotations=nbr_annotations,
            first_annotation_token=first_annotation_token,
            last_annotation_token=last_annotation_token,
        )
        rec_dict = record_for_test.to_dict()
        assert isinstance(rec_dict["nbr_annotations"], int)
        assert isinstance(rec_dict["first_annotation_token"], str)
        assert isinstance(rec_dict["last_annotation_token"], str)
        assert rec_dict["nbr_annotations"] == nbr_annotations
        assert rec_dict["first_annotation_token"] == first_annotation_token
        assert rec_dict["last_annotation_token"] == last_annotation_token


class TestInstanceTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return InstanceTable()

    def test_filename(self, table_for_test: InstanceTable):
        assert table_for_test.FILENAME == "instance.json"

    def test__to_record(self, table_for_test: InstanceTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, InstanceRecord)

    def test_get_token_from_id(self, table_for_test: InstanceTable):
        dataset_name: str = "dataset_name_xxxxx"
        token1 = table_for_test.get_token_from_id(
            instance_id="car1", category_token="car_xxxxx", dataset_name=dataset_name
        )
        assert isinstance(token1, str)
        assert token1 != ""

        # same token
        token2 = table_for_test.get_token_from_id(
            instance_id="car1", category_token="car_xxxxx", dataset_name=dataset_name
        )
        assert isinstance(token2, str)
        assert token2 != ""
        assert token2 == token1

        # different token
        token3 = table_for_test.get_token_from_id(
            instance_id="car2", category_token="car_xxxxx", dataset_name=dataset_name
        )
        assert isinstance(token3, str)
        assert token3 != ""
        assert token3 != token1
