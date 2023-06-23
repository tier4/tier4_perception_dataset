import json
import os.path as osp
import tempfile
from typing import Dict, List

import pytest

from perception_dataset.constants import EXTENSION_ENUM
from perception_dataset.t4_dataset.classes.abstract_class import AbstractRecord, AbstractTable


class AbstractRecordForTest(AbstractRecord):
    def __init__(self):
        super().__init__()

    def to_dict(self) -> Dict[str, str]:
        return {"token": self.token}


class AbstractTableForTest(AbstractTable):
    FILENAME = "test" + EXTENSION_ENUM.JSON.value

    def __init__(self):
        super().__init__()

    def _to_record(self, **kwargs) -> str:
        return AbstractRecordForTest()

    @classmethod
    def from_json(cls, filepath: str):
        return AbstractTableForTest()


class TestAbstractRecord:
    def test_token(self):
        AbstractRecord.__abstractmethods__ = set()
        rec = AbstractRecord()

        assert isinstance(rec.token, str)
        assert rec.token != ""

    def test_to_dict(self):
        """test of the impl for test_abstract_table"""
        rec = AbstractRecordForTest()

        rec_dict = rec.to_dict()
        assert isinstance(rec_dict, dict)
        assert rec_dict["token"] != ""


class TestAbstractTable:
    @pytest.fixture(scope="function")
    def record_for_test(self):
        return AbstractRecordForTest()

    @pytest.fixture(scope="function")
    def table_for_test(self):
        return AbstractTableForTest()

    def test___len__(self, table_for_test: AbstractTableForTest):
        len_table = len(table_for_test)
        assert isinstance(len_table, int)
        assert len_table == 0

    def test_filename(self, table_for_test: AbstractTableForTest):
        """test of the impl for TestAbstractTable"""
        assert isinstance(table_for_test.FILENAME, str)
        assert table_for_test.FILENAME.endswith(".json")

    def test__to_record(self, table_for_test: AbstractTableForTest):
        """test of the impl for TestAbstractTable"""
        rec = table_for_test._to_record()
        assert isinstance(rec, AbstractRecordForTest)

    def test_set_record_to_table(
        self, table_for_test: AbstractTableForTest, record_for_test: AbstractRecordForTest
    ):
        table_for_test.set_record_to_table(record_for_test)
        assert len(table_for_test) == 1

        # check encapsulated value
        assert record_for_test.token in table_for_test._token_to_record
        assert table_for_test._token_to_record[record_for_test.token] == record_for_test
        assert (
            table_for_test._token_to_record[record_for_test.token].token == record_for_test.token
        )

    def test_insert_into_table(self, table_for_test: AbstractTableForTest):
        token = table_for_test.insert_into_table()
        assert len(table_for_test) == 1

        # check encapsulated value
        assert token in table_for_test._token_to_record
        assert isinstance(table_for_test._token_to_record[token], AbstractRecordForTest)
        assert table_for_test._token_to_record[token].token == token

    def test_select_record_from_token(self, table_for_test: AbstractTableForTest):
        token = table_for_test.insert_into_table()

        record = table_for_test.select_record_from_token(token)
        assert isinstance(record, AbstractRecordForTest)
        assert record.token == token

    def test_to_data(self, table_for_test: AbstractTableForTest):
        token = table_for_test.insert_into_table()

        table_data = table_for_test.to_data()
        assert isinstance(table_data, list)
        assert all(isinstance(rec, dict) for rec in table_data)
        assert table_data[0]["token"] == token

    def test_to_records(self, table_for_test: AbstractTableForTest):
        token = table_for_test.insert_into_table()

        table_records = table_for_test.to_records()
        assert isinstance(table_records, list)
        assert all(isinstance(rec, AbstractRecordForTest) for rec in table_records)
        assert table_records[0].token == token

    def test_save_json(self, table_for_test: AbstractTableForTest):
        token = table_for_test.insert_into_table()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json_file = osp.join(tmp_dir, table_for_test.FILENAME)
            table_for_test.save_json(tmp_dir)

            assert osp.exists(tmp_json_file)

            with open(tmp_json_file) as f:
                json_data = json.load(f)

            assert isinstance(json_data, list)
            assert isinstance(json_data[0], dict)
            assert isinstance(json_data[0]["token"], str)
            assert len(json_data) == 1
            assert json_data[0]["token"] == token

    def test_for_multiple_records(self, table_for_test: AbstractTableForTest):
        NUM_TEST_RECORD = 3
        token_list: List[str] = []

        for i in range(NUM_TEST_RECORD):
            # test insert_into_table()
            token = table_for_test.insert_into_table()
            token_list.append(token)
            assert isinstance(token, str)

            # test __len__()
            assert len(table_for_test) == i + 1

            # test select_record_from_token()
            rec = table_for_test.select_record_from_token(token)
            assert isinstance(rec, AbstractRecordForTest)
            assert rec.token == token

            # test to_data()
            table_data = table_for_test.to_data()
            assert isinstance(table_data, list)
            assert isinstance(table_data[i], dict)
            assert table_data[i]["token"] == token

        # test to_data() of all records
        table_data = table_for_test.to_data()
        assert all(isinstance(rec, dict) for rec in table_data)
        for i in range(NUM_TEST_RECORD):
            assert table_data[i]["token"] == token_list[i]
