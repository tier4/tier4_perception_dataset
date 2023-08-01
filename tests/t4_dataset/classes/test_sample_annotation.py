from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.classes.sample_annotation import (
    SampleAnnotationRecord,
    SampleAnnotationTable,
)


@pytest.fixture(scope="function")
def record_data():
    d = {
        "sample_token": "sample_token_xxxxx",
        "instance_token": "instance_token_xxxxx",
        "attribute_tokens": ["attribute_token_xxxxx", "attribute_token_yyyyy"],
        "visibility_token": "visibility_token_xxxxx",
        "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
        "velocity": {"x": 1.0, "y": 2.0, "z": 0.0},
        "acceleration": {"x": 1.0, "y": 2.0, "z": 0.0},
        "size": {"width": 10.0, "length": 20.0, "height": 30.0},
        "rotation": {"w": 100.0, "x": 200.0, "y": 300.0, "z": 400.0},
        "num_lidar_pts": 1000,
        "num_radar_pts": 2000,
    }
    return d


class TestSampleAnnotationRecord:
    @pytest.fixture(scope="function")
    def record_for_test(self, record_data):
        return SampleAnnotationRecord(**record_data)

    def text_next(self, record_for_test: SampleAnnotationRecord):
        next_token = "next_token_xxxxx"
        record_for_test.next = next_token
        assert record_for_test.next == next_token

    def text_prev(self, record_for_test: SampleAnnotationRecord):
        prev_token = "prev_token_xxxxx"
        record_for_test.prev = prev_token
        assert record_for_test.prev == prev_token

    def test_to_dict(self, record_for_test: SampleAnnotationRecord, record_data: Dict[str, Any]):
        translation_list = [
            record_data["translation"]["x"],
            record_data["translation"]["y"],
            record_data["translation"]["z"],
        ]
        velocity_list = [
            record_data["velocity"]["x"],
            record_data["velocity"]["y"],
            record_data["velocity"]["z"],
        ]
        acceleration_list = [
            record_data["acceleration"]["x"],
            record_data["acceleration"]["y"],
            record_data["acceleration"]["z"],
        ]
        size_list = [
            record_data["size"]["width"],
            record_data["size"]["length"],
            record_data["size"]["height"],
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
        assert isinstance(rec_dict["sample_token"], str)
        assert isinstance(rec_dict["instance_token"], str)
        assert isinstance(rec_dict["attribute_tokens"], list)
        assert all(isinstance(token, str) for token in rec_dict["attribute_tokens"])
        assert isinstance(rec_dict["visibility_token"], str)
        for key in ["translation", "size", "rotation"]:
            assert isinstance(rec_dict[key], list)
            assert all(isinstance(v, float) for v in rec_dict[key])
        for key in ("velocity", "acceleration"):
            assert isinstance(rec_dict[key], list)
            assert all(isinstance(v, float) or v is None for v in rec_dict[key])
        assert isinstance(rec_dict["num_lidar_pts"], int)
        assert isinstance(rec_dict["num_radar_pts"], int)

        assert rec_dict["sample_token"] == record_data["sample_token"]
        assert rec_dict["instance_token"] == record_data["instance_token"]
        assert rec_dict["attribute_tokens"] == record_data["attribute_tokens"]
        assert rec_dict["visibility_token"] == record_data["visibility_token"]
        assert rec_dict["translation"] == translation_list
        assert rec_dict["velocity"] == velocity_list
        assert rec_dict["acceleration"] == acceleration_list
        assert rec_dict["size"] == size_list
        assert rec_dict["rotation"] == rotation_list
        assert rec_dict["num_lidar_pts"] == record_data["num_lidar_pts"]
        assert rec_dict["num_radar_pts"] == record_data["num_radar_pts"]


class TestSampleAnnotationTable:
    @pytest.fixture(scope="function")
    def table_for_test(self):
        return SampleAnnotationTable()

    def test_filename(self, table_for_test: SampleAnnotationTable):
        assert table_for_test.FILENAME == "sample_annotation.json"

    def test__to_record(self, table_for_test: SampleAnnotationTable, record_data: Dict[str, Any]):
        record = table_for_test._to_record(**record_data)
        assert isinstance(record, SampleAnnotationRecord)
