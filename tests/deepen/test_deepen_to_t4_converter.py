from typing import Any, Dict, List

import pytest

from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter


@pytest.fixture(scope="function")
def deepen_anno_list():
    # Note: only values required for t4dataset
    deepen_anno_list = [
        {
            "dataset_id": "dataset_xxx",
            "file_id": "0.pcd",
            "label_category_id": "car",
            "label_id": "car:1",
            "label_type": "3d_bbox",
            "attributes": {
                "state": "moving",
                "occlusion": "none",
                "cycle_state": "with_rider",
            },
            "labeller_email": "test@aaa.bbb",
            "sensor_id": "lidar",
            "three_d_bbox": {
                "cx": 1.0,
                "cy": 1.0,
                "cz": 1.0,
                "h": 1.0,
                "l": 1.0,
                "w": 1.0,
                "quaternion": {
                    "x": 1.0,
                    "y": 1.0,
                    "z": 1.0,
                    "w": 1.0,
                },
            },
        },
        {
            "dataset_id": "dataset_xxx",
            "file_id": "1.pcd",
            "label_category_id": "car",
            "label_id": "car:1",
            "label_type": "3d_bbox",
            "attributes": {
                "state": "moving",
                "occlusion": "none",
                "cycle_state": "with_rider",
            },
            "labeller_email": "test@aaa.bbb",
            "sensor_id": "lidar",
            "three_d_bbox": {
                "cx": 2.0,
                "cy": 2.0,
                "cz": 2.0,
                "h": 2.0,
                "l": 2.0,
                "w": 2.0,
                "quaternion": {
                    "x": 2.0,
                    "y": 2.0,
                    "z": 2.0,
                    "w": 2.0,
                },
            },
        },
        {
            "dataset_id": "dataset_xxx",
            "file_id": "1.pcd",
            "label_category_id": "car",
            "label_id": "car:1",
            "label_type": "3d_bbox",
            "attributes": {
                "state": "moving",
                "occlusion": "none",
                "cycle_state": "with_rider",
            },
            "labeller_email": "test@aaa.bbb",
            "sensor_id": "lidar",
            "three_d_bbox": {
                "cx": 3.0,
                "cy": 3.0,
                "cz": 3.0,
                "h": 3.0,
                "l": 3.0,
                "w": 3.0,
                "quaternion": {
                    "x": 3.0,
                    "y": 3.0,
                    "z": 3.0,
                    "w": 3.0,
                },
            },
        },
        {
            "dataset_id": "dataset_xxx",
            "file_id": "1.pcd",
            "label_category_id": "car",
            "label_id": "car:1",
            "label_type": "3d_bbox",
            "attributes": {
                "state": "moving",
                "occlusion": "none",
                "cycle_state": "with_rider",
            },
            "labeller_email": "auto_interpolation",
            "sensor_id": "lidar",
            "three_d_bbox": {
                "cx": 3.0,
                "cy": 3.0,
                "cz": 3.0,
                "h": 3.0,
                "l": 3.0,
                "w": 3.0,
                "quaternion": {
                    "x": 3.0,
                    "y": 3.0,
                    "z": 3.0,
                    "w": 3.0,
                },
            },
        },
    ]

    return deepen_anno_list


class TestDeepenToT4Converter:
    @pytest.fixture(scope="function")
    def converter_for_test(self):
        # TODO(yukke42): test with files
        input_base = ""
        output_base = ""
        input_anno_file = ""
        dataset_corresponding = {}
        overwrite_mode = False

        return DeepenToT4Converter(
            input_base=input_base,
            output_base=output_base,
            input_anno_file=input_anno_file,
            dataset_corresponding=dataset_corresponding,
            overwrite_mode=overwrite_mode,
            description={},
            input_bag_base="",
            topic_list=[],
            ignore_interpolate_label=False,
        )

    @pytest.fixture(scope="function")
    def converter_for_interpolate_test(self):
        input_base = ""
        output_base = ""
        input_anno_file = ""
        dataset_corresponding = {}
        overwrite_mode = False

        return DeepenToT4Converter(
            input_base=input_base,
            output_base=output_base,
            input_anno_file=input_anno_file,
            dataset_corresponding=dataset_corresponding,
            overwrite_mode=overwrite_mode,
            description={},
            input_bag_base="",
            topic_list=[],
            ignore_interpolate_label=True,
        )

    def test_convert(self):
        # TODO(yukke42): impl test_convert
        pass

    def test__format_deepen_annotation(
        self,
        converter_for_test: DeepenToT4Converter,
        deepen_anno_list: List[Dict[str, Any]],
    ):
        scenes_anno_dict = converter_for_test._format_deepen_annotation(deepen_anno_list)

        assert len(scenes_anno_dict) == 1
        assert len(scenes_anno_dict["dataset_xxx"]) == 2
        assert len(scenes_anno_dict["dataset_xxx"][0]) == 1
        assert len(scenes_anno_dict["dataset_xxx"][1]) == 3
        assert isinstance(scenes_anno_dict, dict)
        assert all(isinstance(dataset_id, str) for dataset_id in scenes_anno_dict.keys())
        assert all(
            isinstance(frame_index, int) for frame_index in scenes_anno_dict["dataset_xxx"].keys()
        )
        assert all(
            isinstance(frame_index, list)
            for frame_index in scenes_anno_dict["dataset_xxx"].values()
        )

    def test__format_deepen_annotation_ignore_interpolate(
        self,
        converter_for_interpolate_test: DeepenToT4Converter,
        deepen_anno_list: List[Dict[str, Any]],
    ):
        scenes_anno_dict = converter_for_interpolate_test._format_deepen_annotation(
            deepen_anno_list
        )

        assert len(scenes_anno_dict["dataset_xxx"][0]) == 1
        assert len(scenes_anno_dict["dataset_xxx"][1]) == 2
