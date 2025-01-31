from typing import Any, Dict, List

import pytest

from perception_dataset.deepen.deepen_to_t4_converter import DeepenToT4Converter


@pytest.fixture(scope="function")
def lidarseg_deepen_anno_list() -> List[Dict[str, Any]]:
    # Note: only values required for t4dataset
    return [
        {
            "dataset_id": "dummy_dataset_id",
            "file_id": "0.pcd",
            "label_type": "3d_point",
            "label_id": "none",
            "label_category_id": "none",
            "total_lidar_points": 173430,
            "sensor_id": "lidar",
            "stage_id": "QA",
            "paint_categories": ["car", "wall"],
            "lidarseg_anno_file": "lidar_seg/dummy_dataset_id_0.pcd.bin",
        },
        {
            "dataset_id": "dummy_dataset_id",
            "file_id": "10.pcd",
            "label_type": "3d_point",
            "label_id": "none",
            "label_category_id": "none",
            "total_lidar_points": 173450,
            "sensor_id": "lidar",
            "stage_id": "QA",
            "paint_categories": ["car", "wall"],
            "lidarseg_anno_file": "lidar_seg/dummy_dataset_id_10.pcd.bin",
        },
        {
            "dataset_id": "dummy_dataset_id",
            "file_id": "20.pcd",
            "label_type": "3d_point",
            "label_id": "none",
            "label_category_id": "none",
            "total_lidar_points": 176709,
            "sensor_id": "lidar",
            "stage_id": "QA",
            "paint_categories": ["car", "wall"],
            "lidarseg_anno_file": "lidar_seg/dummy_dataset_id_20.pcd.bin",
        },
    ]


class TestDeepenToT4Converter:
    @pytest.fixture(scope="function")
    def converter_for_test(self):
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

    def test__format_point_3d_annotations(
        self,
        converter_for_test: DeepenToT4Converter,
        lidarseg_deepen_anno_list: List[Dict[str, Any]],
    ):
        """Test _format_point_3d_annotations."""
        scenes_anno_dict = converter_for_test._format_deepen_annotation(lidarseg_deepen_anno_list)

        assert len(scenes_anno_dict) == 1
        assert len(scenes_anno_dict["dummy_dataset_id"][0]) == 1
        assert len(scenes_anno_dict["dummy_dataset_id"][10]) == 1
        assert len(scenes_anno_dict["dummy_dataset_id"][20]) == 1
        assert isinstance(scenes_anno_dict, dict)
        assert all(isinstance(dataset_id, str) for dataset_id in scenes_anno_dict.keys())
        assert all(
            isinstance(frame_index, int)
            for frame_index in scenes_anno_dict["dummy_dataset_id"].keys()
        )
        assert all(
            isinstance(frame_index, list)
            for frame_index in scenes_anno_dict["dummy_dataset_id"].values()
        )
