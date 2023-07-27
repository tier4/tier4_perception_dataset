from typing import Any, Dict

import pytest

from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator


def _default_three_d_bbox() -> Dict[str, Dict[str, float]]:
    box = {
        "translation": {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
        },
        "velocity": {
            "x": None,
            "y": None,
            "z": None,
        },
        "acceleration": {
            "x": None,
            "y": None,
            "z": None,
        },
        "size": {
            "width": 10.0,
            "length": 20.0,
            "height": 30.0,
        },
        "rotation": {
            "w": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        },
    }

    return box


# Note: test case1, case2 use the same scene_anno_dict
class TestAnnotationFilesGenerator:
    @pytest.fixture(scope="function")
    def instance_for_test(self):
        description = {
            "visibility": {
                "full": "No occlusion of the object.",
                "most": "Object is occluded, but by less than 50%.",
                "partial": "The object is occluded by more than 50% (but not completely).",
                "none": "The object is 90-100% occluded and no points/pixels are visible in the label.",
            },
            "camera_index": {
                "CAM_FRONT": 0,
                "CAM_FRONT_RIGHT": 1,
                "CAM_BACK_RIGHT": 2,
                "CAM_BACK": 3,
                "CAM_BACK_LEFT": 4,
                "CAM_FRONT_LEFT": 5,
            },
        }
        return AnnotationFilesGenerator(description=description)

    @pytest.mark.parametrize(
        "scene_anno_dict, frame_index_to_sample_token, expected_values",
        [
            # test case 1 (3 sample, 1 instance)
            (
                # scene_anno_dict
                {
                    0: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                    1: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                    2: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                },
                # frame_index_to_sample_token
                {
                    0: "0_xxx",
                    1: "1_xxx",
                    2: "2_xxx",
                },
                # expected values
                {
                    "attribute": {
                        "len": 1,
                        "names": ["attr_xxx"],
                    },
                    "category": {
                        "len": 1,
                        "names": ["name_xxx"],
                    },
                    "sample_annotation": {
                        "len": 3,
                    },
                    "instance": {
                        "len": 1,
                        "nbr_annotations": [3],
                    },
                },
            ),
            # test case 2 (3 sample, 2 instance)
            (
                # scene_anno_dict
                {
                    0: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                        {
                            "category_name": "name_yyy",
                            "instance_id": "id_yyy",
                            "attribute_names": ["attr_yyy", "attr_zzz"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                    1: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                },
                # frame_index_to_sample_token
                {
                    0: "0_xxx",
                    1: "1_xxx",
                },
                # expected values
                {
                    "attribute": {
                        "len": 3,
                        "names": ["attr_xxx", "attr_yyy", "attr_zzz"],
                    },
                    "category": {
                        "len": 2,
                        "names": ["name_xxx", "name_yyy"],
                    },
                    "sample_annotation": {
                        "len": 3,
                    },
                    "instance": {
                        "len": 2,
                    },
                },
            ),
        ],
    )
    def test__convert_to_t4_format(
        self,
        mocker,
        scope_function,
        scene_anno_dict: Dict[int, Dict[str, Any]],
        frame_index_to_sample_token: Dict[int, str],
        expected_values: Dict[str, Any],
        instance_for_test: AnnotationFilesGenerator,
    ):
        instance_for_test._convert_to_t4_format(
            scene_anno_dict=scene_anno_dict,
            frame_index_to_sample_token=frame_index_to_sample_token,
            dataset_name="test_dataset",
            # for 2d annotations but not used in this test case
            frame_index_to_sample_data_token=[],
            mask=[],
        )

        # check encapsulated value
        attr_data = instance_for_test._attribute_table.to_data()
        assert len(attr_data) == expected_values["attribute"]["len"]
        assert [d["name"] for d in attr_data] == expected_values["attribute"]["names"]

        category_data = instance_for_test._category_table.to_data()
        assert len(category_data) == expected_values["category"]["len"]
        assert [d["name"] for d in category_data] == expected_values["category"]["names"]

        annotation_data = instance_for_test._sample_annotation_table.to_data()
        assert len(annotation_data) == expected_values["sample_annotation"]["len"]

        instance_data = instance_for_test._instance_table.to_data()
        assert len(instance_data) == expected_values["instance"]["len"]

    @pytest.mark.parametrize(
        "scene_anno_dict, frame_index_to_sample_token, expected_values",
        [
            # test case 1 (3 sample, 1 instance)
            (
                # scene_anno_dict
                {
                    0: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                    1: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                    2: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                },
                # frame_index_to_sample_token
                {
                    0: "0_xxx",
                    1: "1_xxx",
                    2: "2_xxx",
                },
                # expected values
                {
                    "sample_annotation": {
                        "connection_indices_list": [
                            [0, 1, 2],
                        ],
                    },
                    "instance": {
                        "nbr_annotations": [3],
                        "first_token_indices": [0],
                        "last_token_indices": [2],
                    },
                },
            ),
            # test case 2 (3 sample, 2 instance)
            (
                # scene_anno_dict
                {
                    0: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                        {
                            "category_name": "name_yyy",
                            "instance_id": "id_yyy",
                            "attribute_names": ["attr_yyy", "attr_zzz"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                    1: [
                        {
                            "category_name": "name_xxx",
                            "instance_id": "id_xxx",
                            "attribute_names": ["attr_xxx"],
                            "three_d_bbox": _default_three_d_bbox(),
                            "num_lidar_pts": 0,
                            "num_radar_pts": 0,
                        },
                    ],
                },
                # frame_index_to_sample_token
                {
                    0: "0_xxx",
                    1: "1_xxx",
                },
                # expected values
                {
                    "sample_annotation": {
                        "connection_indices_list": [
                            [0, 2],
                            [1],
                        ],
                    },
                    "instance": {
                        "nbr_annotations": [2, 1],
                        "first_token_indices": [0, 1],
                        "last_token_indices": [2, 1],
                    },
                },
            ),
        ],
    )
    def test__connect_annotations_in_scene(
        self,
        mocker,
        scope_function,
        scene_anno_dict: Dict[int, Dict[str, Any]],
        frame_index_to_sample_token: Dict[int, str],
        expected_values: Dict[str, Any],
        instance_for_test: AnnotationFilesGenerator,
    ):
        instance_for_test._convert_to_t4_format(
            scene_anno_dict=scene_anno_dict,
            frame_index_to_sample_token=frame_index_to_sample_token,
            dataset_name="test_dataset",
            # for 2d annotations but not used in this test case
            frame_index_to_sample_data_token=[],
            mask=[],
        )
        instance_for_test._connect_annotations_in_scene()

        # check encapsulated value
        anno_data = instance_for_test._sample_annotation_table.to_data()
        for connect_indices in expected_values["sample_annotation"]["connection_indices_list"]:
            for ci in range(len(connect_indices)):
                sample_i = connect_indices[ci]
                if ci == 0:
                    # the first token
                    assert anno_data[sample_i]["prev"] == ""
                else:
                    prev_sample_i = connect_indices[ci - 1]
                    assert anno_data[sample_i]["prev"] == anno_data[prev_sample_i]["token"]

                if ci == len(connect_indices) - 1:
                    # the last token
                    assert anno_data[sample_i]["next"] == ""
                else:
                    next_sample_i = connect_indices[ci + 1]
                    assert anno_data[sample_i]["next"] == anno_data[next_sample_i]["token"]

        expected_first_tokens = [
            anno_data[i]["token"] for i in expected_values["instance"]["first_token_indices"]
        ]
        expected_last_tokens = [
            anno_data[i]["token"] for i in expected_values["instance"]["last_token_indices"]
        ]
        instance_data = instance_for_test._instance_table.to_data()
        assert [d["nbr_annotations"] for d in instance_data] == expected_values["instance"][
            "nbr_annotations"
        ]
        assert [d["first_annotation_token"] for d in instance_data] == expected_first_tokens
        assert [d["last_annotation_token"] for d in instance_data] == expected_last_tokens
