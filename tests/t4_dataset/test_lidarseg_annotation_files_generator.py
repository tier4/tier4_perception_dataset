import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME
from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator


def _default_lidarseg(default_tmp_dir: str) -> Dict[int, List[Dict[str, Any]]]:
    """Create a default lidarseg annotation dictionary for testing."""
    default_lidarseg = {
        0: [
            {
                "paint_categories": ["category1", "category2"],
                "lidarseg_anno_file": f"{default_tmp_dir}/lidarseg_anno_file_0.pcd.bin",
                "total_lidar_points": 100,
            },
        ],
        1: [
            {
                "paint_categories": ["category1", "category2"],
                "lidarseg_anno_file": f"{default_tmp_dir}/lidarseg_anno_file_1.pcd.bin",
                "total_lidar_points": 200,
            }
        ],
        2: [
            {
                "paint_categories": ["category1", "category2"],
                "lidarseg_anno_file": f"{default_tmp_dir}/lidarseg_anno_file_2.pcd.bin",
                "total_lidar_points": 300,
            }
        ],
    }
    # Create duymmy lidarseg annotation files
    for scene_anno in default_lidarseg.values():
        for anno in scene_anno:
            with open(anno["lidarseg_anno_file"], "wb") as f:
                f.write(b"0")

    return default_lidarseg


class DummyNuImages:
    """Dummy clas for nuimages."""

    def __init__(
        self,
        version: str = "dummy_version",
        dataroot: str = "dummy_data_root",
        verbose: bool = False,
    ):
        self.version = version
        self.dummy_data_root = dataroot
        self.verbose = verbose


class DummyNuscenes:
    """Dummy nuscenes class for unit test."""

    def __init__(
        self,
        version: str = "dummy_version",
        dataroot: str = "dummy_data_root",
        verbose: bool = False,
    ):
        self.version = version
        self.dummy_data_root = dataroot
        self.verbose = verbose
        self.sample_data = [
            {"token": "0", "sample_token": "0", "filename": "data/LIDAR_CONCAT/00000.pcd.bin"},
            {"token": "1", "sample_token": "1", "filename": "data/LIDAR_CONCAT/00001.pcd.bin"},
            {"token": "2", "sample_token": "2", "filename": "data/LIDAR_CONCAT/00002.pcd.bin"},
            {"token": "3", "sample_token": "3", "filename": "data/LIDAR_CONCAT/00003.pcd.bin"},
            {"token": "4", "sample_token": "4", "filename": "data/LIDAR_CONCAT/00004.pcd.bin"},
            {"token": "5", "sample_token": "5", "filename": "data/LIDAR_CONCAT/00005.pcd.bin"},
        ]
        self.sample = [{"token": "0", "data": "data/LIDAR_CONCAT/00003.pcd.bin"}]


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
            "with_lidar": True,
            "with_lidarseg": True,
        }
        return AnnotationFilesGenerator(description=description)

    def test_convert_one_scene(
        self,
        instance_for_test: AnnotationFilesGenerator,
    ) -> None:
        """Test running convert_one_scene function."""
        with patch(
            "perception_dataset.t4_dataset.annotation_files_generator.NuScenes", DummyNuscenes
        ), patch(
            "perception_dataset.t4_dataset.annotation_files_generator.NuImages", DummyNuImages
        ):
            with tempfile.TemporaryDirectory() as dir_name:
                anno_dir = os.path.join(dir_name, "t4_format")
                anno_path = Path(anno_dir)
                anno_path_dir = anno_path / "annotation"
                anno_path_dir.mkdir(parents=True, exist_ok=True)

                instance_for_test.convert_one_scene(
                    input_dir=dir_name,
                    output_dir=anno_dir,
                    scene_anno_dict=_default_lidarseg(default_tmp_dir=dir_name),
                    dataset_name="dummy_dataset_name",
                )
                assert os.path.exists(os.path.join(anno_path_dir, "lidarseg.json"))
                lidarseg_path = (
                    Path(anno_dir)
                    / T4_FORMAT_DIRECTORY_NAME.LIDARSEG_ANNO_FOLDER.value
                    / "annotation"
                )
                lidarseg_files = [entry for entry in lidarseg_path.iterdir() if entry.is_file()]
                assert len(lidarseg_files) == 3

    def test__convert_lidarseg_scene_annotations(
        self,
        instance_for_test: AnnotationFilesGenerator,
    ) -> None:
        """Test running _convert_lidarseg_scene_annotations function."""
        with tempfile.TemporaryDirectory() as dir_name:
            dummy_nuscenes = DummyNuscenes()
            anno_dir = os.path.join(dir_name, "annotation")
            instance_for_test._convert_lidarseg_scene_annotations(
                nusc=dummy_nuscenes,
                scene_anno_dict=_default_lidarseg(default_tmp_dir=dir_name),
                lidar_sensor_channel="LIDAR_CONCAT",
                anno_dir=anno_dir,
            )
            assert os.path.exists(os.path.join(anno_dir, "lidarseg.json"))
            lidarseg_path = (
                Path(dir_name) / T4_FORMAT_DIRECTORY_NAME.LIDARSEG_ANNO_FOLDER.value / "annotation"
            )
            lidarseg_files = [entry for entry in lidarseg_path.iterdir() if entry.is_file()]
            assert len(lidarseg_files) == 3
