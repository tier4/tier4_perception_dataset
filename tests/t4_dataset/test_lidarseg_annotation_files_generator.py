import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pytest
from t4_devkit.schema import Sample, SampleData

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME
from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator


def _create_dummy_pcd_bin(filepath: str, num_points: int) -> None:
    """Create a dummy PCD bin file with float32 data of shape (5, num_points)."""
    # Create point cloud data with 5 channels: x, y, z, intensity, ring
    points = np.random.randn(5, num_points).astype(np.float32)
    points.tofile(filepath)


def _create_dummy_annotation_bin(filepath: str, num_points: int) -> None:
    """Create a dummy annotation bin file with uint8 labels of shape (num_points,)."""
    # Create random labels (0-10 as category indices)
    labels = np.random.randint(0, 10, num_points, dtype=np.uint8)
    labels.tofile(filepath)


def _default_lidarseg(default_tmp_dir: str) -> Dict[int, List[Dict[str, Any]]]:
    """Create a default lidarseg annotation dictionary for testing."""
    # Create data directory structure
    data_dir = Path(default_tmp_dir) / "data" / "LIDAR_CONCAT"
    data_dir.mkdir(parents=True, exist_ok=True)

    default_lidarseg = {
        0: [
            {
                "paint_categories": ["category1", "category2"],
                "lidarseg_anno_file": f"{default_tmp_dir}/lidarseg_anno_file_0.bin",
                "total_lidar_points": 100,
            },
        ],
        1: [
            {
                "paint_categories": ["category1", "category2"],
                "lidarseg_anno_file": f"{default_tmp_dir}/lidarseg_anno_file_1.bin",
                "total_lidar_points": 200,
            }
        ],
        2: [
            {
                "paint_categories": ["category1", "category2"],
                "lidarseg_anno_file": f"{default_tmp_dir}/lidarseg_anno_file_2.bin",
                "total_lidar_points": 300,
            }
        ],
    }

    # Create dummy PCD and annotation files
    for frame_idx, scene_anno in default_lidarseg.items():
        for anno in scene_anno:
            num_points = anno["total_lidar_points"]

            # Create dummy PCD point cloud file
            pcd_filepath = data_dir / f"0000{frame_idx}.pcd.bin"
            _create_dummy_pcd_bin(str(pcd_filepath), num_points)

            # Create dummy annotation file
            _create_dummy_annotation_bin(anno["lidarseg_anno_file"], num_points)

    return default_lidarseg


class DummyTier4:
    """Dummy Tier4 class for unit test."""

    def __init__(
        self,
        data_root: str = "dummy_data_root",
        verbose: bool = False,
    ):
        self.dummy_data_root = data_root
        self.verbose = verbose
        self.sample_data = [
            SampleData(
                token=str(i),
                sample_token=str(i),
                ego_pose_token="0",
                calibrated_sensor_token="0",
                filename=f"data/LIDAR_CONCAT/0000{i}.pcd.bin",
                fileformat="pcd.bin",
                width=0,
                height=0,
                timestamp=0,
                is_key_frame=True,
                next="",
                prev="",
                is_valid=True,
            )
            for i in range(6)
        ]
        self.sample = [Sample(timestamp=0, token="0", scene_token="0", next="", prev="")]

    def get(self, table_type: str, token: str) -> Any:
        if table_type == "sample_data":
            return self.sample_data[int(token)]
        elif table_type == "sample":
            return self.sample[0]
        else:
            raise ValueError(f"Unsupported table type: {table_type}")


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
        with patch("perception_dataset.t4_dataset.annotation_files_generator.Tier4", DummyTier4):
            with tempfile.TemporaryDirectory() as dir_name:
                anno_dir = os.path.join(dir_name, "t4_format")
                anno_path = Path(anno_dir)
                anno_path_dir = anno_path / "annotation"
                anno_path_dir.mkdir(parents=True, exist_ok=True)
                instance_for_test.convert_one_scene(
                    input_dir=dir_name,
                    output_dir=anno_dir,
                    scene_anno_dict=_default_lidarseg(default_tmp_dir=anno_dir),
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
            dummy_t4 = DummyTier4()
            anno_dir = os.path.join(dir_name, "annotation")
            instance_for_test._convert_lidarseg_scene_annotations(
                t4_dataset=dummy_t4,
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
