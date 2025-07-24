from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List

from perception_dataset.deepen.deepen_annotation import DeepenAnnotation, LabelType

__all__ = [
    "DeepenSegmentationPainting3DAnnotations",
    "DeepenSegmentationPainting3DScene",
    "DeepenSegmentationPainting3DDataset",
]


class DeepenSegmentationPainting3DScene(DeepenAnnotation):
    """
    A class to save a 3D segmentation annotation for a scene annotated by painting.
    :param paint_categories: List of categories in lidarseg.
    :param lidarseg_anno_file: Binary annotation file for the lidarseg.
    :param total_lidar_points: Total number of lidar pointclouds.
    """

    def __init__(
        self,
        paint_categories: List[str],
        lidarseg_anno_file: str,
        total_lidar_points: int,
        **kwargs: Any,
    ):
        super().__init__(lidarseg_anno_file=lidarseg_anno_file, **kwargs)
        self.paint_categories = paint_categories
        self.lidarseg_anno_file = lidarseg_anno_file
        self.total_lidar_points = total_lidar_points

        assert self.total_lidar_points > 0, "Lidar pointclouds must be more than 0!"
        assert self.lidarseg_anno_file != "", "Must provide a lidarseg annotation file!"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass instance to a dictionary."""
        data = super().to_dict()
        data["paint_categories"] = self.paint_categories
        data["total_lidar_points"] = self.total_lidar_points
        return data


@dataclass
class DeepenSegmentationPainting3DDataset:
    """Lidar segmentation records for a dataset with a dict of scenes/frames."""

    dataset_id: str
    deepen_segmentation_scenes: Dict[
        int, DeepenSegmentationPainting3DScene
    ]  # {scene_id: DeepenSegmentationPainting3DScene}

    def add_record_to_dataset(
        self, segmentation_painting_3d_scene: DeepenSegmentationPainting3DScene
    ) -> None:
        """Add a DeepenSegmentationPainting3DScene to the dataset records."""
        filename = segmentation_painting_3d_scene.file_id.split(".")[0]
        scene_id = int(re.sub(r"\D", "", filename[-6:]))
        self.deepen_segmentation_scenes[scene_id] = segmentation_painting_3d_scene

    def format_scene_annotations(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Format scene annotations to
        {
                id:
                [
                        {
                                "paint_categories": ["car", "wall", ...],
                                "lidarseg_anno_file": "lidar_seg/DOnC2vK05ojPr7qiqCsk2Ee7_0.bin",
                                ...
                        }
                ]
        }
        """
        return {
            scene_id: [scene_annotations.to_dict()]
            for scene_id, scene_annotations in self.deepen_segmentation_scenes.items()
        }


@dataclass(frozen=True)
class DeepenSegmentationPainting3DAnnotations:
    """Lidar segmentation records for all datasets."""

    deepen_segmentation_datasets: Dict[
        str, DeepenSegmentationPainting3DDataset
    ]  # {dataset_id: DeepenSegmentationPainting3DDataset}

    @classmethod
    def from_file(
        cls,
        ann_file: str,
    ) -> DeepenSegmentationPainting3DAnnotations:
        """Return DeepenSegmentationPainting3DDataset from files.

        Args:
                ann_file (str): Annotation files path in json. The format is:
                [
                        {
                                "dataset_id": "dummy_dataset_id",
                                "file_id": "0.pcd",
                                "label_type": "3d_point",
                                "label_id": "none:-1,		# Keep it for consistency with downstream tasks
                                "label_category_id": "none",	# Keep it for consistency with downstream tasks
                                "total_lidar_points": 173430,
                                "sensor_id": "lidar",
                                "stage_id": "QA",
                                "paint_categories": ["car", "wall", ...],
                                "lidarseg_anno_file": "lidarseg/dummy_dataset_id_0..pcd.bin"
                        },
                        ...
                ]
                data_root (str): Root directory of the T4 dataset.
                camera2index (Dict[str, int]): Name mapping from camera name to camera index.
                dataset_corresponding (Dict[str, str]): Key-value mapping of T4 dataset name and Deepen ID.
                as_dict (bool, optional): Whether to output objects as dict or its instance.
                        Defaults to True.

        Returns:
                        List[DeepenSegmentationPainting2D]: List of converted `DeepenSegmentationPainting2D`s.
        """
        with open(ann_file, "r") as f:
            lidarseg_ann_info = json.load(f)

        data_root = os.path.abspath(Path(ann_file).parent)
        print(data_root)
        lidarseg_paint_3d_datasets: Dict[str, DeepenSegmentationPainting3DDataset] = {}

        for lidarseg_ann in lidarseg_ann_info:
            dataset_id = lidarseg_ann["dataset_id"]
            if dataset_id not in lidarseg_paint_3d_datasets:
                lidarseg_paint_3d_datasets[dataset_id] = DeepenSegmentationPainting3DDataset(
                    dataset_id=dataset_id, deepen_segmentation_scenes={}
                )

            lidarseg_paint_3d = DeepenSegmentationPainting3DScene(
                dataset_id=lidarseg_ann["dataset_id"],
                file_id=lidarseg_ann["file_id"],
                label_type=LabelType.POINT_3D.value,
                label_id=lidarseg_ann["label_id"],
                label_category_id=lidarseg_ann["label_category_id"],
                total_lidar_points=lidarseg_ann["total_lidar_points"],
                sensor_id=lidarseg_ann["sensor_id"],
                lidarseg_anno_file=f"{data_root}/{lidarseg_ann['lidarseg_anno_file']}",
                paint_categories=lidarseg_ann["paint_categories"],
            )

            lidarseg_paint_3d_dataset = lidarseg_paint_3d_datasets[dataset_id]
            lidarseg_paint_3d_dataset.add_record_to_dataset(
                segmentation_painting_3d_scene=lidarseg_paint_3d
            )

        return cls(deepen_segmentation_datasets=lidarseg_paint_3d_datasets)

    def format_deepen_annotations(self) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """
        Convert to {
                dataset_id: {
                        scene_id/frame_index:
                        [
                                {
                                        "paint_categories": ["car", "wall", ...],
                                        "lidarseg_anno_file": "lidar_seg/dummy_dataset_id_0.pcd.bin",
                                        ...
                                }
                        ]
                }
        }
        """
        return {
            dataset_id: paint_3d_annotations.format_scene_annotations()
            for dataset_id, paint_3d_annotations in self.deepen_segmentation_datasets.items()
        }
