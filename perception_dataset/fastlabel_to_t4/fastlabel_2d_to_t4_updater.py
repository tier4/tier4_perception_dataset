from __future__ import annotations

import json
import os.path as osp
from pathlib import Path
import shutil
from typing import Dict

from perception_dataset.fastlabel_to_t4.fastlabel_2d_to_t4_converter import (
    FastLabel2dToT4Converter,
)
from perception_dataset.t4_dataset.annotation_files_updater import AnnotationFilesUpdater
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


# TODO: add support of 3D annotation format
class FastLabel2dToT4Updater(FastLabel2dToT4Converter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        input_anno_base: str,
        overwrite_mode: bool,
        description: Dict[str, Dict[str, str]],
        make_t4_dataset_dir: bool = True,
    ):
        super().__init__(
            input_base,
            output_base,
            input_anno_base,
            dataset_corresponding=None,
            overwrite_mode=overwrite_mode,
            description=description,
            input_bag_base=None,
            topic_list=None,
        )
        self._make_t4_dataset_dir = make_t4_dataset_dir

    def convert(self) -> None:
        t4_datasets = sorted([d.name for d in self._input_base.iterdir() if d.is_dir()])
        anno_jsons_dict = self._load_annotation_jsons(t4_datasets, "_CAM")
        fl_annotations = self._format_fastlabel_annotation(anno_jsons_dict)

        for t4dataset_name in t4_datasets:
            # Check if annotation exists
            if t4dataset_name not in fl_annotations.keys():
                continue

            # Check if input directory exists
            input_dir = self._input_base / t4dataset_name
            input_annotation_dir = input_dir / "annotation"
            if not osp.exists(input_annotation_dir):
                logger.warning(f"input_dir {input_dir} not exists.")
                continue

            # Check if output directory already exists
            output_dir = self._output_base / t4dataset_name
            if self._make_t4_dataset_dir:
                output_dir = output_dir / "t4_dataset"
            if self._input_bag_base is not None:
                input_bag_dir = Path(self._input_bag_base) / t4dataset_name

            if osp.exists(output_dir):
                logger.error(f"{output_dir} already exists.")
                is_dir_exist = True
            else:
                is_dir_exist = False

            if self._overwrite_mode or not is_dir_exist:
                # Remove existing output directory
                shutil.rmtree(output_dir, ignore_errors=True)
                # Copy input data to output directory
                self._copy_data(input_dir, output_dir)
                # Make rosbag
                if self._input_bag_base is not None and not osp.exists(
                    osp.join(output_dir, "input_bag")
                ):
                    self._find_start_end_time(input_dir)
                    self._make_rosbag(str(input_bag_dir), str(output_dir))
            else:
                raise ValueError("If you want to overwrite files, use --overwrite option.")

            # Start updating annotations
            annotation_files_updater = AnnotationFilesUpdater(
                description=self._description, surface_categories=self._surface_categories
            )
            annotation_files_updater.convert_one_scene(
                input_dir=input_dir,
                output_dir=output_dir,
                scene_anno_dict=fl_annotations[t4dataset_name],
                dataset_name=t4dataset_name,
            )
            logger.info(f"Finished updating annotations for {t4dataset_name}")
