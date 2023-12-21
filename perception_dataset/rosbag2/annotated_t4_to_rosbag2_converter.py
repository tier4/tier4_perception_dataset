import copy
import glob
import os
import os.path as osp
from pathlib import Path
import time
from typing import Any, Dict, List

from nuimages import NuImages
import numpy as np
from nuscenes.nuscenes import NuScenes


class AnnotatedT4ToRosbag2Converter(AbstractConverter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params.input_base, params.output_base)

        self._params = params
        self._overwrite_mode = params.overwrite_mode

    def convert(self):
        start_time = time.time()

        for scene_dir in glob.glob(osp.join(self._input_base, "*")):
            if not osp.isdir(scene_dir):
                continue
            t4_dataset_path = osp.join(scene_dir, "t4_dataset")
            if not osp.isdir(t4_dataset_path):
                t4_dataset_path = scene_dir

            scene_name = osp.basename(scene_dir)
            self._convert_one_scene(
                t4_dataset_path,
                scene_name,
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.1f} [sec]")

    def _convert_one_scene(self, input_dir: str, scene_name: str):
        output_dir = self._output_base
        os.makedirs(output_dir, exist_ok=True)
        nusc = NuScenes(version="annotation", dataroot=input_dir, verbose=False)
        nuim = NuImages(version="annotation", dataroot=input_dir, verbose=True, lazy=True)

        logger.info(f"Converting {input_dir} to {output_dir}")
        output_label: List = []
