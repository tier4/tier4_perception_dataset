import glob
import os
import os.path as osp
import shutil
import time
from typing import Any, Dict
from pathlib import Path

from nuscenes.nuscenes import NuScenes

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import SENSOR_ENUM
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class NonAnnotatedT4TlrToDeepenConverter(AbstractConverter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        camera_sensors: list,
        workers_number: int = 32,
    ):
        super().__init__(input_base, output_base)
        self._camera_sensor_types = []
        self._workers_number = workers_number
        self._camera_sensor_types = [camera_sensor["channel"] for camera_sensor in camera_sensors]

    def convert(self):
        start_time = time.time()

        for scene_dir in glob.glob(osp.join(self._input_base, "*")):
            if not osp.isdir(scene_dir):
                continue

            out_dir = osp.join(self._output_base, osp.basename(scene_dir).replace(".", "-"))
            self._convert_one_scene(
                scene_dir,
                out_dir,
            )
            shutil.make_archive(f"{out_dir}", "zip", root_dir=out_dir)

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.1f} [sec]")

    def _convert_one_scene(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        nusc = NuScenes(version="annotation", dataroot=input_dir, verbose=False)

        logger.info(f"Converting {input_dir} to {output_dir}")

        for sample in nusc.sample:
            for sample_data_token in sample['data'].values():
                sample_data = nusc.get('sample_data', sample_data_token)
                if sample_data['channel'] not in self._camera_sensor_types:
                    continue
                
                original_filename = sample_data["filename"]
                input_path: Path = Path(input_dir) / original_filename
                output_path: Path = Path(output_dir) / original_filename.replace('/', '_')
                shutil.copy(input_path, output_path)

        logger.info(f"Done Conversion: {input_dir} to {output_dir}")
