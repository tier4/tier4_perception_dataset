import glob
import os
import os.path as osp
from pathlib import Path
import shutil
import time

from t4_devkit import Tier4

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class NonAnnotatedT4TlrToDeepenConverter(AbstractConverter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
    ):
        super().__init__(input_base, output_base)

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
        nusc = Tier4(data_root=input_dir, verbose=False)

        logger.info(f"Converting {input_dir} to {output_dir}")
        for sample in nusc.sample:
            for sample_data_token in sample.data.values():
                # Note: This conversion tool will convert all camera data included in the t4dataset
                sample_data = nusc.get("sample_data", sample_data_token)
                original_filename = sample_data.filename
                input_path: Path = Path(input_dir) / original_filename
                output_path: Path = Path(output_dir) / original_filename.replace("/", "_")
                shutil.copy(input_path, output_path)

        logger.info(f"Done Conversion: {input_dir} to {output_dir}")
