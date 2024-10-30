import os.path as osp
import shutil
import time


from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class NonAnnotatedT4ToEmptyAnnotationT4Converter(AbstractConverter):
    """
    Convert non-annotated T4 dataset to empty annotated T4 dataset, where we assume an output_base contains the following contents:
    status.json
    annotations/
    data/

    And this converter will move rosbags from an input_base to the output_base.
    """
    def __init__(
        self,
        input_base: str,
        output_base: str,
    ):
        super().__init__(input_base, output_base)

    def convert(self):
        start_time = time.time()

        # Move rosbag files to input_rosbags
        self._move_rosbags()
        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.1f} [sec]")

    def _move_rosbags(self) -> None:
        """ Move rosbags from an input_base to ."""
        output_bag_dir: str = osp.join(self._output_base, "input_bag")
        self._copy_data(
            output_bag_dir
        )

    def _copy_data(self, output_dir: str) -> None:
        """ Copy data from the input_dir to the output_dir. """
        if self._input_base != output_dir:
            logger.info(f"Copying {self._input_base} to {output_dir} ... ")
            shutil.copytree(self._input_base, output_dir)
            logger.info("Done!")
