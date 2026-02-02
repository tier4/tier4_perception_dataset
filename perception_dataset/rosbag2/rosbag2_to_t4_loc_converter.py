import copy
import os
import os.path as osp
import shutil
from typing import Dict, List

import builtin_interfaces.msg
from t4_devkit.schema.tables import (
    CalibratedSensor,
    EgoPose,
    Log,
    Map,
    Sample,
    SampleData,
    Scene,
    Sensor,
)

from perception_dataset.constants import (
    EXTENSION_ENUM,
    SENSOR_MODALITY_ENUM,
    T4_FORMAT_DIRECTORY_NAME,
)
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import SensorMode
from perception_dataset.rosbag2.rosbag2_to_t4_converter import (
    Rosbag2ToT4Converter,
    _Rosbag2ToT4Converter,
)
from perception_dataset.t4_dataset.table_handler import TableHandler
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.misc import get_sample_data_filename
import perception_dataset.utils.rosbag2 as rosbag2_utils

logger = configure_logger(modname=__name__)


class Rosbag2ToT4LocConverter(Rosbag2ToT4Converter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params)

    def convert(self):
        bag_dirs: List[str] = self._search_bag_dirs()

        if not self._overwrite_mode:
            dir_exist: bool = False
            for bag_dir in bag_dirs[:]:  # copy to avoid modifying list while iterating
                bag_name: str = osp.basename(bag_dir)

                output_dir = osp.join(self._output_base, bag_name)
                if osp.exists(output_dir):
                    logger.error(f"{output_dir} already exists.")
                    dir_exist = True
                    bag_dirs.remove(bag_dir)
            if dir_exist and len(bag_dirs) == 0:
                logger.error(f"{output_dir} already exists.")
                raise ValueError("If you want to overwrite files, use --overwrite option.")

        for bag_dir in sorted(bag_dirs):
            logger.info(f"Start converting {bag_dir} to T4 format.")
            self._params.input_bag_path = bag_dir
            try:
                self._convert_bag(bag_dir)
            except Exception as e:
                logger.error(f"Error occurred during conversion: {e}")
                continue
            logger.info(f"Conversion of {bag_dir} is completed")
            print(
                "--------------------------------------------------------------------------------------------------------------------------"
            )

    def _convert_bag(self, bag_dir: str):
        try:
            params = copy.deepcopy(self._params)
            params.input_bag_path = bag_dir
            converter = _Rosbag2ToT4LocConverter(params)
            converter.convert()
        except Exception:
            logger.exception(f"{bag_dir} failed with exception")
            raise


class _Rosbag2ToT4LocConverter(_Rosbag2ToT4Converter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        params.world_frame_id = "base_link"
        super().__init__(params)
        self._topic_list = params.topic_list

        # overwrite sensors
        self._sensor_mode = SensorMode.NO_SENSOR
        self._sensor_channel = "LIDAR_CONCAT"

        # overwrite and re-make initial directories
        shutil.rmtree(osp.join(self._output_base, self._bag_name), ignore_errors=True)

        self._output_scene_dir = osp.join(self._output_base, self._bag_name)
        if params.make_t4_dataset_dir:
            self._output_scene_dir = osp.join(self._output_scene_dir, "t4_dataset")
        self._output_anno_dir = osp.join(
            self._output_scene_dir, T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value
        )
        self._output_data_dir = osp.join(
            self._output_scene_dir, T4_FORMAT_DIRECTORY_NAME.DATA.value
        )
        self._make_directories()
        os.makedirs(osp.join(self._output_data_dir, self._sensor_channel), exist_ok=True)

    def _init_tables(self):
        # vehicle
        self._log_table = TableHandler(Log)
        self._map_table = TableHandler(Map)
        self._sensor_table = TableHandler(Sensor)
        for enum in self._sensor_enums:
            self._sensor_table.insert_into_table(
                channel=enum.value["channel"], modality=enum.value["modality"]
            )
        self._calibrated_sensor_table = TableHandler(CalibratedSensor)
        # extraction
        self._scene_table = TableHandler(Scene)
        self._sample_table = TableHandler(Sample)
        self._sample_data_table = TableHandler(SampleData)
        self._ego_pose_table = TableHandler(EgoPose)

    def convert(self):
        self._convert()
        self._save_tables()
        self._save_config()
        self._annotation_files_generator.save_tables(self._output_anno_dir)

        self._make_input_bag()

    def _make_input_bag(self):
        output_bag_dir: str = osp.join(self._output_scene_dir, "input_bag")
        shutil.copytree(self._input_bag, output_bag_dir)

    def _convert(self) -> None:
        """
        1. init tables
        2. fill dummy sensor data
        3. convert simulation data to annotation
        """
        sensor_channel_to_sample_data_token_list: Dict[str, List[str]] = {}

        self._init_tables()
        scene_token = self._convert_static_data()
        self._convert_dummy_pointcloud(sensor_channel_to_sample_data_token_list, scene_token)
        self._set_scene_data()
        self._connect_sample_in_scene()
        self._connect_sample_data_in_scene(sensor_channel_to_sample_data_token_list)
        self._add_scene_description(self._scene_description)

    def _convert_dummy_pointcloud(
        self,
        sensor_channel_to_sample_data_token_list: Dict[str, List[str]],
        scene_token: str,
    ):
        # Convert dummy pointcloud data to sample_data and sample
        start_timestamp = self._calc_start_timestamp()
        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        nusc_timestamp = rosbag2_utils.stamp_to_nusc_timestamp(start_time_in_time)
        lidar_sensor_channel = self._sensor_channel

        logger.info(f"set start_timestamp to {start_timestamp}")
        calibrated_sensor_token = self._generate_dummy_sensor(start_time_in_time)
        ego_pose_token = self._generate_ego_pose(start_time_in_time)
        sample_token = self._sample_table.insert_into_table(
            timestamp=nusc_timestamp, scene_token=scene_token, next="", prev=""
        )

        fileformat = EXTENSION_ENUM.PCDBIN.value[1:]
        filename = get_sample_data_filename(lidar_sensor_channel, 0, fileformat)
        sample_data_token = self._sample_data_table.insert_into_table(
            sample_token=sample_token,
            ego_pose_token=ego_pose_token,
            calibrated_sensor_token=calibrated_sensor_token,
            filename=filename,
            fileformat=fileformat,
            timestamp=nusc_timestamp,
            is_key_frame=True,
            width=0,
            height=0,
            next="",
            prev="",
        )
        sample_data_record: SampleData = self._sample_data_table.get_record_from_token(
            sample_data_token
        )
        points_arr = rosbag2_utils.pointcloud_msg_to_numpy(None)
        points_arr.tofile(osp.join(self._output_scene_dir, sample_data_record.filename))

        sensor_channel_to_sample_data_token_list[lidar_sensor_channel] = [sample_data_token]

    def _generate_dummy_sensor(self, start_timestamp: builtin_interfaces.msg.Time) -> str:
        calibrated_sensor_token = str()
        if self._sensor_mode == SensorMode.NO_SENSOR:

            channel = self._sensor_channel
            modality = SENSOR_MODALITY_ENUM.LIDAR.value

            sensor_token = self._sensor_table.insert_into_table(
                channel=channel,
                modality=modality,
            )

            frame_id = "base_link"
            transform_stamped = self._bag_reader.get_transform_stamped(
                target_frame=self._calibrated_sensor_target_frame,
                source_frame=frame_id,
                stamp=start_timestamp,
            )
            translation = (
                transform_stamped.transform.translation.x,
                transform_stamped.transform.translation.y,
                transform_stamped.transform.translation.z,
            )
            rotation = (
                transform_stamped.transform.rotation.w,
                transform_stamped.transform.rotation.x,
                transform_stamped.transform.rotation.y,
                transform_stamped.transform.rotation.z,
            )

            calibrated_sensor_token = self._calibrated_sensor_table.insert_into_table(
                sensor_token=sensor_token,
                translation=translation,
                rotation=rotation,
                camera_intrinsic=[],
                camera_distortion=[],
            )
            return calibrated_sensor_token
        else:
            raise ValueError("Invalid sensor mode")
