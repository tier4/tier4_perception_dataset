import copy
from multiprocessing import Pool
import os
import os.path as osp
import shutil
from typing import Any, Dict, List

from perception_dataset.constants import SENSOR_ENUM, T4_FORMAT_DIRECTORY_NAME
from perception_dataset.ros2.tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from perception_dataset.rosbag2.converter_params import DataType, Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_converter import Rosbag2Converter
from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import SensorMode
from perception_dataset.rosbag2.rosbag2_to_t4_converter import (
    Rosbag2ToT4Converter,
    _Rosbag2ToT4Converter,
)
from perception_dataset.t4_dataset.table_handler import TableHandler
from perception_dataset.utils.logger import configure_logger
import perception_dataset.utils.rosbag2 as rosbag2_utils
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

from .autoware_msgs import parse_perception_objects

logger = configure_logger(modname=__name__)


class Rosbag2ToT4TrackingConverter(Rosbag2ToT4Converter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params)

    def convert(self):
        bag_dirs: List[str] = self._search_bag_dirs()

        if not self._overwrite_mode:
            # check if already exists
            exist_dir = False
            for bag_dir in bag_dirs:
                path_to_output = os.path.join(self._output_base, os.path.basename(bag_dir))
                if os.path.exists(path_to_output):
                    logger.error(f"{path_to_output} already exists.")
                    exist_dir = True
                    break
            if exist_dir:
                raise ValueError("Use --overwrite option to overwrite files.")
            else:
                logger.info("All files does not exist. Will be created")

        # parallel rosbag conversion
        if self._params.workers_number > 1:
            with Pool(processes=self._params.workers_number) as pool:
                pool.map(self._convert_bag, bag_dirs)
        else:
            for bag_dir in bag_dirs:
                self._convert_bag(bag_dir)

    def _convert_bag(self, bag_dir: str):
        try:
            params = copy.deepcopy(self._params)
            params.input_bag_path = bag_dir
            # TODO: Prior to Rosbag2ToT4TrackingConversion, add the functions of 'add_objects' and 'add_noise' as options.
            converter = _Rosbag2ToT4TrackingConverter(params)
            converter.convert()
            if params.data_type == DataType.SYNTHETIC:
                converter._add_scene_description("synthetic")
        except Exception:
            logger.exception(f"{bag_dir} failed with exception")
            raise


class _Rosbag2ToT4TrackingConverter(_Rosbag2ToT4Converter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params)
        self._topic_list = params.topic_list

        # overwrite sensors
        self._sensor_mode = SensorMode.NO_SENSOR
        self._sensor_channel = "LIDAR_CONCAT"
        if SENSOR_ENUM.has_channel(self._sensor_channel):
            self._sensor_enums.append(getattr(SENSOR_ENUM, self._sensor_channel))

        # overwrite and re-make initial directories
        shutil.rmtree(osp.join(self._output_base, self._bag_name), ignore_errors=True)
        self._output_scene_dir = osp.join(self._output_base, self._bag_name, "t4_dataset")
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
            self._sensor_table.insert_into_table(channel=enum.value["channel"], modality=enum.value["modality"])
        self._calibrated_sensor_table = TableHandler(CalibratedSensor)
        # extraction
        self._scene_table = TableHandler(Scene)
        self._sample_table = TableHandler(Sample)
        self._sample_data_table = TableHandler(SampleData)
        self._ego_pose_table = TableHandler(EgoPose)

    def convert(self):
        start_timestamp = self._calc_start_timestamp()

        assert (
            self._bag_reader.get_topic_count(self._object_topic_name) > 0
        ), f"No object topic name: {self._object_topic_name}"

        self._convert()
        self._convert_objects(start_timestamp)
        self._save_tables()
        self._save_config()
        self._annotation_files_generator.save_tables(self._output_anno_dir)

        self._make_input_bag()

    def _make_input_bag(self):
        output_bag_dir_temp: str = osp.join(self._output_scene_dir, osp.basename(self._input_bag))
        output_bag_dir: str = osp.join(self._output_scene_dir, "input_bag")
        converter = Rosbag2Converter(
            self._input_bag,
            output_bag_dir_temp,
            self._topic_list,
            mandatory_topics=["/tf", "/tf_static"],
        )
        converter.convert()
        shutil.move(output_bag_dir_temp, output_bag_dir)

    def _convert(self) -> None:
        """
        1. init tables
        2. fill dummy sensor data
        3. convert simulation data to annotation
        """
        sensor_channel_to_sample_data_token_list: Dict[str, List[str]] = {}

        self._init_tables()
        scene_token = self._convert_static_data()
        self._convert_dummy_pointcloud(
            sensor_channel_to_sample_data_token_list, scene_token, self._object_topic_name
        )
        self._set_scene_data()
        self._connect_sample_in_scene()
        self._connect_sample_data_in_scene(sensor_channel_to_sample_data_token_list)
        self._add_scene_description(self._scene_description)

    def _convert_dummy_pointcloud(
        self,
        sensor_channel_to_sample_data_token_list: Dict[str, List[str]],
        scene_token: str,
        objects_topic_name: str,
    ):
        # Convert dummy pointcloud data to sample_data and sample
        start_timestamp = self._calc_start_timestamp()
        logger.info(f"set start_timestamp to {start_timestamp}")

        if self._sensor_mode == SensorMode.NO_SENSOR:
            lidar_sensor_channel = self._lidar_sensor["channel"]
            sensor_channel_to_sample_data_token_list[lidar_sensor_channel] = (
                self._convert_pointcloud(
                    start_timestamp=start_timestamp,
                    sensor_channel="LIDAR_CONCAT",
                    topic=objects_topic_name,
                    scene_token=scene_token,
                )
            )

    def _convert_objects(self, start_timestamp: float):
        """read object bbox ground truth from rosbag"""
        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        scene_timestamp_objects_pair_list: List[Dict[str, Any]] = []
        for message in self._bag_reader.read_messages(
            topics=[self._object_topic_name],
            start_time=start_time_in_time,
        ):
            if self._object_msg_type in ("DetectedObjects", "TrackedObjects"):
                if message.header.frame_id != self._ego_pose_target_frame:
                    transform_stamped = self._bag_reader.get_transform_stamped(
                        target_frame=self._ego_pose_target_frame,
                        source_frame=message.header.frame_id,
                        stamp=message.header.stamp,
                    )
                    for obj in message.objects:
                        obj.kinematics.pose_with_covariance.pose = do_transform_pose(
                            obj.kinematics.pose_with_covariance.pose, transform_stamped
                        )

                scene_annotation_list = parse_perception_objects(message)
            else:
                raise ValueError(f"Invalid Object message type: {self._object_msg_type}")

            timestamp = rosbag2_utils.stamp_to_nusc_timestamp(message.header.stamp)
            scene_timestamp_objects_pair = {
                "timestamp": timestamp,
                "scene_annotation_list": scene_annotation_list,
            }
            scene_timestamp_objects_pair_list.append(scene_timestamp_objects_pair)

        assert len(scene_timestamp_objects_pair_list) > 0, "There are NO objects."

        # use the objects closest to the timestamp of the lidar
        scene_anno_dict: Dict[int, List[Dict[str, Any]]] = {}
        frame_index_to_sample_token: Dict[int, str] = {}
        for idx, sample in enumerate(self._sample_table.to_records()):
            sample: Sample
            object_dict = self._get_closest_timestamp(
                scene_timestamp_objects_pair_list,
                sample.timestamp,
            )
            scene_anno_dict[idx] = object_dict["scene_annotation_list"]
            frame_index_to_sample_token[idx] = sample.token

        self._annotation_files_generator.convert_annotations(
            scene_anno_dict=scene_anno_dict,
            frame_index_to_sample_token=frame_index_to_sample_token,
            dataset_name="synthetic",
            mask=None,
            frame_index_to_sample_data_token=None,
        )

    def _get_closest_timestamp(self, objects_list: List, timestamp: float):
        """Get the closest element to 'timestamp' from the input list."""
        res = min(
            objects_list,
            key=lambda objects_list_item: abs(objects_list_item["timestamp"] - timestamp),
        )
        return res
