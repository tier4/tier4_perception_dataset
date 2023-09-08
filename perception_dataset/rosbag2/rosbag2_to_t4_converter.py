import copy
import glob
from multiprocessing import Pool
import os
import sys
from typing import Any, Dict, List

from perception_dataset.abstract_converter_to_t4 import AbstractAnnotatedToT4Converter
from perception_dataset.ros2.tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import (
    _Rosbag2ToNonAnnotatedT4Converter,
)
from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from perception_dataset.t4_dataset.classes.calibrated_sensor import CalibratedSensorTable
from perception_dataset.t4_dataset.classes.ego_pose import EgoPoseTable
from perception_dataset.t4_dataset.classes.log import LogTable
from perception_dataset.t4_dataset.classes.map import MapTable
from perception_dataset.t4_dataset.classes.sample import SampleRecord, SampleTable
from perception_dataset.t4_dataset.classes.sample_data import SampleDataTable
from perception_dataset.t4_dataset.classes.scene import SceneTable
from perception_dataset.t4_dataset.classes.sensor import SensorTable
from perception_dataset.utils.calculate_num_points import calculate_num_points
from perception_dataset.utils.create_2d_annotations import create_2d_annotations
from perception_dataset.utils.logger import configure_logger
import perception_dataset.utils.rosbag2 as rosbag2_utils

from .autoware_msgs import parse_dynamic_object_array, parse_perception_objects

logger = configure_logger(modname=__name__)


class Rosbag2ToT4Converter(AbstractAnnotatedToT4Converter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params.input_base, params.output_base)

        self._params = params
        self._overwrite_mode = params.overwrite_mode

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
        with Pool(processes=self._params.workers_number) as pool:
            pool.map(self._convert_bag, bag_dirs)

    def _convert_bag(self, bag_dir: str):
        try:
            params = copy.deepcopy(self._params)
            params.input_bag_path = bag_dir
            converter = _Rosbag2ToT4Converter(params)
            converter.convert()
        except Exception:
            logger.exception(f"{bag_dir} failed with exception")
            raise

    def _search_bag_dirs(self):
        ret_bag_files: List[str] = []
        logger.info(f"Searching bag files in {self._input_base}")
        for bag_dir in glob.glob(os.path.join(self._input_base, "*")):
            if not os.path.isdir(bag_dir):
                continue
            logger.info(f"Found bag dir: {bag_dir}")

            meta_file = os.path.join(bag_dir, "metadata.yaml")
            if not os.path.exists(meta_file):
                logger.warning(f"{bag_dir} is directory, but metadata.yaml doesn't exist.")
                continue

            ret_bag_files.append(bag_dir)

        return ret_bag_files


class _Rosbag2ToT4Converter(_Rosbag2ToNonAnnotatedT4Converter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params)

        self._object_topic_name: str = params.object_topic_name

        # frame_id of coordinate transformation
        self._object_msg_type: str = params.object_msg_type
        self._ego_pose_target_frame: str = params.world_frame_id
        self._ego_pose_source_frame: str = "base_link"
        self._calibrated_sensor_target_frame: str = "base_link"

        self._annotation_files_generator = AnnotationFilesGenerator(with_camera=params.with_camera)

    def _init_tables(self):
        # vehicle
        self._log_table = LogTable()
        self._map_table = MapTable()
        self._sensor_table = SensorTable(
            channel_to_modality={
                enum.value["channel"]: enum.value["modality"] for enum in self._sensor_enums
            }
        )
        self._calibrated_sensor_table = CalibratedSensorTable()
        # extraction
        self._scene_table = SceneTable()
        self._sample_table = SampleTable()
        self._sample_data_table = SampleDataTable()
        self._ego_pose_table = EgoPoseTable()

    def convert(self):
        if self._start_timestamp < sys.float_info.epsilon:
            start_timestamp = self._bag_reader.start_timestamp
        else:
            start_timestamp = self._start_timestamp
        start_timestamp = start_timestamp + self._skip_timestamp

        assert (
            self._bag_reader.get_topic_count(self._object_topic_name) > 0
        ), f"No object topic name: {self._object_topic_name}"

        self._save_config()
        self._convert()
        self._convert_objects(start_timestamp)
        self._save_tables()
        self._annotation_files_generator.save_tables(self._output_anno_dir)
        # Calculate and overwrite num_lidar_prs in annotations
        self._calculate_num_points()
        self._create_2d_annotations()

    def _calculate_num_points(self):
        logger.info("Calculating number of points...")
        annotation_table = self._annotation_files_generator._sample_annotation_table
        calculate_num_points(
            self._output_scene_dir,
            lidar_sensor_channel=self._lidar_sensor["channel"],
            annotation_table=annotation_table,
        )
        annotation_table.save_json(self._output_anno_dir)

    def _create_2d_annotations(self):
        logger.info("Creating 2d camera annotations...")
        object_ann_table = self._annotation_files_generator._object_ann_table
        create_2d_annotations(
            self._output_scene_dir,
            self._camera_sensors,
            self._annotation_files_generator._sample_annotation_table,
            self._sample_data_table,
            object_ann_table,
            self._annotation_files_generator._instance_table,
        )
        object_ann_table.save_json(self._output_anno_dir)

    def _convert_objects(self, start_timestamp: float):
        """read object bbox ground truth from rosbag"""
        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        scene_timestamp_objects_pair_list: List[Dict[str, Any]] = []
        for message in self._bag_reader.read_messages(
            topics=[self._object_topic_name],
            start_time=start_time_in_time,
        ):
            if self._object_msg_type == "DynamicObjectArray":
                scene_annotation_list = parse_dynamic_object_array(message)
            elif self._object_msg_type in ("DetectedObjects", "TrackedObjects"):
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
            sample: SampleRecord
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
