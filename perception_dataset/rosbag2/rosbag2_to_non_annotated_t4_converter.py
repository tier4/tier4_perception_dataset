import enum
import glob
import json
import os
import os.path as osp
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import warnings

import builtin_interfaces.msg
import cv2
import numpy as np
from pyquaternion import Quaternion
from radar_msgs.msg import RadarTracks
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import (
    EXTENSION_ENUM,
    SENSOR_ENUM,
    SENSOR_MODALITY_ENUM,
    T4_FORMAT_DIRECTORY_NAME,
)
from perception_dataset.rosbag2.converter_params import DataType, Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.t4_dataset.classes.abstract_class import AbstractTable
from perception_dataset.t4_dataset.classes.attribute import AttributeTable
from perception_dataset.t4_dataset.classes.calibrated_sensor import CalibratedSensorTable
from perception_dataset.t4_dataset.classes.category import CategoryTable
from perception_dataset.t4_dataset.classes.ego_pose import EgoPoseRecord, EgoPoseTable
from perception_dataset.t4_dataset.classes.instance import InstanceTable
from perception_dataset.t4_dataset.classes.log import LogTable
from perception_dataset.t4_dataset.classes.map import MapTable
from perception_dataset.t4_dataset.classes.sample import SampleRecord, SampleTable
from perception_dataset.t4_dataset.classes.sample_annotation import SampleAnnotationTable
from perception_dataset.t4_dataset.classes.sample_data import SampleDataRecord, SampleDataTable
from perception_dataset.t4_dataset.classes.scene import SceneRecord, SceneTable
from perception_dataset.t4_dataset.classes.sensor import SensorTable
from perception_dataset.t4_dataset.classes.vehicle_state import VehicleStateTable
from perception_dataset.t4_dataset.classes.visibility import VisibilityTable
from perception_dataset.utils.logger import configure_logger
import perception_dataset.utils.misc as misc_utils
import perception_dataset.utils.rosbag2 as rosbag2_utils

logger = configure_logger(modname=__name__)


class SensorMode(enum.Enum):
    DEFAULT = "default"
    NO_LIDAR = "no_lidar"
    NO_SENSOR = "no_sensor"


class Rosbag2ToNonAnnotatedT4Converter(AbstractConverter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params.input_base, params.output_base)

        self._params = params
        self._overwrite_mode = params.overwrite_mode

    def _get_bag_dirs(self):
        ret_bag_files: List[str] = []
        for bag_dir in glob.glob(osp.join(self._input_base, "*")):
            if not osp.isdir(bag_dir):
                continue

            db3_file = osp.join(bag_dir, "metadata.yaml")
            if not osp.exists(db3_file):
                logger.warning(f"{bag_dir} is directory, but metadata.yaml doesn't exist.")
                continue

            ret_bag_files.append(bag_dir)

        return ret_bag_files

    def convert(self):
        bag_dirs: List[str] = self._get_bag_dirs()

        if not self._overwrite_mode:
            dir_exist: bool = False
            for bag_dir in bag_dirs[:]:  # copy to avoid modifying list while iterating
                bag_name: str = osp.basename(bag_dir)

                output_dir = osp.join(self._output_base, bag_name)
                if osp.exists(output_dir):
                    logger.warning(f"{output_dir} already exists.")
                    dir_exist = True
                    bag_dirs.remove(bag_dir)
            if dir_exist and len(bag_dirs) == 0:
                logger.warning(f"{output_dir} already exists.")
                raise ValueError("If you want to overwrite files, use --overwrite option.")

        for bag_dir in sorted(bag_dirs):
            logger.info(f"Start converting {bag_dir} to T4 format.")
            self._params.input_bag_path = bag_dir
            try:
                bag_converter = _Rosbag2ToNonAnnotatedT4Converter(self._params)
                bag_converter.convert()
            except Exception as e:
                logger.error(f"Error occurred during conversion: {e}")
                if self._params.raise_exception:
                    raise e
                continue
            logger.info(f"Conversion of {bag_dir} is completed")
            print(
                "--------------------------------------------------------------------------------------------------------------------------"
            )


class _Rosbag2ToNonAnnotatedT4Converter:
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        self._input_bag: str = params.input_bag_path
        self._output_base: str = params.output_base
        self._skip_timestamp: float = params.skip_timestamp
        self._num_load_frames: int = params.num_load_frames
        self._crop_frames_unit: int = params.crop_frames_unit
        self._without_compress: bool = params.without_compress
        self._system_scan_period_sec: float = min(
            params.system_scan_period_sec, params.camera_scan_period_sec
        )
        self._lidar_scan_period_sec: float = params.system_scan_period_sec
        self._camera_scan_period_sec: float = params.camera_scan_period_sec
        if self._lidar_scan_period_sec == self._camera_scan_period_sec:
            self._camera_lidar_sync_mode: bool = True
        else:
            self._camera_lidar_sync_mode: bool = False
        self._max_camera_jitter_sec: float = params.max_camera_jitter_sec
        self._lidar_latency: float = params.lidar_latency_sec
        self._lidar_points_ratio_threshold: float = params.lidar_points_ratio_threshold
        self._start_timestamp: float = params.start_timestamp_sec
        self._end_timestamp: float = sys.float_info.max
        self._data_type: DataType = params.data_type
        self._ignore_no_ego_transform_at_rosbag_beginning: bool = (
            params.ignore_no_ego_transform_at_rosbag_beginning
        )
        self._generate_frame_every: float = params.generate_frame_every
        self._generate_frame_every_meter: float = params.generate_frame_every_meter
        self._scene_description: str = params.scene_description
        self._accept_frame_drop: bool = params.accept_frame_drop
        self._undistort_image: bool = params.undistort_image

        # frame_id of coordinate transformation
        self._ego_pose_target_frame: str = params.world_frame_id
        self._ego_pose_source_frame: str = "base_link"
        self._calibrated_sensor_target_frame: str = "base_link"

        # Note: To determine if there is any message dropout, including a delay tolerance of 10Hz.
        # Note: The delay tolerance is set to 1.5 times the system scan period.
        self._timestamp_diff = params.system_scan_period_sec * 1.5

        self._lidar_sensor: Dict[str, str] = params.lidar_sensor
        self._radar_sensors: List[Dict[str, str]] = params.radar_sensors
        self._camera_sensors: List[Dict[str, str]] = params.camera_sensors
        self._sensor_enums: List = []
        self._set_sensors()

        self._sensor_mode: SensorMode = SensorMode.DEFAULT
        if (
            self._lidar_sensor["topic"] == ""
            and len(self._radar_sensors) == 0
            and len(self._camera_sensors) == 0
        ):
            self._sensor_mode = SensorMode.NO_SENSOR
        elif self._lidar_sensor["topic"] == "":
            self._sensor_mode = SensorMode.NO_LIDAR

        # init directories
        self._bag_name = osp.basename(self._input_bag)
        self._output_scene_dir = osp.join(self._output_base, self._bag_name)
        self._output_anno_dir = osp.join(
            self._output_scene_dir, T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value
        )
        self._output_data_dir = osp.join(
            self._output_scene_dir, T4_FORMAT_DIRECTORY_NAME.DATA.value
        )
        self._msg_display_interval = 100

        shutil.rmtree(self._output_scene_dir, ignore_errors=True)
        self._make_directories()

        # NOTE: even if `with_world_frame_conversion=True`, `/tf` topic is not needed and
        # it is retrieved from INS messages.
        with_world_frame_conversion = self._ego_pose_target_frame != self._ego_pose_source_frame
        is_tf_needed = with_world_frame_conversion and not params.with_ins
        is_tf_static_needed = len(self._radar_sensors) > 0 or len(self._camera_sensors) > 0
        self._bag_reader = Rosbag2Reader(self._input_bag, is_tf_needed, is_tf_static_needed)
        self._calc_actual_num_load_frames()

        # for Co-MLOps
        self._with_ins = params.with_ins
        self._with_vehicle_status = params.with_vehicle_status

        if self._with_ins:
            from perception_dataset.ros2.oxts_msgs.ins_handler import INSHandler

            self._ins_handler = INSHandler(params.input_bag_path)
        else:
            self._ins_handler = None

        if self._with_vehicle_status:
            from perception_dataset.ros2.vehicle_msgs.vehicle_status_handler import (
                VehicleStatusHandler,
            )

            self._vehicle_status_handler = VehicleStatusHandler(params.input_bag_path)
        else:
            self._vehicle_status_handler = None

    def _calc_actual_num_load_frames(self):
        cam_topic_names: List[str] = [s["topic"] for s in self._camera_sensors]
        lidar_topic_names: List[str] = []
        if self._sensor_mode == SensorMode.DEFAULT:
            lidar_topic_names.append(self._lidar_sensor["topic"])
            for radar in self._radar_sensors:
                lidar_topic_names.append(radar["topic"])
        if len(lidar_topic_names) == 0:
            self._num_load_lidar_frames = self._num_load_frames
            return

        num_cam_frames_in_bag = min([self._bag_reader.get_topic_count(t) for t in cam_topic_names])
        num_lidar_frames_in_bag = min(
            [self._bag_reader.get_topic_count(t) for t in lidar_topic_names]
        )
        for topic in lidar_topic_names + cam_topic_names:
            count = self._bag_reader.get_topic_count(topic)
            if count == 0:
                raise KeyError(f"In {self._input_bag}, {topic} message is not found.")
        cam_freq = 1 / self._camera_scan_period_sec
        lidar_freq = 1 / self._lidar_scan_period_sec
        num_cam_frames_to_skip = int(self._skip_timestamp * cam_freq)
        num_lidar_frames_to_skip = int(self._skip_timestamp * lidar_freq)
        max_num_cam_frames = num_cam_frames_in_bag - num_cam_frames_to_skip - 1
        max_num_lidar_frames = num_lidar_frames_in_bag - num_lidar_frames_to_skip - 1
        if self._camera_lidar_sync_mode:
            max_num_lidar_frames = max_num_cam_frames = min(
                max_num_lidar_frames, max_num_cam_frames
            )

        if self._num_load_frames <= 0 or self._num_load_frames > max_num_lidar_frames:
            self._num_load_lidar_frames = max_num_lidar_frames
            self._num_load_cam_frames = max_num_cam_frames
            logger.info(
                f"max. possible number of frames will be loaded based on topic count"
                f" since the value in config is not in (0, num_frames_in_bag - num_frames_to_skip = {max_num_lidar_frames}> range."
            )
        else:
            self._num_load_lidar_frames = self._num_load_frames
            self._num_load_cam_frames = int(self._num_load_frames * cam_freq / lidar_freq)

        # Set self._num_load_frames to None to indicate it is no longer needed.
        self._num_load_frames = None

        num_frames_to_crop = self._num_load_lidar_frames % self._crop_frames_unit
        self._num_load_lidar_frames -= num_frames_to_crop

        logger.info(
            f"lidar frames in bag: {num_lidar_frames_in_bag}, actual number of frames to load: {self._num_load_lidar_frames}, "
            f"skipped: {num_lidar_frames_to_skip + 1}, cropped: {num_frames_to_crop})."
            f"camera frames in bag: {num_cam_frames_in_bag}, actual number of frames to load: {self._num_load_cam_frames}, "
            f"skipped: {num_cam_frames_to_skip + 1})."
        )

    def _set_sensors(self):
        sensors: List[Dict[str, str]] = (
            [self._lidar_sensor] + self._radar_sensors + self._camera_sensors
        )
        for sensor in sensors:
            sensor_channel = sensor["channel"]
            if SENSOR_ENUM.has_channel(sensor_channel):
                self._sensor_enums.append(getattr(SENSOR_ENUM, sensor_channel))

    def _make_directories(self):
        os.makedirs(self._output_anno_dir, exist_ok=True)
        os.makedirs(self._output_data_dir, exist_ok=True)

        for sensor_enum in self._sensor_enums:
            os.makedirs(
                osp.join(self._output_data_dir, sensor_enum.value["channel"]),
                exist_ok=True,
            )

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
        # annotation (empty file)
        self._instance_table = InstanceTable()
        self._sample_annotation_table = SampleAnnotationTable()
        # taxonomy (empty file)
        self._category_table = CategoryTable(name_to_description={}, default_value="")
        self._attribute_table = AttributeTable(name_to_description={}, default_value="")
        self._visibility_table = VisibilityTable(level_to_description={}, default_value="")

        # additional (used in Co-MLops)
        self._vehicle_state_table = VehicleStateTable()

    def convert(self):
        self._convert()

        self._save_tables()
        self._save_config()
        if not self._without_compress:
            self._compress_directory()

    def _save_tables(self):
        print(
            "--------------------------------------------------------------------------------------------------------------------------"
        )
        for cls_attr in self.__dict__.values():
            if isinstance(cls_attr, AbstractTable):
                print(f"{cls_attr.FILENAME}: #rows {len(cls_attr)}")
                cls_attr.save_json(self._output_anno_dir)
        print(
            "--------------------------------------------------------------------------------------------------------------------------"
        )

    def _save_config(self):
        config_data = {
            key: getattr(self, key)
            for key in filter(
                lambda o: not o.startswith("__")
                and "sensor_enum" not in o
                and not o.endswith("_table")
                and not o.endswith("_dir")
                and not o.endswith("_base")
                and o != "_input_bag"
                and o != "_bag_reader",
                self.__dict__,
            )
        }
        config_data = {"rosbag2_to_non_annotated_t4_converter": config_data}
        with open(osp.join(self._output_scene_dir, "status.json"), "w") as f:
            json.dump(
                config_data,
                f,
                indent=4,
                default=lambda o: getattr(o, "__dict__", str(o)),
            )

    def _compress_directory(self):
        shutil.make_archive(
            self._output_scene_dir,
            "zip",
            root_dir=os.path.dirname(self._output_scene_dir),
            base_dir=self._bag_name,
        )
        shutil.make_archive(self._input_bag, "zip", root_dir=self._input_bag)

    def _convert(self) -> None:
        """
        1. init tables
            - log: dummy file
            - map: dummy file
            - sensor: fixed sensor configuration
            - calibrated_sensor: /tf_static
            - scene: dummy file, name = rosbag filename
            - sample: lidar
            - sample_data: lidar + camera x 6
            - ego_pose: /tf
            - others: empty data file
        2. read pointcloud
            - add to sample, sample_data, and ego_pose
            - is_key_frame=True
            - fill in next/prev
        3. Load the images sequentially according to their mounting positions.
            - figure out sample_data and ego_pose
            - is_key_frame=True
            - fill in next/prev
        """
        start = time.time()
        sensor_channel_to_sample_data_token_list: Dict[str, List[str]] = {}

        self._init_tables()
        scene_token = self._convert_static_data()
        self._convert_sensor_data(sensor_channel_to_sample_data_token_list, scene_token)
        self._set_scene_data()
        self._connect_sample_in_scene()
        self._connect_sample_data_in_scene(sensor_channel_to_sample_data_token_list)
        self._add_scene_description(self._scene_description)
        print(f"Total elapsed time: {time.time() - start:.2f} sec")

        if self._with_vehicle_status:
            self._convert_vehicle_state()

    def _calc_start_timestamp(self) -> float:
        if self._start_timestamp < sys.float_info.epsilon:
            start_timestamp = self._bag_reader.start_timestamp + self._skip_timestamp
            self._start_timestamp = start_timestamp
        else:
            start_timestamp = self._start_timestamp
        return start_timestamp

    def _convert_sensor_data(
        self,
        sensor_channel_to_sample_data_token_list: Dict[str, List[str]],
        scene_token: str,
    ):
        start_timestamp = self._calc_start_timestamp()
        logger.info(f"set start_timestamp to {start_timestamp}")

        if self._sensor_mode == SensorMode.DEFAULT:
            start = time.time()
            lidar_sensor_channel = self._lidar_sensor["channel"]
            sensor_channel_to_sample_data_token_list[lidar_sensor_channel] = (
                self._convert_pointcloud(
                    start_timestamp=start_timestamp,
                    sensor_channel=lidar_sensor_channel,
                    topic=self._lidar_sensor["topic"],
                    scene_token=scene_token,
                )
            )
            print(f"LiDAR conversion. total elapsed time: {time.time() - start:.2f} sec\n")
            for radar_sensor in self._radar_sensors:
                radar_sensor_channel = radar_sensor["channel"]
                sensor_channel_to_sample_data_token_list[radar_sensor_channel] = (
                    self._convert_radar_tracks(
                        start_timestamp=start_timestamp,
                        sensor_channel=radar_sensor_channel,
                        topic=radar_sensor["topic"],
                        scene_token=scene_token,
                    )
                )

            # Note: Align the loading order of the cameras with the shutter sequence.
            # Note: The timing of lidar scan initiation and the first camera data acquisition is the same, but the camera has a delay due to data transfer and edge processing on the timestamp.
            first_sample_data_record: SampleDataRecord = self._sample_data_table.to_records()[0]

        if self._sensor_mode == SensorMode.NO_LIDAR or self._sensor_mode == SensorMode.NO_SENSOR:
            # temporaly use start_timestamp instead of recorded timestamp for non synced data
            camera_start_timestamp = start_timestamp
        else:
            camera_start_timestamp = misc_utils.nusc_timestamp_to_unix_timestamp(
                first_sample_data_record.timestamp
            )

        for camera_sensor in self._camera_sensors:
            start = time.time()
            sensor_channel = camera_sensor["channel"]
            sensor_channel_to_sample_data_token_list[sensor_channel] = self._convert_image(
                start_timestamp=camera_start_timestamp,
                sensor_channel=camera_sensor["channel"],
                topic=camera_sensor["topic"],
                delay_msec=float(camera_sensor["delay_msec"]),
                scene_token=scene_token,
            )

            print(
                f"camera {camera_sensor['channel']} conversion. total elapsed time: {time.time() - start:.2f} sec\n"
            )

    def _convert_static_data(self):
        # Log, Map
        log_token = self._log_table.insert_into_table(
            logfile="",
            vehicle="",
            data_captured="",
            location="",
        )
        self._map_table.insert_into_table(log_tokens=[log_token], category="", filename="")

        # Scene
        scene_token = self._scene_table.insert_into_table(
            name=self._bag_name,
            description="",
            log_token=log_token,
        )

        return scene_token

    def _convert_pointcloud(
        self,
        start_timestamp: float,
        sensor_channel: str,
        topic: str,
        scene_token: str,
    ) -> List[str]:
        sample_data_token_list: List[str] = []

        prev_frame_unix_timestamp: float = 0.0
        frame_index: int = 0

        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        calibrated_sensor_token = self._generate_calibrated_sensor(
            sensor_channel, start_time_in_time, topic
        )

        # Calculate the maximum number of points
        max_num_points = 0
        topic_check_count = 0
        for pointcloud_msg in self._bag_reader.read_messages(
            topics=[topic],
            start_time=start_time_in_time,
        ):
            # Find the maximum number of points in the point cloud in the first 100 messages.
            if hasattr(pointcloud_msg, "width"):
                max_num_points = max(max_num_points, pointcloud_msg.width)
            else:
                max_num_points = 0
            topic_check_count += 1
            if topic_check_count > 100:
                break

        # Main iteration
        for pointcloud_msg in self._bag_reader.read_messages(
            topics=[topic],
            start_time=start_time_in_time,
        ):
            pointcloud_msg: PointCloud2

            try:
                ego_pose_token = self._generate_ego_pose(pointcloud_msg.header.stamp)
            except Exception as e:
                if self._ignore_no_ego_transform_at_rosbag_beginning:
                    warnings.warn(
                        f"Skipping frame with header stamp: {pointcloud_msg.header.stamp}"
                    )
                    continue
                else:
                    raise e

            if frame_index >= self._num_load_lidar_frames:
                break

            unix_timestamp = rosbag2_utils.stamp_to_unix_timestamp(pointcloud_msg.header.stamp)
            if frame_index > 0:
                time_diff = unix_timestamp - prev_frame_unix_timestamp
                if frame_index % self._msg_display_interval == 0:
                    print(
                        f"frame_index:{frame_index}: {unix_timestamp}, unix_timestamp - prev_frame_unix_timestamp: {time_diff}"
                    )
                # Note: LiDAR Message drops are not accepted unless accept_frame_drop is True.
                if not self._accept_frame_drop and (
                    time_diff > self._timestamp_diff or unix_timestamp < prev_frame_unix_timestamp
                ):
                    raise ValueError(
                        f"PointCloud message is dropped [{frame_index}]: cur={unix_timestamp} prev={prev_frame_unix_timestamp}"
                    )

            if hasattr(pointcloud_msg, "width"):
                num_points = pointcloud_msg.width
            else:
                num_points = 0
            if num_points < max_num_points * self._lidar_points_ratio_threshold:
                if not self._accept_frame_drop:
                    raise ValueError(
                        f"PointCloud message is relatively lower than the maximum size, which is not acceptable. "
                        f"If you would like to accept, please change accept_frame_drop parameter. "
                        f"frame_index: {frame_index}, stamp: {unix_timestamp}, # points: {num_points}"
                    )
                else:
                    warnings.warn(
                        f"PointCloud message is relatively lower than the maximum size. "
                        f"May be encountering a LiDAR message drop. Skip frame_index: {frame_index}, stamp: {unix_timestamp}, # points: {num_points}"
                    )
                    continue

            nusc_timestamp = rosbag2_utils.stamp_to_nusc_timestamp(pointcloud_msg.header.stamp)
            sample_token = self._sample_table.insert_into_table(
                timestamp=nusc_timestamp, scene_token=scene_token
            )

            fileformat = EXTENSION_ENUM.PCDBIN.value[1:]
            filename = misc_utils.get_sample_data_filename(sensor_channel, frame_index, fileformat)
            sample_data_token = self._sample_data_table.insert_into_table(
                sample_token=sample_token,
                ego_pose_token=ego_pose_token,
                calibrated_sensor_token=calibrated_sensor_token,
                filename=filename,
                fileformat=fileformat,
                timestamp=nusc_timestamp,
                is_key_frame=True,
            )
            sample_data_record: SampleDataRecord = (
                self._sample_data_table.select_record_from_token(sample_data_token)
            )

            # TODO(yukke42): Save data in the PCD file format, which allows flexible field configuration.
            points_arr = rosbag2_utils.pointcloud_msg_to_numpy(pointcloud_msg)
            if len(points_arr) == 0:
                warnings.warn(
                    f"PointCloud message is empty [{frame_index}]: cur={unix_timestamp} prev={prev_frame_unix_timestamp}"
                )

            points_arr.tofile(osp.join(self._output_scene_dir, sample_data_record.filename))

            sample_data_token_list.append(sample_data_token)
            prev_frame_unix_timestamp = unix_timestamp
            frame_index += 1

        return sample_data_token_list

    def _convert_radar_tracks(
        self,
        start_timestamp: float,
        sensor_channel: str,
        topic: str,
        scene_token: str,
    ) -> List[str]:
        sample_data_token_list: List[str] = []

        prev_frame_unix_timestamp: float = 0.0
        frame_index: int = 0

        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        calibrated_sensor_token = self._generate_calibrated_sensor(
            sensor_channel, start_time_in_time, topic
        )
        for radar_tracks_msg in self._bag_reader.read_messages(
            topics=[topic], start_time=start_time_in_time
        ):
            radar_tracks_msg: RadarTracks
            try:
                ego_pose_token = self._generate_ego_pose(radar_tracks_msg.header.stamp)
            except Exception as e:
                if self._ignore_no_ego_transform_at_rosbag_beginning:
                    warnings.warn(
                        f"Skipping frame with header stamp: {radar_tracks_msg.header.stamp}"
                    )
                    continue
                else:
                    raise e

            if frame_index >= self._num_load_lidar_frames:
                break

            unix_timestamp = rosbag2_utils.stamp_to_unix_timestamp(radar_tracks_msg.header.stamp)
            if frame_index > 0:
                # NOTE: Message drops are not tolerated.
                print(
                    f"frame_index: {frame_index}, unix_timestamp - prev_frame_unix_timestamp: {unix_timestamp - prev_frame_unix_timestamp}"
                )
                if (
                    unix_timestamp - prev_frame_unix_timestamp
                ) > self._timestamp_diff or unix_timestamp < prev_frame_unix_timestamp:
                    raise ValueError(
                        f"{topic} message is dropped [{frame_index}]: cur={unix_timestamp} prev={prev_frame_unix_timestamp}"
                    )

            nusc_timestamp = rosbag2_utils.stamp_to_nusc_timestamp(radar_tracks_msg.header.stamp)
            sample_token = self._sample_table.insert_into_table(
                timestamp=nusc_timestamp, scene_token=scene_token
            )

            # TODO(ktro2828): add support of PCD format
            fileformat = EXTENSION_ENUM.JSON.value[1:]
            filename = misc_utils.get_sample_data_filename(sensor_channel, frame_index, fileformat)
            sample_data_token = self._sample_data_table.insert_into_table(
                sample_token=sample_token,
                ego_pose_token=ego_pose_token,
                calibrated_sensor_token=calibrated_sensor_token,
                filename=filename,
                fileformat=fileformat,
                timestamp=nusc_timestamp,
                is_key_frame=False,
            )
            sample_data_record: SampleDataRecord = (
                self._sample_data_table.select_record_from_token(sample_data_token)
            )

            # TODO(ktro2828): Add support of PCD format.
            radar_tracks = rosbag2_utils.radar_tracks_msg_to_list(radar_tracks_msg)
            with open(osp.join(self._output_scene_dir, sample_data_record.filename), "w") as f:
                json.dump(radar_tracks, f, ensure_ascii=False, indent=4)

            sample_data_token_list.append(sample_data_token)
            prev_frame_unix_timestamp = unix_timestamp
            frame_index += 1

        return sample_data_token_list

    def _convert_image(
        self,
        start_timestamp: float,
        sensor_channel: str,
        topic: str,
        delay_msec: float,
        scene_token: str,
    ):
        """convert image topic to raw image data"""
        sample_data_token_list: List[str] = []
        sample_records: List[SampleRecord] = self._sample_table.to_records()

        # Get calibrated sensor token
        start_timestamp = (
            start_timestamp - 2 * self._lidar_scan_period_sec - self._max_camera_jitter_sec
        )  # assume the camera might be triggered before the LiDAR
        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        calibrated_sensor_token, camera_info = self._generate_calibrated_sensor(
            sensor_channel, start_time_in_time, topic
        )

        if self._undistort_image:
            self.undistort_map_x, self.undistort_map_y = cv2.initUndistortRectifyMap(
                camera_info.k.reshape(3, 3),
                np.array(camera_info.d),
                None,
                camera_info.p.reshape(3, 4)[:3],
                (camera_info.width, camera_info.height),
                cv2.CV_32FC1,
            )

        if self._sensor_mode != SensorMode.NO_LIDAR:  # w/ LiDAR mode
            image_timestamp_list = [
                # handle sensor delay here
                rosbag2_utils.stamp_to_unix_timestamp(image_msg.header.stamp) - 1e-3 * delay_msec
                for image_msg in self._bag_reader.read_messages(
                    topics=[topic], start_time=start_time_in_time
                )
            ]
            lidar_timestamp_list = [
                # handle sensor delay here
                misc_utils.nusc_timestamp_to_unix_timestamp(sample_record.timestamp)
                - self._lidar_latency
                for sample_record in sample_records
            ]

            if self._camera_lidar_sync_mode or not self._accept_frame_drop:
                synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
                    image_timestamp_list=image_timestamp_list,
                    lidar_timestamp_list=lidar_timestamp_list,
                    system_scan_period_sec=self._system_scan_period_sec,
                    max_camera_jitter_sec=self._max_camera_jitter_sec,
                    num_load_frames=self._num_load_lidar_frames,
                    msg_display_interval=self._msg_display_interval,
                )
            else:
                synced_frame_info_list = misc_utils.get_lidar_camera_frame_info_async(
                    image_timestamp_list=image_timestamp_list,
                    lidar_timestamp_list=lidar_timestamp_list,
                    max_camera_jitter=self._max_camera_jitter_sec,
                    camera_scan_period=self._camera_scan_period_sec,
                    num_load_image_frames=self._num_load_cam_frames,
                    num_load_lidar_frames=self._num_load_lidar_frames,
                    msg_display_interval=self._msg_display_interval,
                )

            # Get image shape
            temp_image_msg = next(self._bag_reader.read_messages(topics=[topic]))
            image_shape = rosbag2_utils.compressed_msg_to_numpy(temp_image_msg).shape

            # Save image
            sample_data_token_list: List[str] = []
            image_index_counter = -1
            image_generator = self._bag_reader.read_messages(
                topics=[topic], start_time=start_time_in_time
            )
            for (
                image_index,
                lidar_frame_index,
                dummy_image_timestamp,
            ) in synced_frame_info_list:
                lidar_sample_token: str = (
                    sample_records[lidar_frame_index].token
                    if lidar_frame_index is not None
                    else None
                )
                if image_index is None:  # Image dropped
                    sample_data_token = self._generate_image_data(
                        np.zeros(shape=image_shape, dtype=np.uint8),  # dummy image
                        dummy_image_timestamp,
                        lidar_sample_token,
                        calibrated_sensor_token,
                        sensor_channel,
                        lidar_frame_index,
                        output_blank_image=True,
                        is_key_frame=False,
                    )
                    sample_data_token_list.append(sample_data_token)
                elif lidar_frame_index is None and not self._accept_frame_drop:  # LiDAR dropped
                    warnings.warn(f"LiDAR message dropped at image_index: {image_index}")
                else:  # Both messages available
                    image_msg = None
                    while image_index_counter < image_index:
                        image_msg = next(image_generator)
                        image_index_counter += 1

                    file_index = lidar_frame_index if self._camera_lidar_sync_mode else image_index
                    sample_data_token = self._generate_image_data(
                        image_msg,
                        rosbag2_utils.stamp_to_unix_timestamp(image_msg.header.stamp),
                        lidar_sample_token,
                        calibrated_sensor_token,
                        sensor_channel,
                        file_index,
                        image_shape,
                        camera_info=camera_info,
                    )
                    sample_data_token_list.append(sample_data_token)
        else:  # camera only mode

            def get_move_distance(trans1: Dict[str, float], trans2: Dict[str, float]) -> float:
                dx: float = trans1["x"] - trans2["x"]
                dy: float = trans1["y"] - trans2["y"]
                dz: float = trans1["z"] - trans2["z"]
                return (dx * dx + dy * dy + dz * dz) ** 0.5

            frame_index: int = 0
            generated_frame_index: int = 0

            last_translation: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
            for image_msg in self._bag_reader.read_messages(
                topics=[topic],
                start_time=start_time_in_time,
            ):
                image_msg: CompressedImage
                if generated_frame_index >= self._num_load_cam_frames:
                    break

                image_unix_timestamp = rosbag2_utils.stamp_to_unix_timestamp(
                    image_msg.header.stamp
                )

                is_data_found: bool = False

                # camera_only_mode
                if (frame_index % self._generate_frame_every) == 0:
                    try:
                        ego_pose_token = self._generate_ego_pose(image_msg.header.stamp)
                    except Exception as e:
                        print(e)
                        continue
                    ego_pose: EgoPoseRecord = self._ego_pose_table.select_record_from_token(
                        ego_pose_token
                    )
                    translation: Dict[str, float] = ego_pose.translation
                    distance = get_move_distance(translation, last_translation)
                    if distance >= self._generate_frame_every_meter:
                        last_translation = translation

                        nusc_timestamp = rosbag2_utils.stamp_to_nusc_timestamp(
                            image_msg.header.stamp
                        )
                        sample_token: str = self._sample_table.insert_into_table(
                            timestamp=nusc_timestamp, scene_token=scene_token
                        )
                        is_data_found = True

                if is_data_found:
                    print(f"frame{generated_frame_index}, image stamp: {image_unix_timestamp}")
                    sample_data_token = self._generate_image_data(
                        rosbag2_utils.compressed_msg_to_numpy(image_msg),
                        image_unix_timestamp,
                        sample_token,
                        calibrated_sensor_token,
                        sensor_channel,
                        generated_frame_index,
                    )
                    sample_data_token_list.append(sample_data_token)
                    generated_frame_index += 1
                frame_index += 1

        assert len(sample_data_token_list) > 0

        return sample_data_token_list

    def _convert_vehicle_state(self) -> None:
        msgs = self._vehicle_status_handler.get_actuation_statuses()
        stamps = [msg.header.stamp for msg in msgs]
        for stamp in stamps:
            self._generate_vehicle_state(stamp)

    def _generate_image_data(
        self,
        image_arr: Union[np.ndarray, CompressedImage],
        image_unix_timestamp: float,
        sample_token: Optional[str],
        calibrated_sensor_token: str,
        sensor_channel: str,
        frame_index: int,
        image_shape: Tuple[int, int, int] = (0, 0, 0),
        output_blank_image: bool = False,
        is_key_frame: bool = True,
        camera_info: Optional[CameraInfo] = None,
    ):
        ego_pose_token = self._generate_ego_pose(
            rosbag2_utils.unix_timestamp_to_stamp(image_unix_timestamp)
        )

        fileformat = EXTENSION_ENUM.JPG.value[1:]  # Note: png for all images
        filename = misc_utils.get_sample_data_filename(sensor_channel, frame_index, fileformat)
        if hasattr(image_arr, "shape"):
            image_shape = image_arr.shape
        if sample_token is None:
            is_key_frame = False
        sample_data_token = self._sample_data_table.insert_into_table(
            sample_token=sample_token,
            ego_pose_token=ego_pose_token,
            calibrated_sensor_token=calibrated_sensor_token,
            filename=filename,
            fileformat=fileformat,
            timestamp=misc_utils.unix_timestamp_to_nusc_timestamp(image_unix_timestamp),
            is_key_frame=is_key_frame,
            height=image_shape[0],
            width=image_shape[1],
            is_valid=(not output_blank_image),
        )

        sample_data_record: SampleDataRecord = self._sample_data_table.select_record_from_token(
            sample_data_token
        )
        if isinstance(image_arr, np.ndarray):
            cv2.imwrite(
                osp.join(self._output_scene_dir, sample_data_record.filename),
                image_arr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
        elif isinstance(image_arr, CompressedImage):
            output_image_path: str = osp.join(self._output_scene_dir, sample_data_record.filename)
            if camera_info is None or not self._undistort_image:
                # save compressed image as is
                with open(output_image_path, "xb") as fw:
                    fw.write(image_arr.data)
            else:
                # load image and undistort
                image = rosbag2_utils.compressed_msg_to_numpy(image_arr)
                image = cv2.remap(
                    image, self.undistort_map_x, self.undistort_map_y, cv2.INTER_LINEAR
                )
                cv2.imwrite(output_image_path, image)

        return sample_data_token

    def _generate_ego_pose(self, stamp: builtin_interfaces.msg.Time) -> str:
        if self._with_ins:
            ego_state = self._ins_handler.get_ego_state(stamp=stamp)
            geocoordinate = self._ins_handler.lookup_nav_sat_fixes(stamp)

            ego_pose_token = self._ego_pose_table.insert_into_table(
                translation={
                    "x": ego_state.translation.x,
                    "y": ego_state.translation.y,
                    "z": ego_state.translation.z,
                },
                rotation={
                    "w": ego_state.rotation.w,
                    "x": ego_state.rotation.x,
                    "y": ego_state.rotation.y,
                    "z": ego_state.rotation.z,
                },
                timestamp=rosbag2_utils.stamp_to_nusc_timestamp(ego_state.header.stamp),
                twist={
                    "vx": ego_state.twist.linear.x,
                    "vy": ego_state.twist.linear.y,
                    "vz": ego_state.twist.linear.z,
                    "yaw_rate": ego_state.twist.angular.z,
                    "pitch_rate": ego_state.twist.angular.y,
                    "roll_rate": ego_state.twist.angular.x,
                },
                acceleration={
                    "ax": ego_state.accel.x,
                    "ay": ego_state.accel.y,
                    "az": ego_state.accel.z,
                },
                geocoordinate=(
                    {
                        "latitude": geocoordinate.latitude,
                        "longitude": geocoordinate.longitude,
                        "altitude": geocoordinate.altitude,
                    }
                    if geocoordinate is not None
                    else None
                ),
            )
        else:
            transform_stamped = self._bag_reader.get_transform_stamped(
                target_frame=self._ego_pose_target_frame,
                source_frame=self._ego_pose_source_frame,
                stamp=stamp,
            )

            ego_pose_token = self._ego_pose_table.insert_into_table(
                translation={
                    "x": transform_stamped.transform.translation.x,
                    "y": transform_stamped.transform.translation.y,
                    "z": transform_stamped.transform.translation.z,
                },
                rotation={
                    "w": transform_stamped.transform.rotation.w,
                    "x": transform_stamped.transform.rotation.x,
                    "y": transform_stamped.transform.rotation.y,
                    "z": transform_stamped.transform.rotation.z,
                },
                timestamp=rosbag2_utils.stamp_to_nusc_timestamp(transform_stamped.header.stamp),
            )

        return ego_pose_token

    def _generate_vehicle_state(self, stamp: builtin_interfaces.msg.Time) -> str:
        nusc_timestamp = rosbag2_utils.stamp_to_nusc_timestamp(stamp)

        # TODO(ktro2828): Implement operation to insert vehicle state into table
        actuation_msg = self._vehicle_status_handler.get_closest_msg(
            key="actuation_status", stamp=stamp
        )

        # steering tire
        steering_tire_msg = self._vehicle_status_handler.get_closest_msg(
            key="steering_status",
            stamp=stamp,
        )

        # steering wheel
        steering_wheel_msg = self._vehicle_status_handler.get_closest_msg(
            key="steering_wheel_status",
            stamp=stamp,
        )

        # gear -> shift
        gear_msg = self._vehicle_status_handler.get_closest_msg(key="gear_status", stamp=stamp)
        shift_state = self._vehicle_status_handler.gear_to_shift(gear_msg.report)

        # indicators
        turn_indicators_msg = self._vehicle_status_handler.get_closest_msg(
            key="turn_indicators_status", stamp=stamp
        )
        indicators_state = self._vehicle_status_handler.indicator_to_state(
            turn_indicators_msg.report
        )

        # additional info
        # --- speed ---
        velocity_report_msg = self._vehicle_status_handler.get_closest_msg(
            key="velocity_status", stamp=stamp
        )
        speed = np.linalg.norm(
            [
                velocity_report_msg.longitudinal_velocity,
                velocity_report_msg.lateral_velocity,
            ]
        )

        vehicle_state_token = self._vehicle_state_table.insert_into_table(
            timestamp=nusc_timestamp,
            accel_pedal=actuation_msg.status.accel_status,
            brake_pedal=actuation_msg.status.brake_status,
            steer_pedal=actuation_msg.status.steer_status,
            steering_tire_angle=steering_tire_msg.steering_tire_angle,
            steering_wheel_angle=steering_wheel_msg.data,
            shift_state=shift_state,
            indicators=indicators_state,
            additional_info={"speed": speed},
        )

        return vehicle_state_token

    def _generate_calibrated_sensor(
        self,
        sensor_channel: str,
        start_timestamp: builtin_interfaces.msg.Time,
        topic_name="",
    ) -> Union[str, Tuple[str, CameraInfo]]:
        calibrated_sensor_token = str()
        camera_info = None
        for sensor_enum in self._sensor_enums:
            channel = sensor_enum.value["channel"]
            modality = sensor_enum.value["modality"]

            if channel != sensor_channel:
                continue

            sensor_token = self._sensor_table.insert_into_table(
                channel=channel,
                modality=modality,
            )

            translation = {"x": 0.0, "y": 0.0, "z": 0.0}
            rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
            frame_id = self._bag_reader.sensor_topic_to_frame_id.get(topic_name)
            print(
                f"generate_calib_sensor, start_timestamp:{start_timestamp}, topic name:{topic_name}, frame id:{frame_id}"
            )
            if frame_id is not None:
                transform_stamped = self._bag_reader.get_transform_stamped(
                    target_frame=self._calibrated_sensor_target_frame,
                    source_frame=frame_id,
                    stamp=start_timestamp,
                )
                translation = {
                    "x": transform_stamped.transform.translation.x,
                    "y": transform_stamped.transform.translation.y,
                    "z": transform_stamped.transform.translation.z,
                }
                rotation = {
                    "w": transform_stamped.transform.rotation.w,
                    "x": transform_stamped.transform.rotation.x,
                    "y": transform_stamped.transform.rotation.y,
                    "z": transform_stamped.transform.rotation.z,
                }

            if modality in (
                SENSOR_MODALITY_ENUM.LIDAR.value,
                SENSOR_MODALITY_ENUM.RADAR.value,
            ):
                calibrated_sensor_token = self._calibrated_sensor_table.insert_into_table(
                    sensor_token=sensor_token,
                    translation=translation,
                    rotation=rotation,
                    camera_intrinsic=[],
                    camera_distortion=[],
                )
                return calibrated_sensor_token
            elif modality == SENSOR_MODALITY_ENUM.CAMERA.value:
                if self._data_type.value == "synthetic":
                    # fix of the orientation of camera view
                    rotation = Quaternion(
                        rotation["w"], rotation["x"], rotation["y"], rotation["z"]
                    )
                    axes_fix_rotation = Quaternion(0.5, -0.5, 0.5, -0.5)
                    rotation = rotation * axes_fix_rotation

                    rotation = {
                        "w": rotation.w,
                        "x": rotation.x,
                        "y": rotation.y,
                        "z": rotation.z,
                    }

                topic_name_splitted = topic_name.split("/")
                cam_info_topic = "/".join(topic_name_splitted[:4]) + "/camera_info"
                info = self._bag_reader.camera_info.get(cam_info_topic)
                if info is None:
                    raise ValueError(f"Camera info not found for {cam_info_topic}")
                camera_intrinsic, camera_distortion, camera_info = self._parse_camera_info(
                    info,
                    undistort_image=self._undistort_image,
                    is_already_rectified="image_rect" in topic_name,
                )

                calibrated_sensor_token = self._calibrated_sensor_table.insert_into_table(
                    sensor_token=sensor_token,
                    translation=translation,
                    rotation=rotation,
                    camera_intrinsic=camera_intrinsic,
                    camera_distortion=camera_distortion,
                )
                return calibrated_sensor_token, camera_info
            else:
                raise ValueError(f"Unexpected sensor modality: {modality}")
        raise ValueError(f"Sensor channel {sensor_channel} not found in the sensor list.")

    def _parse_camera_info(
        self,
        info: CameraInfo,
        undistort_image: bool = False,
        is_already_rectified: bool = False,
    ) -> Tuple[List[float], List[float], CameraInfo]:
        camera_info = None
        if is_already_rectified:
            # image is already undistorted
            camera_intrinsic = np.delete(info.p.reshape(3, 4), 3, 1).tolist()
            camera_distortion = info.d.tolist()
        elif undistort_image:
            camera_intrinsic = np.delete(info.p.reshape(3, 4), 3, 1).tolist()
            camera_distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info = info
        else:
            camera_intrinsic = info.k.reshape(3, 3).tolist()
            camera_distortion = info.d.tolist()
        return camera_intrinsic, camera_distortion, camera_info

    def _set_scene_data(self):
        scene_records: List[SceneRecord] = self._scene_table.to_records()
        assert len(scene_records) == 1, "#scene_records must be 1."

        sample_token_list: List[str] = [rec.token for rec in self._sample_table.to_records()]
        scene_record: SceneRecord = scene_records[0]

        scene_record.nbr_samples = len(sample_token_list)
        scene_record.first_sample_token = sample_token_list[0]
        scene_record.last_sample_token = sample_token_list[-1]

    def _add_scene_description(self, text: str):
        scene_records: List[SceneRecord] = self._scene_table.to_records()
        if scene_records[0].description != "":
            scene_records[0].description += ", "
        scene_records[0].description += f"{text}"

    def _connect_sample_in_scene(self):
        """add prev/next of Sample"""
        sample_token_list: List[str] = [rec.token for rec in self._sample_table.to_records()]

        for token_i in range(1, len(sample_token_list)):
            prev_token: str = sample_token_list[token_i - 1]
            cur_token: str = sample_token_list[token_i]

            prev_rec: SampleRecord = self._sample_table.select_record_from_token(prev_token)
            prev_rec.next = cur_token
            self._sample_table.set_record_to_table(prev_rec)

            cur_rec: SampleRecord = self._sample_table.select_record_from_token(cur_token)
            cur_rec.prev = prev_token
            self._sample_table.set_record_to_table(cur_rec)

    def _connect_sample_data_in_scene(
        self, sensor_channel_to_sample_data_token_list: Dict[str, List[str]]
    ):
        """add prev/next of SampleData"""
        for sample_data_token_list in sensor_channel_to_sample_data_token_list.values():
            for token_i in range(1, len(sample_data_token_list)):
                prev_token: str = sample_data_token_list[token_i - 1]
                cur_token: str = sample_data_token_list[token_i]

                prev_rec: SampleRecord = self._sample_data_table.select_record_from_token(
                    prev_token
                )
                prev_rec.next = cur_token
                self._sample_data_table.set_record_to_table(prev_rec)

                cur_rec: SampleRecord = self._sample_data_table.select_record_from_token(cur_token)
                cur_rec.prev = prev_token
                self._sample_data_table.set_record_to_table(cur_rec)
                prev_rec: SampleRecord = self._sample_data_table.select_record_from_token(
                    prev_token
                )
                prev_rec.next = cur_token
                self._sample_data_table.set_record_to_table(prev_rec)

                cur_rec: SampleRecord = self._sample_data_table.select_record_from_token(cur_token)
                cur_rec.prev = prev_token
                self._sample_data_table.set_record_to_table(cur_rec)
