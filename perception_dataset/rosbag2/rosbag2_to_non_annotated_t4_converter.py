import enum
import glob
import json
import os
import os.path as osp
import shutil
import sys
from typing import Dict, List
import warnings

import builtin_interfaces.msg
import cv2
import numpy as np
from pyquaternion import Quaternion
from radar_msgs.msg import RadarTracks
from sensor_msgs.msg import CompressedImage, PointCloud2

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
            for bag_dir in bag_dirs:
                bag_name: str = osp.basename(bag_dir)

                output_dir = osp.join(self._output_base, bag_name)
                if osp.exists(output_dir):
                    logger.error(f"{output_dir} already exists.")
                    dir_exist = True

            if dir_exist:
                raise ValueError("If you want to overwrite files, use --overwrite option.")

        for bag_dir in bag_dirs:
            self._params.input_bag_path = bag_dir
            bag_converter = _Rosbag2ToNonAnnotatedT4Converter(self._params)
            bag_converter.convert()


class _Rosbag2ToNonAnnotatedT4Converter:
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        self._input_bag: str = params.input_bag_path
        self._output_base: str = params.output_base
        self._skip_timestamp: float = params.skip_timestamp
        self._num_load_frames: int = params.num_load_frames
        self._crop_frames_unit: int = params.crop_frames_unit
        self._without_compress: bool = params.without_compress
        self._system_scan_period_sec: float = params.system_scan_period_sec
        self._max_camera_jitter_sec: float = params.max_camera_jitter_sec
        self._lidar_latency: float = params.lidar_latency_sec
        self._lidar_points_ratio_threshold: float = params.lidar_points_ratio_threshold
        self._start_timestamp: float = params.start_timestamp_sec
        self._end_timestamp: float = 0
        self._data_type: DataType = params.data_type
        self._ignore_no_ego_transform_at_rosbag_beginning: bool = (
            params.ignore_no_ego_transform_at_rosbag_beginning
        )
        self._generate_frame_every: float = params.generate_frame_every
        self._generate_frame_every_meter: float = params.generate_frame_every_meter
        self._scene_description: str = params.scene_description
        self._accept_frame_drop: bool = params.accept_frame_drop

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
        self._msg_display_interval = 10

        shutil.rmtree(self._output_scene_dir, ignore_errors=True)
        self._make_directories()

        with_world_frame_conversion = self._ego_pose_target_frame != self._ego_pose_source_frame
        self._bag_reader = Rosbag2Reader(self._input_bag, with_world_frame_conversion)
        self._calc_actual_num_load_frames()

    def _calc_actual_num_load_frames(self):
        topic_names: List[str] = [s["topic"] for s in self._camera_sensors]
        if self._sensor_mode == SensorMode.DEFAULT:
            topic_names.append(self._lidar_sensor["topic"])
            for radar in self._radar_sensors:
                topic_names.append(radar["topic"])
            topic_names.append(self._lidar_sensor["topic"])
        if len(topic_names) == 0:
            return

        num_frames_in_bag = min([self._bag_reader.get_topic_count(t) for t in topic_names])
        freq = 10
        num_frames_to_skip = int(self._skip_timestamp * freq)
        max_num_frames = num_frames_in_bag - num_frames_to_skip - 1
        num_frames_to_crop = 0

        if not (self._num_load_frames > 0 and self._num_load_frames <= max_num_frames):
            self._num_load_frames = max_num_frames
            logger.info(
                f"max. possible number of frames will be loaded based on topic count"
                f" since the value in config is not in (0, num_frames_in_bag - num_frames_to_skip = {max_num_frames}> range."
            )

        num_frames_to_crop = self._num_load_frames % self._crop_frames_unit
        self._num_load_frames -= num_frames_to_crop

        logger.info(
            f"frames in bag: {num_frames_in_bag}, actual number of frames to load: {self._num_load_frames}, "
            f"skipped: {num_frames_to_skip}, cropped: {num_frames_to_crop})"
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
                osp.join(self._output_data_dir, sensor_enum.value["channel"]), exist_ok=True
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

    def convert(self):
        self._convert()
        self._save_tables()
        self._save_config()
        if not self._without_compress:
            self._compress_directory()

    def _save_tables(self):
        for cls_attr in self.__dict__.values():
            if isinstance(cls_attr, AbstractTable):
                print(f"{cls_attr.FILENAME}: #rows {len(cls_attr)}")
                cls_attr.save_json(self._output_anno_dir)

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
            json.dump(config_data, f, indent=4, default=lambda o: getattr(o, "__dict__", str(o)))

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
        sensor_channel_to_sample_data_token_list: Dict[str, List[str]] = {}

        self._init_tables()
        scene_token = self._convert_static_data()
        self._convert_sensor_data(sensor_channel_to_sample_data_token_list, scene_token)
        self._set_scene_data()
        self._connect_sample_in_scene()
        self._connect_sample_data_in_scene(sensor_channel_to_sample_data_token_list)
        self._add_scene_description(self._scene_description)

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
            lidar_sensor_channel = self._lidar_sensor["channel"]
            sensor_channel_to_sample_data_token_list[lidar_sensor_channel] = (
                self._convert_pointcloud(
                    start_timestamp=start_timestamp,
                    sensor_channel=lidar_sensor_channel,
                    topic=self._lidar_sensor["topic"],
                    scene_token=scene_token,
                )
            )

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
            sensor_channel = camera_sensor["channel"]
            sensor_channel_to_sample_data_token_list[sensor_channel] = self._convert_image(
                start_timestamp=camera_start_timestamp,
                sensor_channel=camera_sensor["channel"],
                topic=camera_sensor["topic"],
                delay_msec=float(camera_sensor["delay_msec"]),
                scene_token=scene_token,
            )

            first_sample_data_token: str = sensor_channel_to_sample_data_token_list[
                sensor_channel
            ][0]
            first_sample_data_record: SampleDataRecord = (
                self._sample_data_table.select_record_from_token(first_sample_data_token)
            )
            camera_start_timestamp = misc_utils.nusc_timestamp_to_unix_timestamp(
                first_sample_data_record.timestamp
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

            if frame_index >= self._num_load_frames:
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

            point_num = pointcloud_msg.width
            if point_num < max_num_points * self._lidar_points_ratio_threshold:
                if not self._accept_frame_drop:
                    raise ValueError(
                        f"PointCloud message is relatively lower than the maximum size, which is not acceptable. "
                        f"If you would like to accept, please change accept_frame_drop parameter. "
                        f"frame_index: {frame_index}, stamp: {unix_timestamp}, # points: {point_num}"
                    )
                else:
                    warnings.warn(
                        f"PointCloud message is relatively lower than the maximum size. "
                        f"May be encountering a LiDAR message drop. Skip frame_index: {frame_index}, stamp: {unix_timestamp}, # points: {point_num}"
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

            if frame_index >= self._num_load_frames:
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
        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        calibrated_sensor_token = self._generate_calibrated_sensor(
            sensor_channel, start_time_in_time, topic
        )

        if self._sensor_mode != SensorMode.NO_LIDAR:  # w/ LiDAR mode
            image_timestamp_list = [
                rosbag2_utils.stamp_to_unix_timestamp(image_msg.header.stamp)
                for image_msg in self._bag_reader.read_messages(
                    topics=[topic], start_time=start_time_in_time
                )
            ]
            lidar_timestamp_list = [
                misc_utils.nusc_timestamp_to_unix_timestamp(sample_record.timestamp)
                for sample_record in sample_records
            ]
            lidar_to_camera_latency_sec = self._lidar_latency + 1e-3 * delay_msec

            synced_frame_info_list = misc_utils.get_lidar_camera_synced_frame_info(
                image_timestamp_list=image_timestamp_list,
                lidar_timestamp_list=lidar_timestamp_list,
                lidar_to_camera_latency_sec=lidar_to_camera_latency_sec,
                system_scan_period_sec=self._system_scan_period_sec,
                max_camera_jitter_sec=self._max_camera_jitter_sec,
                num_load_frames=self._num_load_frames,
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
            for image_index, lidar_frame_index, dummy_image_timestamp in synced_frame_info_list:
                lidar_sample_token: str = sample_records[lidar_frame_index].token
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
                elif lidar_frame_index is None:  # LiDAR dropped
                    warnings.warn(f"LiDAR message dropped at image_index: {image_index}")
                else:  # Both messages available
                    image_msg = None
                    while image_index_counter < image_index:
                        image_msg = next(image_generator)
                        image_index_counter += 1

                    sample_data_token = self._generate_image_data(
                        rosbag2_utils.compressed_msg_to_numpy(image_msg),
                        rosbag2_utils.stamp_to_unix_timestamp(image_msg.header.stamp),
                        lidar_sample_token,
                        calibrated_sensor_token,
                        sensor_channel,
                        lidar_frame_index,
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
                if generated_frame_index >= self._num_load_frames:
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

    def _generate_image_data(
        self,
        image_arr: np.ndarray,
        image_unix_timestamp: float,
        sample_token: str,
        calibrated_sensor_token: str,
        sensor_channel: str,
        frame_index: int,
        output_blank_image: bool = False,
        is_key_frame: bool = True,
    ):
        ego_pose_token = self._generate_ego_pose(
            rosbag2_utils.unix_timestamp_to_stamp(image_unix_timestamp)
        )

        fileformat = EXTENSION_ENUM.JPG.value[1:]  # Note: png for all images
        filename = misc_utils.get_sample_data_filename(sensor_channel, frame_index, fileformat)
        sample_data_token = self._sample_data_table.insert_into_table(
            sample_token=sample_token,
            ego_pose_token=ego_pose_token,
            calibrated_sensor_token=calibrated_sensor_token,
            filename=filename,
            fileformat=fileformat,
            timestamp=misc_utils.unix_timestamp_to_nusc_timestamp(image_unix_timestamp),
            is_key_frame=is_key_frame,
            height=image_arr.shape[0],
            width=image_arr.shape[1],
            is_valid=is_key_frame and (not output_blank_image),
        )

        sample_data_record: SampleDataRecord = self._sample_data_table.select_record_from_token(
            sample_data_token
        )
        cv2.imwrite(osp.join(self._output_scene_dir, sample_data_record.filename), image_arr)

        return sample_data_token

    def _generate_ego_pose(self, stamp: builtin_interfaces.msg.Time) -> str:
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

    def _generate_calibrated_sensor(
        self, sensor_channel: str, start_timestamp: builtin_interfaces.msg.Time, topic_name=""
    ) -> str:
        calibrated_sensor_token = str()
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

            if modality in (SENSOR_MODALITY_ENUM.LIDAR.value, SENSOR_MODALITY_ENUM.RADAR.value):
                calibrated_sensor_token = self._calibrated_sensor_table.insert_into_table(
                    sensor_token=sensor_token,
                    translation=translation,
                    rotation=rotation,
                    camera_intrinsic=[],
                    camera_distortion=[],
                )
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
                    continue
                camera_intrinsic = np.delete(info.p.reshape(3, 4), 3, 1).tolist()
                camera_distortion = info.d.tolist()

                calibrated_sensor_token = self._calibrated_sensor_table.insert_into_table(
                    sensor_token=sensor_token,
                    translation=translation,
                    rotation=rotation,
                    camera_intrinsic=camera_intrinsic,
                    camera_distortion=camera_distortion,
                )
            else:
                raise ValueError(f"Unexpected sensor modality: {modality}")

        return calibrated_sensor_token

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
