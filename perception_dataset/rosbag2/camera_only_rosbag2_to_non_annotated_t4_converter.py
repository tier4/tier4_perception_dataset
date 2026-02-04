from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import glob
import json
import os
import os.path as osp
import shutil
import sys
import time

from builtin_interfaces.msg import Time
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import EXTENSION_ENUM, SENSOR_ENUM, T4_FORMAT_DIRECTORY_NAME
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.t4_dataset.classes.abstract_class import AbstractTable
from perception_dataset.t4_dataset.classes.attribute import AttributeTable
from perception_dataset.t4_dataset.classes.calibrated_sensor import CalibratedSensorTable
from perception_dataset.t4_dataset.classes.category import CategoryTable
from perception_dataset.t4_dataset.classes.ego_pose import EgoPoseTable
from perception_dataset.t4_dataset.classes.instance import InstanceTable
from perception_dataset.t4_dataset.classes.log import LogTable
from perception_dataset.t4_dataset.classes.map import MapTable
from perception_dataset.t4_dataset.classes.object_ann import ObjectAnnTable
from perception_dataset.t4_dataset.classes.sample import SampleRecord, SampleTable
from perception_dataset.t4_dataset.classes.sample_annotation import SampleAnnotationTable
from perception_dataset.t4_dataset.classes.sample_data import SampleDataRecord, SampleDataTable
from perception_dataset.t4_dataset.classes.scene import SceneRecord, SceneTable
from perception_dataset.t4_dataset.classes.sensor import SensorTable
from perception_dataset.t4_dataset.classes.surface_ann import SurfaceAnnTable
from perception_dataset.t4_dataset.classes.visibility import VisibilityTable
from perception_dataset.utils.logger import configure_logger
import perception_dataset.utils.misc as misc_utils
import perception_dataset.utils.rosbag2 as rosbag2_utils

logger = configure_logger(modname=__name__)


@dataclass
class CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutputItem:
    uncompressed_output_path: str
    zipped_output_path: str | None = None
    zipped_input_path: str | None = None


@dataclass
class CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutput:
    items: list[CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutputItem]


class CameraOnlyRosbag2ToNonAnnotatedT4Converter(
    AbstractConverter[CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutput]
):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params.input_base, params.output_base)

        self._params = params
        self._overwrite_mode = params.overwrite_mode

    def _get_bag_dirs(self) -> list[str]:
        ret_bag_files: list[str] = []
        for metadata_file in glob.glob(
            osp.join(self._input_base, "**/metadata.yaml"), recursive=True
        ):
            bag_dir = osp.dirname(metadata_file)
            if not osp.isdir(bag_dir):
                continue

            ret_bag_files.append(bag_dir)

        return ret_bag_files

    def convert(self) -> CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutput:
        bag_dirs = self._get_bag_dirs()

        if not self._overwrite_mode:
            dir_exist = False
            for bag_dir in bag_dirs[:]:  # copy to avoid modifying list while itertating
                bag_name = osp.basename(bag_dir)

                output_dir = osp.join(self._output_base, bag_name)
                if osp.exists(output_dir):
                    logger.warning(f"{output_dir} already exists.")
                    dir_exist = True
                    bag_dirs.remove(bag_dir)
            if dir_exist and len(bag_dirs) == 0:
                logger.warning(f"{output_dir} already exists.")
                raise ValueError("If you want to overwrite, use --overwrite option.")

        items: list[CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutputItem] = []
        for bag_dir in sorted(bag_dirs):
            logger.info(f"Start converting {bag_dir} to T4 format.")
            self._params.input_bag_path = bag_dir
            try:
                bag_converter = _ConverterImpl(self._params)
                output = bag_converter.convert()
                items.append(output)
            except Exception as e:
                logger.error(f"Error occurred during conversion: {e}")
                if self._params.raise_exception:
                    raise e
                continue
            logger.info(f"Conversion of {bag_dir} is completed.")
            print("----------------------------------------------------------------")

        return CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutput(items)


class _ConverterImpl:
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        self._input_bag: str = params.input_bag_path
        self._output_base: str = params.output_base
        self._skip_timestamp: float = params.skip_timestamp
        self._without_compress: bool = params.without_compress
        self._start_timestamp: float = params.start_timestamp_sec
        self._system_scan_period_sec: float = params.system_scan_period_sec
        self._max_camera_jitter_sec: float = params.max_camera_jitter_sec
        self._scene_description: str = params.scene_description

        # rosbag reader
        self._bag_reader = Rosbag2Reader(
            self._input_bag,
            with_world_frame_conversion=False,  # Is /tf needed?
            with_sensor_frame_conversion=False,  # Is /tf_static needed?
        )

        # sensors
        self._camera_sensors: list[dict[str, str]] = params.camera_sensors
        self._sensor_enums: list[SENSOR_ENUM] = []
        self._set_sensor()

        # init directories
        self._bag_name = osp.basename(self._input_bag)
        self._output_scene_dir = osp.join(self._output_base, self._bag_name)
        self._output_anno_dir = osp.join(
            self._output_scene_dir, T4_FORMAT_DIRECTORY_NAME.ANNOTATION.value
        )
        self._output_data_dir = osp.join(
            self._output_scene_dir, T4_FORMAT_DIRECTORY_NAME.DATA.value
        )

        # make output directories
        shutil.rmtree(self._output_scene_dir, ignore_errors=True)
        self._make_directories()

    def _set_sensor(self) -> None:
        sensors = self._camera_sensors.copy()
        for sensor in self._camera_sensors:
            sensor_channel = sensor["channel"]
            sensor_topic = sensor["topic"]
            if self._bag_reader.get_topic_count(sensor_topic) == 0:
                logger.warning(
                    f"Sensor topic {sensor_topic} for channel {sensor_channel} does not exist in the rosbag. This sensor will be skipped."
                )
                sensors.remove(sensor)
                continue

            if SENSOR_ENUM.has_channel(sensor_channel):
                self._sensor_enums.append(getattr(SENSOR_ENUM, sensor_channel))
        self._camera_sensors = sensors

    def _make_directories(self) -> None:
        os.makedirs(self._output_anno_dir, exist_ok=True)
        os.makedirs(self._output_data_dir, exist_ok=True)

        for sensor_enum in self._sensor_enums:
            os.makedirs(
                osp.join(self._output_data_dir, sensor_enum.value["channel"]),
                exist_ok=True,
            )

    def convert(self) -> CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutputItem:
        self._convert()

        self._save_tables()
        self._save_config()

        zipped_output_path: str | None = None
        zipped_input_path: str | None = None
        if not self._without_compress:
            (zipped_output_path, zipped_input_path) = self._compress_directory()

        return CameraOnlyRosbag2ToNonAnnotatedT4ConverterOutputItem(
            uncompressed_output_path=self._output_scene_dir,
            zipped_output_path=zipped_output_path,
            zipped_input_path=zipped_input_path,
        )

    def _save_tables(self) -> None:
        print("-----------------------------------------------------------")
        for cls_attr in self.__dict__.values():
            if isinstance(cls_attr, AbstractTable):
                print(f"{cls_attr.FILENAME}: #rows {len(cls_attr)}")
                cls_attr.save_json(self._output_anno_dir)
        print("-----------------------------------------------------------")

    def _save_config(self) -> None:
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
        config_data = {"camera_only_rosbag2_to_non_annotated_t4_converter": config_data}
        with open(osp.join(self._output_scene_dir, "status.json"), "w") as f:
            json.dump(
                config_data,
                f,
                indent=4,
                default=lambda o: getattr(o, "__dict__", str(o)),
            )

    def _compress_directory(self) -> tuple[str, str]:
        zipped_output_path = shutil.make_archive(
            self._output_scene_dir,
            "zip",
            root_dir=osp.dirname(self._output_scene_dir),
            base_dir=self._bag_name,
        )
        zipped_input_path = shutil.make_archive(self._input_bag, "zip", root_dir=self._input_bag)
        return zipped_output_path, zipped_input_path

    def _convert(self) -> None:
        start = time.time()

        sensor_channel_to_sample_data_tokens: dict[str, list[str]] = defaultdict(list)

        self._init_tables()
        scene_token = self._convert_static_data()
        self._convert_sensor_data(sensor_channel_to_sample_data_tokens, scene_token)
        self._set_scene_data()
        self._connect_sample_in_scene()
        self._connect_sample_data_in_scene(sensor_channel_to_sample_data_tokens)
        self._add_scene_description(self._scene_description)
        print(f"Total elapsed time: {time.time() - start:.2f} sec")

    def _init_tables(self) -> None:
        self._log_table = LogTable()
        self._map_table = MapTable()
        self._sensor_table = SensorTable(
            channel_to_modality={
                enum.value["channel"]: enum.value["modality"] for enum in self._sensor_enums
            }
        )
        self._calibrated_sensor_table = CalibratedSensorTable()  # NOTE: dummy records
        # extraction
        self._scene_table = SceneTable()
        self._sample_table = SampleTable()
        self._sample_data_table = SampleDataTable()
        self._ego_pose_table = EgoPoseTable()  # NOTE: dummy records
        # annotation (empty file)
        self._instance_table = InstanceTable()
        self._sample_annotation_table = SampleAnnotationTable()
        self._object_ann_table = ObjectAnnTable()
        self._surface_ann_table = SurfaceAnnTable()
        # taxonomy (empty file)
        self._category_table = CategoryTable(name_to_description={}, default_value="")
        self._attribute_table = AttributeTable(name_to_description={}, default_value="")
        self._visibility_table = VisibilityTable(level_to_description={}, default_value="")

    def _convert_static_data(self) -> str:
        # Log, Map
        log_token = self._log_table.insert_into_table(
            logfile="", vehicle="", data_captured="", location=""
        )
        self._map_table.insert_into_table(log_tokens=[log_token], category="", filename="")

        # Scene
        scene_token = self._scene_table.insert_into_table(
            name=self._bag_name, description="", log_token=log_token
        )

        return scene_token

    def _convert_sensor_data(
        self,
        sensor_channel_to_sample_data_tokens: dict[str, list[str]],
        scene_token: str,
    ) -> None:
        start_timestamp = self._calc_start_timestamp()
        logger.info(f"Set start_timestamp to {start_timestamp}")

        for camera_sensor in self._camera_sensors:
            start = time.time()
            sensor_channel = camera_sensor["channel"]
            sensor_channel_to_sample_data_tokens[sensor_channel] = self._convert_image(
                start_timestamp=start_timestamp,
                sensor_channel=sensor_channel,
                topic=camera_sensor["topic"],
                delay_msec=float(camera_sensor["delay_msec"]),
                scene_token=scene_token,
            )

            print(
                f"Camera {camera_sensor['channel']} converted. Total elapsed time: {time.time() - start:.2f} seconds"
            )

    def _calc_start_timestamp(self) -> float:
        if self._start_timestamp < sys.float_info.epsilon:
            start_timestamp = self._bag_reader.start_timestamp + self._skip_timestamp
            self._start_timestamp = start_timestamp
        return self._start_timestamp

    def _convert_image(
        self,
        start_timestamp: float,
        sensor_channel: str,
        topic: str,
        delay_msec: float,
        scene_token: str,
    ) -> list[str]:
        sample_data_tokens: list[str] = []

        start_timestamp = (
            start_timestamp - 2 * self._system_scan_period_sec - self._max_camera_jitter_sec
        )
        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        # NOTE: make dummy calibrated sensor data because rosbag does not contain camera info
        calibrated_sensor_token = self._generate_dummy_calibrated_sensor(sensor_channel)

        generated_frame_index: int = 0
        for image_msg in self._bag_reader.read_messages(
            topics=[topic], start_time=start_time_in_time
        ):
            image_msg: CompressedImage | Image
            image_unix_timestamp = rosbag2_utils.stamp_to_unix_timestamp(image_msg.header.stamp)

            # NOTE: skip inserting ego pose record because /tf is not available in the rosbag
            # Insert sample record for all camera messages
            nusc_timestamp = rosbag2_utils.stamp_to_nusc_timestamp(image_msg.header.stamp)
            sample_token: str = self._sample_table.insert_into_table(
                timestamp=nusc_timestamp, scene_token=scene_token
            )

            # TODO(ktro2828): PNG if raw image, JPEG if JPEG compressed image. How should I handle VIDEO encoded image?
            if isinstance(image_msg, CompressedImage):
                image_arr = rosbag2_utils.compressed_msg_to_numpy(image_msg)
                fileformat = (
                    EXTENSION_ENUM.JPG.value[1:]
                    if image_msg.format == "jpeg"
                    else EXTENSION_ENUM.PNG.value[1:]
                )
            else:
                image_arr = rosbag2_utils.image_msg_to_numpy(image_msg)
                fileformat = EXTENSION_ENUM.PNG.value[1:]

            print(f"Frame{generated_frame_index}, image stamp: {image_unix_timestamp}")
            sample_data_token = self._generate_image_data(
                image_arr,
                fileformat,
                image_unix_timestamp,
                sample_token,
                calibrated_sensor_token,
                sensor_channel,
                generated_frame_index,
            )
            sample_data_tokens.append(sample_data_token)
            generated_frame_index += 1

        assert len(sample_data_tokens) > 0
        return sample_data_tokens

    def _generate_dummy_calibrated_sensor(self, sensor_channel: str) -> str:
        for sensor_enum in self._sensor_enums:
            channel = sensor_enum.value["channel"]
            modality = sensor_enum.value["modality"]

            if channel != sensor_channel:
                continue

            sensor_token = self._sensor_table.insert_into_table(channel=channel, modality=modality)

            translation = {"x": 0.0, "y": 0.0, "z": 0.0}
            rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}

        calibrated_sensor_token = self._calibrated_sensor_table.insert_into_table(
            sensor_token=sensor_token,
            translation=translation,
            rotation=rotation,
            camera_intrinsic=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            camera_distortion=[0, 0, 0, 0, 0],
        )
        return calibrated_sensor_token

    def _generate_image_data(
        self,
        image_arr: np.ndarray,
        fileformat: str,
        image_unix_timestamp: float,
        sample_token: str,
        calibrated_sensor_token: str,
        sensor_channel: str,
        frame_index: int,
    ) -> str:
        ego_pose_token = self._generate_dummy_ego_pose(
            rosbag2_utils.unix_timestamp_to_stamp(image_unix_timestamp)
        )

        filename = misc_utils.get_sample_data_filename(sensor_channel, frame_index, fileformat)
        height, width = image_arr.shape[:2]
        sample_data_token = self._sample_data_table.insert_into_table(
            sample_token=sample_token,
            ego_pose_token=ego_pose_token,
            calibrated_sensor_token=calibrated_sensor_token,
            filename=filename,
            fileformat=fileformat,
            timestamp=misc_utils.unix_timestamp_to_nusc_timestamp(image_unix_timestamp),
            height=height,
            width=width,
            is_key_frame=True,
            is_valid=True,
        )

        sample_data_record: SampleDataRecord = self._sample_data_table.select_record_from_token(
            sample_data_token
        )
        # TODO(ktro2828): specify write parameters depending on fileformat
        cv2.imwrite(
            osp.join(self._output_scene_dir, sample_data_record.filename),
            image_arr,
        )
        return sample_data_token

    def _generate_dummy_ego_pose(self, stamp: Time) -> str:
        translation = {"x": 0.0, "y": 0.0, "z": 0.0}
        rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        ego_pose_token = self._ego_pose_table.insert_into_table(
            timestamp=rosbag2_utils.stamp_to_nusc_timestamp(stamp),
            translation=translation,
            rotation=rotation,
        )
        return ego_pose_token

    def _set_scene_data(self) -> None:
        scene_records: list[SceneRecord] = self._scene_table.to_records()
        assert len(scene_records) == 1, "#scene_records must be 1."

        sample_token_list: list[str] = [rec.token for rec in self._sample_table.to_records()]
        scene_record: SceneRecord = scene_records[0]

        scene_record.nbr_samples = len(sample_token_list)
        scene_record.first_sample_token = sample_token_list[0]
        scene_record.last_sample_token = sample_token_list[-1]

    def _add_scene_description(self, text: str) -> None:
        scene_records: list[SceneRecord] = self._scene_table.to_records()
        if scene_records[0].description != "":
            scene_records[0].description += ", "
        scene_records[0].description += f"{text}"

    def _connect_sample_in_scene(self) -> None:
        sample_token_list: list[str] = [rec.token for rec in self._sample_table.to_records()]

        for token_i in range(1, len(sample_token_list)):
            prev_token = sample_token_list[token_i - 1]
            cur_token = sample_token_list[token_i]

            prev_rec: SampleRecord = self._sample_table.select_record_from_token(prev_token)
            prev_rec.next = cur_token
            self._sample_table.set_record_to_table(prev_rec)

            cur_rec: SampleRecord = self._sample_table.select_record_from_token(cur_token)
            cur_rec.prev = prev_token
            self._sample_table.set_record_to_table(cur_rec)

    def _connect_sample_data_in_scene(
        self, sensor_channel_to_sample_data_token_list: dict[str, list[str]]
    ) -> None:
        for sample_data_token_list in sensor_channel_to_sample_data_token_list.values():
            for token_i in range(1, len(sample_data_token_list)):
                prev_token = sample_data_token_list[token_i - 1]
                cur_token = sample_data_token_list[token_i]

                prev_rec: SampleDataRecord = self._sample_data_table.select_record_from_token(
                    prev_token
                )
                prev_rec.next = cur_token
                self._sample_data_table.set_record_to_table(prev_rec)

                cur_rec: SampleDataRecord = self._sample_data_table.select_record_from_token(
                    cur_token
                )
                cur_rec.prev = prev_token
                self._sample_data_table.set_record_to_table(cur_rec)
