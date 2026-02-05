import base64
import copy
import glob
import os.path as osp
import sys
from typing import Any, Dict, List, Set, Union
import warnings

import numpy as np
from pycocotools import mask as cocomask
from sensor_msgs.msg import CompressedImage
from t4_devkit.schema import EgoPose, Sample, SampleData
from tier4_perception_msgs.msg import TrafficLightArray, TrafficLightRoiArray
import yaml

from perception_dataset.rosbag2.autoware_msgs import parse_traffic_lights
from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_converter import Rosbag2Converter
from perception_dataset.rosbag2.rosbag2_to_non_annotated_t4_converter import SensorMode
from perception_dataset.rosbag2.rosbag2_to_t4_converter import (
    Rosbag2ToT4Converter,
    _Rosbag2ToT4Converter,
)
from perception_dataset.utils.logger import configure_logger
import perception_dataset.utils.misc as misc_utils
from perception_dataset.utils.misc import unix_timestamp_to_nusc_timestamp
import perception_dataset.utils.rosbag2 as rosbag2_utils

logger = configure_logger(modname=__name__)


class Rosbag2ToAnnotatedT4TlrConverter(Rosbag2ToT4Converter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params)

    def _convert_bag(self, bag_dir: str):
        try:
            params = copy.deepcopy(self._params)
            params.input_bag_path = bag_dir
            converter = _Rosbag2ToAnnotatedT4TlrConverter(params)
            converter.convert()
        except Exception:
            logger.exception(f"{bag_dir} failed with exception")
            raise


# TODO: This implementation has a lot in common with Rosbag2ToNonAnnotatedT4Converter,
# and thus these two classes should be merged somehow in the near future.
class _Rosbag2ToAnnotatedT4TlrConverter(_Rosbag2ToT4Converter):
    def __init__(self, params: Rosbag2ConverterParams) -> None:
        super().__init__(params)
        self._traffic_light_signal_topic_name: str = params.traffic_light_signal_topic_name
        self._traffic_light_rois_topic_name: str = params.traffic_light_rois_topic_name
        # traffic light lists
        self._traffic_light_roi_msgs: List[TrafficLightRoiArray] = []
        self._traffic_light_label_msgs: List[TrafficLightArray] = []
        # timestamps when there are traffic lights being detected
        self._non_empty_timestamps: Set[float] = set()

        self._topic_list = params.topic_list
        self._mandatory_topics = params.mandatory_topic_list
        self._with_gt_label = params.with_gt_label
        self._gt_label_base = params.gt_label_base

    def convert(self):
        if self._start_timestamp < sys.float_info.epsilon:
            start_timestamp = self._bag_reader.start_timestamp
        else:
            start_timestamp = self._start_timestamp
        start_timestamp = start_timestamp + self._skip_timestamp

        self._save_config()
        if self._with_gt_label:
            self._gt_label_dict = self._read_gt_label()
        scene_timestamp_objects_pair_list = self._preprocess_traffic_lights(start_timestamp)
        self._convert()
        self._add_scene_description("TLR")
        if self._with_gt_label:
            self._add_scene_description("regulatory_element")
        self._convert_traffic_lights(scene_timestamp_objects_pair_list)
        self._save_tables()
        self._annotation_files_generator.save_tables(self._output_anno_dir)

        if self._with_gt_label:
            input_bag_maker = Rosbag2Converter(
                self._input_bag,
                self._output_scene_dir,
                self._topic_list,
                self._start_timestamp,
                self._end_timestamp,
                self._mandatory_topics,
            )
            input_bag_maker.make_input_bag()

    def _read_gt_label(self) -> List[Dict[float, Any]]:
        gt_label_list: Dict[str, Dict[float, Any]] = {}
        for label_file in glob.glob(osp.join(self._gt_label_base, "*yaml")):
            scene_name = osp.splitext(osp.basename(label_file))[0].replace("label_", "")
            with open(label_file, "r") as f:
                label_dict = yaml.safe_load(f)
            gt_label_list[scene_name] = label_dict
        return gt_label_list

    def _preprocess_traffic_lights(self, start_timestamp: float) -> List[Dict[str, Any]]:
        """read object bbox ground truth from rosbag"""
        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        scene_timestamp_objects_pair_list: List[Dict[str, Any]] = []
        if not self._with_gt_label:
            for message in self._bag_reader.read_messages(
                topics=[
                    self._traffic_light_signal_topic_name,
                    self._traffic_light_rois_topic_name,
                ],
                start_time=start_time_in_time,
            ):
                scene_annotation_list = self._process_traffic_lights(message)
                if scene_annotation_list is None:
                    continue
                timestamp = rosbag2_utils.stamp_to_nusc_timestamp(message.header.stamp)
                if scene_annotation_list.__len__() >= 1:
                    self._non_empty_timestamps.add(timestamp)
                scene_timestamp_objects_pair = {
                    "timestamp": timestamp,
                    "scene_annotation_list": scene_annotation_list,
                }
                scene_timestamp_objects_pair_list.append(scene_timestamp_objects_pair)
        else:
            label = self._gt_label_dict[self._bag_name]
            self._non_empty_timestamps = {
                unix_timestamp_to_nusc_timestamp(ts) for ts in label.keys()
            }
            scene_timestamp_objects_pair_list = self._process_tlr_label(label)

        return scene_timestamp_objects_pair_list

    def _convert_image(
        self,
        start_timestamp: float,
        sensor_channel: str,
        topic: str,
        delay_msec: float,
        scene_token: str,
    ):
        """Convert image topic to raw image data.
        This function add the method to the parents function to skip the frame if there is no traffic light label.
        """
        sample_data_token_list: List[str] = []
        sample_records: List[Sample] = self._sample_table.to_records()

        # Get calibrated sensor token
        start_time_in_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
        # For camera: (calibrated_sensor_token, camera_info)
        calibrated_sensor_token, _ = self._generate_calibrated_sensor(
            sensor_channel, start_time_in_time, topic
        )

        if self._sensor_mode != SensorMode.NO_LIDAR:
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

                    nusc_timestamp = rosbag2_utils.stamp_to_nusc_timestamp(image_msg.header.stamp)
                    if self._end_timestamp < nusc_timestamp / 10**6:
                        self._end_timestamp = nusc_timestamp / 10**6
                    if not self._is_traffic_light_label_available(nusc_timestamp):
                        continue
                    sample_data_token = self._generate_image_data(
                        image_msg,
                        rosbag2_utils.stamp_to_unix_timestamp(image_msg.header.stamp),
                        lidar_sample_token,
                        calibrated_sensor_token,
                        sensor_channel,
                        lidar_frame_index,
                        image_shape,
                    )
                    sample_data_token_list.append(sample_data_token)
        else:  # camera only mode

            def get_move_distance(trans1: List[float], trans2: List[float]) -> float:
                dx: float = trans1[0] - trans2[0]
                dy: float = trans1[1] - trans2[1]
                dz: float = trans1[2] - trans2[2]
                return (dx * dx + dy * dy + dz * dz) ** 0.5

            frame_index: int = 0
            generated_frame_index: int = 0

            last_translation: List[float] = [0.0, 0.0, 0.0]
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

                is_data_found = False

                # camera_only mode
                if (frame_index % self._generate_frame_every) == 0:
                    try:
                        ego_pose_token = self._generate_ego_pose(image_msg.header.stamp)
                    except Exception as e:
                        print(e)
                        continue
                    ego_pose: EgoPose = self._ego_pose_table.get_record_from_token(ego_pose_token)
                    translation: Dict[str, float] = ego_pose.translation
                    distance = get_move_distance(translation, last_translation)
                    if distance >= self._generate_frame_every_meter:
                        last_translation = translation

                        nusc_timestamp = rosbag2_utils.stamp_to_nusc_timestamp(
                            image_msg.header.stamp
                        )
                        if self._end_timestamp < nusc_timestamp / 10**6:
                            self._end_timestamp = nusc_timestamp / 10**6
                        if not self._is_traffic_light_label_available(nusc_timestamp):
                            continue
                        sample_token: str = self._sample_table.insert_into_table(
                            timestamp=nusc_timestamp,
                            scene_token=scene_token,
                            next="tmp_token",  # cannot be left empty, will be replaced downstream
                            prev="tmp_token",  # cannot be left empty, will be replaced downstream
                        )
                        is_data_found = True

                    if is_data_found:
                        print(
                            f"frame: {generated_frame_index}, image stamp: {image_unix_timestamp}"
                        )
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

    def _is_traffic_light_label_available(self, timestamp: float) -> bool:
        offset = abs(self._find_nearest_timestamp(timestamp) - timestamp)
        return offset < 75000

    def _find_nearest_timestamp(self, timestamp: float) -> float:
        """find the nearest timestamp in the gt label"""
        if timestamp in self._non_empty_timestamps:
            return timestamp
        else:
            return min(self._non_empty_timestamps, key=lambda x: abs(x - timestamp))

    def _convert_traffic_lights(self, scene_timestamp_objects_pair_list: List[Dict[str, Any]]):
        # use the objects closest to the timestamp of the lidar
        scene_anno_dict: Dict[int, List[Dict[str, Any]]] = {}
        frame_index_to_sample_token: Dict[int, str] = {}
        frame_index_to_sample_data_token: List[Dict[int, str]] = [{}]
        sample_records = self._sample_table.to_records()
        sample_data_records = self._sample_data_table.to_records()
        if sample_data_records.__len__() == 0:
            return

        # generate mask
        mask: List[Dict[int, str]] = [{}]
        first_sample_data: SampleData = sample_data_records[0]
        height = first_sample_data.height
        width = first_sample_data.width
        object_mask = np.zeros(shape=(height, width), dtype=np.uint8)
        object_mask = cocomask.encode(np.asfortranarray(object_mask))
        object_mask["counts"] = repr(base64.b64encode(object_mask["counts"]))[2:]

        def get_sample_idx(sample_records: List[Sample], sample_data: SampleData) -> int | None:
            """get the index of the sample in the sample_records under the following conditions:
            Image data exists and is key frame.
            """
            if sample_data.fileformat != "jpg":
                return None
            if not sample_data.is_key_frame:
                return None
            for idx, sample in enumerate(sample_records):
                if sample.token == sample_data.sample_token:
                    return idx
            raise RuntimeError()

        for sample_data in sample_data_records:
            idx = get_sample_idx(sample_records, sample_data)
            if idx is None:
                continue
            sample = sample_records[idx]

            object_dict = self._get_closest_timestamp(
                scene_timestamp_objects_pair_list,
                sample.timestamp,
            )
            scene_anno_dict[idx] = object_dict["scene_annotation_list"]
            frame_index_to_sample_token[idx] = sample.token
            frame_index_to_sample_data_token[0][idx] = sample_data.token
            mask[0].update({idx: object_mask})

        self._annotation_files_generator.convert_annotations(
            scene_anno_dict=scene_anno_dict,
            frame_index_to_sample_token=frame_index_to_sample_token,
            frame_index_to_sample_data_token=frame_index_to_sample_data_token,
            dataset_name="synthetic",
            mask=mask,
        )

    def _process_traffic_lights(
        self, message: Union[TrafficLightRoiArray, TrafficLightArray]
    ) -> List[Dict[str, Any]]:
        """
        Args:
          message: autoware_auto_perception_msgs.msg.TrafficLightRoiArray
            or autoware_auto_perception_msgs.msg.TrafficSignalArray
        Returns:
            List[Dict[str, Any]]: dict format
        """
        if isinstance(message, TrafficLightRoiArray):
            for label in self._traffic_light_label_msgs:
                if (
                    label.header.stamp.sec == message.header.stamp.sec
                    and label.header.stamp.nanosec == message.header.stamp.nanosec
                ):
                    res = parse_traffic_lights(message, label)
                    self._traffic_light_label_msgs.remove(label)
                    return res
            self._traffic_light_roi_msgs.append(message)
        elif isinstance(message, TrafficLightArray):
            for roi in self._traffic_light_roi_msgs:
                if (
                    roi.header.stamp.sec == message.header.stamp.sec
                    and roi.header.stamp.nanosec == message.header.stamp.nanosec
                ):
                    res = parse_traffic_lights(roi, message)
                    self._traffic_light_roi_msgs.remove(roi)
                    return res
            self._traffic_light_label_msgs.append(message)
        else:
            raise NotImplementedError("Not supported message type!")
        return None

    def _process_tlr_label(self, label_dict: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Args:
          label_dict: dict format
        Returns:
            List[Dict[str, Any]]: dict format
        """
        if label_dict is None:
            return None
        ts_label_list = []
        for ts, annotation in label_dict.items():
            scene_annotation_list = []
            for k, v in annotation[0].items():
                scene_annotation = {}
                scene_annotation["attribute_names"] = []
                scene_annotation["category_name"] = v
                scene_annotation["instance_id"] = k
                scene_annotation["sensor_id"] = 0
                scene_annotation["two_d_box"] = [960, 620, 1920, 1240]  # ToDo: temporary value
                scene_annotation_list.append(scene_annotation)

            timestamp = unix_timestamp_to_nusc_timestamp(ts)
            ts_label_pair = {
                "timestamp": timestamp,
                "scene_annotation_list": scene_annotation_list,
            }
            ts_label_list.append(ts_label_pair)
        return ts_label_list
