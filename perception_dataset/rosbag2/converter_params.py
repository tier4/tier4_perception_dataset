import enum
from typing import Dict, List, Optional

from pydantic import BaseModel, validator
import yaml

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class DataType(enum.Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class Rosbag2ConverterParams(BaseModel):
    task: str
    input_base: str  # path to the input rosbag2 directory (multiple rosbags in the directory)
    input_bag_path: Optional[str]  # path to the input rosbag2 (a single rosbag)
    output_base: str  # path to the output directory
    gt_label_base: str = ""  # path to the gt labels directory
    overwrite_mode: bool = False
    without_compress: bool = False
    workers_number: int = 1
    with_gt_label: bool = False  # whether to use gt labels
    scene_description: str = ""  # scene description
    accept_frame_drop: bool = False  # whether to accept frame drop

    # rosbag data type
    data_type: DataType = DataType.REAL  # real or synthetic

    # rosbag config
    lidar_sensor: Dict[str, str] = {
        "topic": "",
        "channel": "",
    }  # lidar_sensor, {topic: , channel, }
    radar_sensors: List[Dict[str, str]] = []  # radar sensors
    camera_sensors: List[Dict[str, str]] = []  # camera sensors,
    object_topic_name: str = ""
    object_msg_type: str = ""
    traffic_light_signal_topic_name: str = ""
    traffic_light_rois_topic_name: str = ""
    world_frame_id: str = "map"
    with_camera: bool = True
    generate_bbox_from_cuboid: bool = False

    # rosbag reader
    num_load_frames: int  # the number of frames to be loaded. if the value isn't positive, read all messages.
    skip_timestamp: float  # not read for the second after the first topic
    start_timestamp_sec: float = 0.0  # conversion start timestamp in sec
    crop_frames_unit: int = 1  # crop frames from the end so that the number of frames is divisible by crop_frames_unit. Set to 0 or 1 so as not to crop any frames.
    camera_latency_sec: float = (
        0.0  # camera latency in seconds between the header.stamp and shutter trigger
    )
    timestamp_diff: float = 0.15
    topic_list: list = []  # topic list for input_bag
    # in synthetic data (from AWSIM) it may be the case that there is no ego transform available at the beginning of rosbag
    ignore_no_ego_transform_at_rosbag_beginning: bool = False
    generate_frame_every: int = 1  # pick frames out of every this number.
    generate_frame_every_meter: float = 5.0  # pick frames when ego vehicle moves certain meters

    def __init__(self, **args):
        if "scene_description" in args and isinstance(args["scene_description"], list):
            args["scene_description"] = ", ".join(args["scene_description"])
        if "topic_list" in args and isinstance(args["topic_list"], str):
            with open(args["topic_list"]) as f:
                args["topic_list"] = yaml.safe_load(f)
        if "topic_list" in args and  isinstance(args["topic_list"], dict) and "topic_list" in args["topic_list"]:
            args["topic_list"] = args["topic_list"]["topic_list"]
        super().__init__(**args)

        if len(self.camera_sensors) == 0:
            logger.warning(
                "The config of `camera_sensors` field is empty, so disable to load camera data."
            )
            self.with_camera = False
        self.with_gt_label = self.gt_label_base != ""

    @validator("workers_number")
    def check_workers_number(cls, v):
        if v < 1:
            logger.warning("workers_number must be positive, replaced to 1.")
            v = 1
        return v

    @validator("skip_timestamp")
    def check_skip_timestamp(cls, v):
        if v < 0:
            logger.warning("skip_timestamp must be positive or zero, replaced to 0.")
            v = 0
        return v

    @validator("crop_frames_unit")
    def check_crop_frames_unit(cls, v):
        if v <= 0:
            logger.warning("crop_frames_unit must be positive, replaced to 1.")
            v = 1
        return v

    @validator("object_msg_type")
    def check_object_msg_type(cls, v):
        if v not in ["DetectedObjects", "TrackedObjects", "TrafficLights"]:
            raise ValueError(f"Unexpected object message type: {type(v)}")
        return v
