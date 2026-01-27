import enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, field_validator, model_validator
import yaml

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class DataType(enum.Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class BaseModelWithDictAccess(BaseModel):
    def __getitem__(self, key: str):
        """Make the model subscriptable by key."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow setting values using dictionary-style syntax."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the model."""
        return hasattr(self, key)

    def get(self, key: str, default=None):
        """Get a value by key with a default fallback, just like a dict."""
        return getattr(self, key, default)


class LidarSourceMapping(BaseModel):
    """Model for individual lidar source mapping entry."""

    topic: str  # e.g., "/sensing/lidar/rear_upper/pointcloud_before_sync"
    channel: str  # e.g., "LIDAR_REAR_UPPER"
    frame_id: str  # e.g., "rear_upper/lidar_base_link"


class LidarSensor(BaseModelWithDictAccess):
    topic: Optional[str] = None  # e.g., "/lidar_points"
    channel: Optional[str] = None  # e.g., "LIDAR_TOP"
    lidar_info_topic: Optional[str] = None  # topic for lidar info, e.g., "/lidar_info"
    lidar_info_channel: Optional[str] = None  # channel for lidar info, e.g., "LIDAR_INFO"
    accept_no_info: Optional[bool] = (
        None  # if True, the conversion will continue even if no lidar_info message is found for a point cloud timestamp.
    )
    lidar_sources_mapping: Optional[List[LidarSourceMapping]] = (
        None  # list of lidar source mappings with topic, channel, and frame_id
    )

    @model_validator(mode="after")
    def check_lidar_info_fields(self):
        """Validate that if any lidar_info field is defined, all must be defined."""
        fields = {
            "lidar_info_topic": _check_check_lidar_info_field(self.lidar_info_topic),
            "lidar_info_channel": _check_check_lidar_info_field(self.lidar_info_channel),
            "accept_no_info": _check_check_lidar_info_field(self.accept_no_info),
            "lidar_sources_mapping": _check_check_lidar_info_field(self.lidar_sources_mapping),
        }

        defined_fields = [name for name, is_defined in fields.items() if is_defined]

        if defined_fields and len(defined_fields) != len(fields):
            undefined_fields = [name for name, is_defined in fields.items() if not is_defined]
            raise ValueError(
                "If any of lidar_info_topic, lidar_info_channel, accept_no_info, or lidar_sources_mapping is defined, all must be defined. "
                f"Defined: {defined_fields}. Undefined: {undefined_fields}"
            )

        return self


def _check_check_lidar_info_field(
    value: Union[str, bool, list[LidarSourceMapping] | None],
) -> bool:
    if value is None:
        return False
    if type(value) is list:
        return len(value) > 0
    return True


class Rosbag2ConverterParams(BaseModelWithDictAccess):
    task: str
    input_base: str  # path to the input rosbag2 directory (multiple rosbags in the directory)
    input_bag_path: Optional[str] = None  # path to the input rosbag2 (a single rosbag)
    output_base: str  # path to the output directory
    gt_label_base: str = ""  # path to the gt labels directory
    overwrite_mode: bool = False
    without_compress: bool = False
    raise_exception: bool = False
    workers_number: int = 1
    with_gt_label: bool = False  # whether to use gt labels
    scene_description: str = ""  # scene description
    accept_frame_drop: bool = False  # whether to accept frame drop
    undistort_image: bool = False  # whether to undistort image
    make_t4_dataset_dir: bool = True  # whether to make t4 dataset directory

    # rosbag data type
    data_type: DataType = DataType.REAL  # real or synthetic

    # rosbag config
    lidar_sensor: LidarSensor = LidarSensor()
    radar_sensors: List[Dict[str, str]] = []  # radar sensors
    camera_sensors: List[Dict[str, Union[str, float]]] = []  # camera sensors,
    object_topic_name: str = ""
    object_msg_type: str = ""
    traffic_light_signal_topic_name: str = ""
    traffic_light_rois_topic_name: str = ""
    world_frame_id: str = "map"
    with_camera: bool = True
    generate_bbox_from_cuboid: bool = False

    # rosbag reader
    num_load_frames: (
        int  # the number of frames to be loaded. if the value isn't positive, read all messages.
    )
    skip_timestamp: float  # not read for the second after the first topic
    start_timestamp_sec: float = 0.0  # conversion start timestamp in sec
    crop_frames_unit: int = (
        1  # crop frames from the end so that the number of frames is divisible by crop_frames_unit. Set to 0 or 1 so as not to crop any frames.
    )

    # Maximum camera jitter in seconds. This value MUST be set large enough since the camera jitter smaller than this value is not considererd.
    # Also, do not set this value larger than system_scan_period_sec.
    max_camera_jitter_sec: float = 0.005

    lidar_latency_sec: float = (
        0.005  # lidar latency in seconds between the header.stamp and shutter trigger
    )
    system_scan_period_sec: float = 0.1  # LiDAR scan period in seconds
    camera_scan_period_sec: float = 0.1  # Camera scan period in seconds
    topic_list: list = []  # topic list for input_bag
    mandatory_topic_list: list = []  # mandatory topic list for input_bag

    lidar_points_ratio_threshold: float = (
        0.2  # ratio of lidar points to be used proportion to the maximum number of lidar points in a frame
    )

    # in synthetic data (from AWSIM) it may be the case that there is no ego transform available at the beginning of rosbag
    ignore_no_ego_transform_at_rosbag_beginning: bool = False
    generate_frame_every: int = 1  # pick frames out of every this number.
    generate_frame_every_meter: float = 5.0  # pick frames when ego vehicle moves certain meters

    # for Co-MLOps
    with_ins: bool = False  # whether to convert rosbag with INS topics
    with_vehicle_status: bool = False  # whether to convert rosbag with vehicle status

    def __init__(self, **args):
        if "scene_description" in args and isinstance(args["scene_description"], list):
            args["scene_description"] = ", ".join(args["scene_description"])
        if "camera_scan_period_sec" not in args and "system_scan_period_sec" in args:
            args["camera_scan_period_sec"] = args["system_scan_period_sec"]
        if "topic_list" in args and isinstance(args["topic_list"], str):
            with open(args["topic_list"]) as f:
                args["topic_list"] = yaml.safe_load(f)
        if (
            "topic_list" in args
            and isinstance(args["topic_list"], dict)
            and "topic_list" in args["topic_list"]
        ):
            topic_list_dict = args["topic_list"]
            args["topic_list"] = topic_list_dict["topic_list"]
            if "mandatory_topic_list" in topic_list_dict.keys():
                args["mandatory_topic_list"] = topic_list_dict["mandatory_topic_list"]
        super().__init__(**args)

        if len(self.camera_sensors) == 0:
            logger.warning(
                "The config of `camera_sensors` field is empty, so disable to load camera data."
            )
            self.with_camera = False
        self.with_gt_label = self.gt_label_base != ""

    @field_validator("workers_number")
    def check_workers_number(cls, v):
        if v < 1:
            logger.warning("workers_number must be positive, replaced to 1.")
            v = 1
        return v

    @field_validator("skip_timestamp")
    def check_skip_timestamp(cls, v):
        if v < 0:
            logger.warning("skip_timestamp must be positive or zero, replaced to 0.")
            v = 0
        return v

    @field_validator("crop_frames_unit")
    def check_crop_frames_unit(cls, v):
        if v <= 0:
            logger.warning("crop_frames_unit must be positive, replaced to 1.")
            v = 1
        return v

    @field_validator("object_msg_type")
    def check_object_msg_type(cls, v):
        if v not in ["DetectedObjects", "TrackedObjects", "TrafficLights"]:
            raise ValueError(f"Unexpected object message type: {type(v)}")
        return v
