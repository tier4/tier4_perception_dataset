import json
from pathlib import Path
import shutil
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from autoware_perception_msgs.msg import (
    DetectedObject,
    DetectedObjectKinematics,
    DetectedObjects,
    ObjectClassification,
    Shape,
    TrackedObject,
    TrackedObjects,
)
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
import numpy as np
from rclpy.serialization import serialize_message
from rosbag2_py import TopicMetadata
from sensor_msgs.msg import CameraInfo, CompressedImage, PointField
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage

from perception_dataset.rosbag2.autoware_msgs import object_classification_to_category_name
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.rosbag2 import create_writer

logger = configure_logger(modname=__name__)

TF_STATIC_QOS = """- history: 3
  depth: 0
  reliability: 1
  durability: 1
  deadline:
    sec: 2147483647
    nsec: 4294967295
  lifespan:
    sec: 2147483647
    nsec: 4294967295
  liveliness: 1
  liveliness_lease_duration:
    sec: 2147483647
    nsec: 4294967295
  avoid_ros_namespace_conventions: false
"""

CATEGORY_NAME_TO_LABEL: Dict[str, int] = {}
for _label in range(30):
    CATEGORY_NAME_TO_LABEL.setdefault(object_classification_to_category_name(_label), _label)

LIDAR_FEATS = ("x", "y", "z", "intensity", "ring")


def _stamp(timestamp_us: int) -> Time:
    ns = int(timestamp_us) * 1000
    return Time(sec=ns // 10**9, nanosec=ns % 10**9)


def _load(annotation_dir: Path, table: str) -> List[Dict[str, Any]]:
    with open(annotation_dir / f"{table}.json") as f:
        return json.load(f)


def _transform(parent: str, child: str, translation, rotation_wxyz, stamp: Time):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(
        float, translation
    )
    q = t.transform.rotation
    q.w, q.x, q.y, q.z = map(float, rotation_wxyz)
    return t


class T4ToRosbag2Converter:
    """Convert T4 datasets (searched recursively under input_base) into rosbag2 files.

    Lossy by design: images stay as the (undistorted) JPEG bytes, point clouds carry
    float32 x/y/z/intensity/ring only, and CameraInfo has zero distortion.
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        overwrite_mode: bool = False,
        storage_id: str = "sqlite3",
        object_topic_name: str = "/ground_truth/objects",
        tracked_object_topic_name: str = "/ground_truth/tracked_objects",
        world_frame_id: str = "map",
        base_frame_id: str = "base_link",
        lidar_sensor: Optional[Dict[str, str]] = None,
        camera_sensors: Optional[List[Dict[str, Any]]] = None,
    ):
        self._input_base = Path(input_base)
        self._output_base = Path(output_base)
        self._overwrite_mode = overwrite_mode
        self._storage_id = storage_id
        self._object_topic_name = object_topic_name
        self._tracked_object_topic_name = tracked_object_topic_name
        self._world_frame_id = world_frame_id
        self._base_frame_id = base_frame_id
        self._lidar_override = lidar_sensor
        self._camera_override = camera_sensors

    def convert(self):
        dataset_dirs = sorted(p.parent.parent for p in self._input_base.rglob("annotation/sample_data.json"))
        if not dataset_dirs:
            raise ValueError(f"No T4 dataset (annotation/sample_data.json) found in {self._input_base}")
        for dataset_dir in dataset_dirs:
            output_dir = self._output_base / dataset_dir.relative_to(self._input_base)
            if output_dir.exists():
                if not self._overwrite_mode:
                    raise ValueError(f"{output_dir} exists. Use --overwrite to replace it.")
                shutil.rmtree(output_dir)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"[BEGIN] {dataset_dir} -> {output_dir}")
            self._convert_dataset(dataset_dir, output_dir)
            logger.info(f"[END] {output_dir}")

    def _resolve_sensors(self, dataset_dir: Path) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        status: Dict[str, Any] = {}
        status_path = dataset_dir / "status.json"
        if status_path.exists():
            with open(status_path) as f:
                status = next(iter(json.load(f).values()), {})
        lidar = self._lidar_override or status.get("_lidar_sensor")
        cameras = self._camera_override or status.get("_camera_sensors")
        if lidar is None or cameras is None:
            raise ValueError(
                f"{dataset_dir}: no status.json sensor info; set lidar_sensor/camera_sensors in the config"
            )
        return lidar, cameras

    def _convert_dataset(self, dataset_dir: Path, output_dir: Path):
        anno_dir = dataset_dir / "annotation"
        lidar, cameras = self._resolve_sensors(dataset_dir)
        sample_data = _load(anno_dir, "sample_data")
        ego_poses = {e["token"]: e for e in _load(anno_dir, "ego_pose")}
        calibrated = {c["token"]: c for c in _load(anno_dir, "calibrated_sensor")}
        sensors = {s["token"]: s for s in _load(anno_dir, "sensor")}
        samples = {s["token"]: s for s in _load(anno_dir, "sample")}
        categories = {c["token"]: c["name"] for c in _load(anno_dir, "category")}
        instances = {i["token"]: i for i in _load(anno_dir, "instance")}
        annotations_by_sample: Dict[str, List[Dict[str, Any]]] = {}
        for ann in _load(anno_dir, "sample_annotation"):
            annotations_by_sample.setdefault(ann["sample_token"], []).append(ann)

        channel_to_topic = {lidar["channel"]: lidar["topic"]}
        channel_to_topic.update({c["channel"]: c["topic"] for c in cameras})

        # (timestamp_ns, topic, type, message factory)
        events: List[Tuple[int, str, str, Callable[[], Any]]] = []
        static_tfs: List[TransformStamped] = []
        seen_ego_tokens = set()
        seen_channels = set()
        first_us = min(sd["timestamp"] for sd in sample_data)

        for sd in sample_data:
            cs = calibrated[sd["calibrated_sensor_token"]]
            channel = sensors[cs["sensor_token"]]["channel"]
            if channel not in channel_to_topic:
                logger.warning(f"Skip channel {channel}: no topic configured")
                continue
            topic = channel_to_topic[channel]
            is_lidar = channel == lidar["channel"]
            frame_id = self._base_frame_id if is_lidar else channel.lower()
            path = dataset_dir / sd["filename"]

            if channel not in seen_channels:
                seen_channels.add(channel)
                if not is_lidar:
                    static_tfs.append(
                        _transform(
                            self._base_frame_id,
                            frame_id,
                            cs["translation"],
                            cs["rotation"],
                            _stamp(first_us),
                        )
                    )

            if sd["ego_pose_token"] not in seen_ego_tokens:
                seen_ego_tokens.add(sd["ego_pose_token"])
                ego = ego_poses[sd["ego_pose_token"]]
                tf = TFMessage(
                    transforms=[
                        _transform(
                            self._world_frame_id,
                            self._base_frame_id,
                            ego["translation"],
                            ego["rotation"],
                            _stamp(ego["timestamp"]),
                        )
                    ]
                )
                events.append((ego["timestamp"] * 1000, "/tf", "tf2_msgs/msg/TFMessage", lambda tf=tf: tf))

            ts_ns = sd["timestamp"] * 1000
            if is_lidar:
                events.append(
                    (ts_ns, topic, "sensor_msgs/msg/PointCloud2",
                     lambda p=path, t=sd["timestamp"], f=frame_id: self._pointcloud_msg(p, t, f))
                )
            else:
                events.append(
                    (ts_ns, topic, "sensor_msgs/msg/CompressedImage",
                     lambda p=path, t=sd["timestamp"], f=frame_id: self._image_msg(p, t, f))
                )
                info_topic = topic.replace("/image_raw/compressed", "/camera_info")
                events.append(
                    (ts_ns, info_topic, "sensor_msgs/msg/CameraInfo",
                     lambda c=cs, s=sd, f=frame_id: self._camera_info_msg(c, s, f))
                )

        for sample in samples.values():
            anns = annotations_by_sample.get(sample["token"], [])
            for topic, msg_type, tracked in (
                (self._object_topic_name, "DetectedObjects", False),
                (self._tracked_object_topic_name, "TrackedObjects", True),
            ):
                events.append(
                    (sample["timestamp"] * 1000, topic,
                     f"autoware_perception_msgs/msg/{msg_type}",
                     lambda s=sample, a=anns, t=tracked: self._objects_msg(
                         s, a, categories, instances, t
                     ))
                )

        if static_tfs:
            events.append(
                (first_us * 1000, "/tf_static", "tf2_msgs/msg/TFMessage",
                 lambda: TFMessage(transforms=static_tfs))
            )

        events.sort(key=lambda e: e[0])
        writer = create_writer(str(output_dir), storage_id=self._storage_id)
        topic_types: Dict[str, str] = {}
        for _ts, topic, msg_type, _factory in events:
            topic_types.setdefault(topic, msg_type)
        for topic, msg_type in topic_types.items():
            writer.create_topic(
                TopicMetadata(
                    name=topic,
                    type=msg_type,
                    serialization_format="cdr",
                    offered_qos_profiles=TF_STATIC_QOS if topic == "/tf_static" else "",
                )
            )
        for ts_ns, topic, _, make_msg in events:
            writer.write(topic, serialize_message(make_msg()), ts_ns)
        del writer

    def _pointcloud_msg(self, path: Path, timestamp_us: int, frame_id: str) -> PointCloud2:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, len(LIDAR_FEATS))
        msg = PointCloud2()
        msg.header = Header(stamp=_stamp(timestamp_us), frame_id=frame_id)
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate(LIDAR_FEATS)
        ]
        msg.is_bigendian = False
        msg.point_step = 4 * len(LIDAR_FEATS)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg

    def _image_msg(self, path: Path, timestamp_us: int, frame_id: str) -> CompressedImage:
        msg = CompressedImage()
        msg.header = Header(stamp=_stamp(timestamp_us), frame_id=frame_id)
        msg.format = "jpeg"
        msg.data = path.read_bytes()
        return msg

    def _camera_info_msg(self, calibrated: Dict[str, Any], sd: Dict[str, Any], frame_id: str) -> CameraInfo:
        k = np.array(calibrated["camera_intrinsic"], dtype=float)
        msg = CameraInfo()
        msg.header = Header(stamp=_stamp(sd["timestamp"]), frame_id=frame_id)
        msg.width, msg.height = sd["width"], sd["height"]
        msg.distortion_model = "plumb_bob"
        msg.d = list(calibrated.get("camera_distortion") or [0.0] * 5)
        msg.k = k.reshape(-1).tolist()
        msg.r = np.eye(3).reshape(-1).tolist()
        msg.p = np.hstack([k, np.zeros((3, 1))]).reshape(-1).tolist()
        return msg

    def _objects_msg(
        self,
        sample: Dict[str, Any],
        annotations: List[Dict[str, Any]],
        categories: Dict[str, str],
        instances: Dict[str, Dict[str, Any]],
        tracked: bool,
    ):
        msg = TrackedObjects() if tracked else DetectedObjects()
        msg.header = Header(stamp=_stamp(sample["timestamp"]), frame_id=self._world_frame_id)
        for ann in annotations:
            obj = TrackedObject() if tracked else DetectedObject()
            if tracked:
                obj.object_id.uuid = list(uuid.UUID(hex=ann["instance_token"]).bytes)
            obj.existence_probability = 1.0
            category = categories[instances[ann["instance_token"]]["category_token"]]
            obj.classification = [
                ObjectClassification(
                    label=CATEGORY_NAME_TO_LABEL.get(category, 0), probability=1.0
                )
            ]
            pose = obj.kinematics.pose_with_covariance.pose
            pose.position.x, pose.position.y, pose.position.z = map(float, ann["translation"])
            q = pose.orientation
            q.w, q.x, q.y, q.z = map(float, ann["rotation"])
            obj.kinematics.orientation_availability = DetectedObjectKinematics.AVAILABLE
            if ann.get("velocity"):
                lin = obj.kinematics.twist_with_covariance.twist.linear
                lin.x, lin.y, lin.z = map(float, ann["velocity"])
            if tracked and ann.get("acceleration"):
                acc = obj.kinematics.acceleration_with_covariance.accel.linear
                acc.x, acc.y, acc.z = map(float, ann["acceleration"])
            obj.shape.type = Shape.BOUNDING_BOX
            # T4 size is [width, length, height]; Autoware dimensions are (x=length, y=width, z=height)
            width, length, height = map(float, ann["size"])
            obj.shape.dimensions.x, obj.shape.dimensions.y, obj.shape.dimensions.z = (
                length,
                width,
                height,
            )
            msg.objects.append(obj)
        return msg
