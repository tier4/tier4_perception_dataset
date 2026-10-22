from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LidarTopicConfig:
    name: str
    msg_type: str
    is_reset_topic: bool = False
    sensor_model: str = "PandarQT128"
    return_mode: str = "LastStrongest"
    min_range: float | None = None
    max_range: float | None = None
    cloud_min_angle: int | None = None
    cloud_max_angle: int | None = None
    cut_angle: float | None = None
    dual_return_distance_threshold: float = 0.1

    @property
    def lidar_name(self) -> str:
        parts = self.name.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "sensing" and parts[1] == "lidar":
            return parts[2]
        return parts[-2] if len(parts) > 1 else self.name.strip("/")


@dataclass(frozen=True)
class ConverterConfig:
    raw: dict[str, Any]
    scene_description: str
    conversion: dict[str, Any]
    lidar_topics: list[LidarTopicConfig]
    allow_concat_drop: bool
    concat_options: dict[str, Any]
    distortion_options: dict[str, Any]
    other_topics: list[dict[str, Any]]
    filter_topics: list[str]


def _find_task(raw: dict[str, Any], task_type: str) -> dict[str, Any] | None:
    for section in ("preprocess", "annotation", "postprocess"):
        for task in raw.get(section, {}).get("tasks", []):
            if task.get("type") == task_type:
                return task
    return None


def _extract_topic_list(config: dict[str, Any]) -> list[str]:
    for key in ("topic_list", "topics"):
        value = config.get(key)
        if isinstance(value, dict):
            topics = _extract_topic_list(value)
            if topics:
                return topics
        elif isinstance(value, list):
            return [topic for topic in value if isinstance(topic, str) and topic]
    return []


def _unique_topics(topics: list[str]) -> list[str]:
    return list(dict.fromkeys(topics))


def _concat_options(config: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    strategy_config = config.get("matching_strategy", {}) or {}
    strategy = (
        strategy_config.get("type", strategy_config)
        if isinstance(strategy_config, dict)
        else strategy_config
    )
    options: dict[str, Any] = {}
    param_file = config.get("param_file") or config.get("concat_param_file")
    if param_file:
        param_path = Path(param_file)
        if not param_path.is_absolute():
            param_path = (base_dir / param_path).resolve()
        options["param_file"] = str(param_path)
    if strategy or config.get("matching_strategy_type"):
        options["matching_strategy"] = str(strategy or config["matching_strategy_type"])
    if "timeout_sec" in config:
        options["timeout_sec"] = float(config["timeout_sec"])
    if "is_motion_compensated" in config:
        options["is_motion_compensated"] = bool(config["is_motion_compensated"])
    if "output_timestamp_offset_sec" in config:
        options["output_timestamp_offset_sec"] = float(config["output_timestamp_offset_sec"])
    for key in (
        "lidar_timestamp_offsets",
        "lidar_timestamp_noise_window",
        "input_timestamp_offsets",
    ):
        value = config.get(key)
        if value is None and isinstance(strategy_config, dict):
            value = strategy_config.get(key)
        if value is not None:
            options[key] = dict(value) if isinstance(value, dict) else list(value)
    return options


def _load_ros_param_file(param_file: str | Path) -> dict[str, Any]:
    with Path(param_file).open("r") as fp:
        raw = yaml.safe_load(fp) or {}
    return raw.get("/**", {}).get("ros__parameters", raw.get("ros__parameters", raw))


def _resolve_optional_path(value: Any, *, base_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _distortion_options(config: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    options: dict[str, Any] = {
        "input_imu_topic": "/sensing/imu/imu_data",
        "input_twist_topic": "/sensing/vehicle_velocity_converter/twist_with_covariance",
    }
    param_file = _resolve_optional_path(
        config.get("distortion_param_file") or config.get("distortion_corrector_param_file"),
        base_dir=base_dir,
    )
    if param_file:
        options["param_file"] = param_file
        params = _load_ros_param_file(param_file)
        for key in ("use_imu", "use_3d_distortion_correction", "update_azimuth_and_distance"):
            if key in params:
                options[key] = params[key]
    for key in (
        "enabled",
        "use_imu",
        "use_3d_distortion_correction",
        "update_azimuth_and_distance",
    ):
        if key in config:
            options[key] = config[key]
    for key in ("input_imu_topic", "input_twist_topic"):
        if key in config:
            options[key] = config[key]
    return options


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def load_config(path: str | Path) -> ConverterConfig:
    config_path = Path(path)
    with config_path.open("r") as fp:
        raw = yaml.safe_load(fp)

    inject_task = _find_task(raw, "inject_concatenated_pointcloud")
    if inject_task is None:
        raise ValueError("Config is missing preprocess task: inject_concatenated_pointcloud")
    inject_config = inject_task.get("config", {})
    lidar_topics = [
        LidarTopicConfig(
            name=item["name"],
            msg_type=item.get("type", "pandar_msgs/msg/PandarScan"),
            is_reset_topic=bool(item.get("is_reset_topic", False)),
            sensor_model=item.get("sensor_model", "PandarQT128"),
            return_mode=item.get("return_mode", "LastStrongest"),
            min_range=_optional_float(item.get("min_range")),
            max_range=_optional_float(item.get("max_range")),
            cloud_min_angle=_optional_int(item.get("cloud_min_angle")),
            cloud_max_angle=_optional_int(item.get("cloud_max_angle")),
            cut_angle=_optional_float(item.get("cut_angle")),
            dual_return_distance_threshold=float(item.get("dual_return_distance_threshold", 0.1)),
        )
        for item in inject_config.get("lidar_topics", [])
    ]
    if not lidar_topics:
        raise ValueError("inject_concatenated_pointcloud.config.lidar_topics must not be empty")
    if not any(topic.is_reset_topic for topic in lidar_topics):
        lidar_topics = [
            lidar_topics[0].__class__(**{**lidar_topics[0].__dict__, "is_reset_topic": True})
        ] + lidar_topics[1:]

    t4_task = _find_task(raw, "convert_rosbag2_to_non_annotated_t4")
    if t4_task is None:
        raise ValueError("Config is missing preprocess task: convert_rosbag2_to_non_annotated_t4")
    t4_config = t4_task.get("config", {})
    conversion = dict(t4_config.get("conversion", {}))
    description = t4_config.get("description", {})
    scene_description = description.get("scene", "") if isinstance(description, dict) else ""

    filter_task = _find_task(raw, "filter_and_slice_rosbag")
    filter_config = (filter_task.get("config", {}) or {}) if filter_task is not None else {}
    deepen_task = _find_task(raw, "convert_deepen_to_t4")
    deepen_config = (deepen_task.get("config", {}) or {}) if deepen_task is not None else {}
    deepen_conversion = deepen_config.get("conversion", {}) or {}
    filter_topics = (
        _extract_topic_list(deepen_conversion)
        or _extract_topic_list(conversion)
        or _extract_topic_list(filter_config)
    )

    return ConverterConfig(
        raw=raw,
        scene_description=scene_description,
        conversion=conversion,
        lidar_topics=lidar_topics,
        allow_concat_drop=bool(inject_config.get("allow_concat_drop", False)),
        concat_options=_concat_options(inject_config, base_dir=config_path.parent),
        distortion_options=_distortion_options(inject_config, base_dir=config_path.parent),
        other_topics=list(inject_config.get("other_topics", [])),
        filter_topics=_unique_topics(filter_topics),
    )
