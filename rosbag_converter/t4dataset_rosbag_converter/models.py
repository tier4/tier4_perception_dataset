from __future__ import annotations

from dataclasses import dataclass

from .geometry import RigidTransform


@dataclass(frozen=True)
class LidarSource:
    packet_topic: str
    info_topic: str
    lidar_name: str
    frame_id: str
    is_reset_topic: bool
    base_to_sensor: RigidTransform
    sensor_model: str
    return_mode: str
    min_range: float | None
    max_range: float | None
    cloud_min_angle: int | None
    cloud_max_angle: int | None
    cut_angle: float | None
    dual_return_distance_threshold: float
