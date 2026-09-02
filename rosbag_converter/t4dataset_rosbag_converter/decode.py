from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nebula_lib import HesaiOfflineDecoder, PandarScan
from sensor_msgs.msg import PointCloud2

from .geometry import RigidTransform
from .pointcloud import nebula_array_to_pointcloud, stamp_from_seconds


@dataclass
class LidarDecoder:
    lidar_name: str
    packet_topic: str
    frame_id: str
    calibration_csv: Path
    base_to_lidar: RigidTransform
    sensor_model: str = "PandarQT128"
    return_mode: str = "LastStrongest"
    min_range: float | None = None
    max_range: float | None = None
    cloud_min_angle: int | None = None
    cloud_max_angle: int | None = None
    cut_angle: float | None = None
    dual_return_distance_threshold: float = 0.1

    def __post_init__(self) -> None:
        self.decoder = HesaiOfflineDecoder(
            sensor_model=self.sensor_model,
            calibration_csv=str(self.calibration_csv),
            return_mode=self.return_mode,
            min_range=self.min_range,
            max_range=self.max_range,
            cloud_min_angle=self.cloud_min_angle,
            cloud_max_angle=self.cloud_max_angle,
            cut_angle=self.cut_angle,
            dual_return_distance_threshold=self.dual_return_distance_threshold,
        )
        self._packet_decode_errors = 0

    def decode_scan_cdr(self, serialized: bytes) -> PointCloud2 | None:
        scan = PandarScan.from_cdr(serialized)
        last_cloud = None
        for packet in getattr(scan, "packets", []):
            packet_data = bytes(getattr(packet, "data"))
            packet_size = int(getattr(packet, "size", len(packet_data)))
            try:
                metadata, points = self.decoder.decode_packet(packet_data[:packet_size])
            except RuntimeError as exc:
                self._packet_decode_errors += 1
                if self._packet_decode_errors <= 3:
                    print(
                        f"Skipping undecodable packet on {self.packet_topic} "
                        f"with {self.sensor_model}: {exc}"
                    )
                continue
            if points is None:
                continue
            stamp = _cloud_stamp(scan, metadata)
            last_cloud = nebula_array_to_pointcloud(points, stamp=stamp, frame_id=self.frame_id)
        return last_cloud


def _cloud_stamp(scan: Any, metadata: dict[str, Any]):
    frame_timestamp = metadata.get("frame_timestamp_s")
    if frame_timestamp is not None:
        return stamp_from_seconds(float(frame_timestamp))
    packet_timestamp = metadata.get("packet_timestamp_ns")
    if packet_timestamp is not None:
        return stamp_from_seconds(float(packet_timestamp) * 1e-9)
    header = getattr(scan, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and (getattr(stamp, "sec", 0) or getattr(stamp, "nanosec", 0)):
        return stamp_from_seconds(
            float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
        )
    return stamp_from_seconds(0.0)
