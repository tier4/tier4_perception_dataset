from __future__ import annotations

from bisect import bisect_left
from typing import Iterable

from rclpy.duration import Duration
from tf2_msgs.msg import TFMessage
from tf2_py import BufferCore

from .camera_calibration import CameraCalibration
from .calibration import CalibrationSet
from .geometry import RigidTransform
from .geometry import quat_from_rpy
from .pointcloud import stamp_from_seconds
from .rosbag import transform_stamped_to_rigid


class TfManager:
    def __init__(
        self,
        calibration: CalibrationSet,
        camera_calibrations: Iterable[CameraCalibration] = (),
        cache_time_sec: float = 24.0 * 60.0 * 60.0,
    ) -> None:
        self._buffer = BufferCore(Duration(seconds=float(cache_time_sec)))
        self._authority = "t4dataset_rosbag_converter"
        self._dynamic_times: list[float] = []
        self._dynamic_times_sorted = True
        for transform in (calibration.transforms or {}).values():
            self.add_static_transform(transform)
        for camera in camera_calibrations:
            self.add_camera_optical_transform(camera)

    def add_tf_message(self, msg: TFMessage, *, is_static: bool) -> None:
        for transform in msg.transforms:
            if is_static:
                self._buffer.set_transform_static(transform, self._authority)
            else:
                self._buffer.set_transform(transform, self._authority)
                self._dynamic_times.append(_stamp_to_seconds(transform.header.stamp))
                self._dynamic_times_sorted = False

    def add_static_transform(self, transform: RigidTransform) -> None:
        self._buffer.set_transform_static(transform.to_msg(), self._authority)

    def add_camera_optical_transform(self, camera: CameraCalibration) -> None:
        if camera.link_frame_id == camera.frame_id:
            return
        transform = RigidTransform(
            parent=camera.link_frame_id,
            child=camera.frame_id,
            translation=(0.0, 0.0, 0.0),
            rotation_xyzw=quat_from_rpy(-1.5707963267948966, 0.0, -1.5707963267948966),
        )
        self.add_static_transform(transform)

    def lookup(self, target_frame: str, source_frame: str, timestamp: float) -> RigidTransform | None:
        try:
            msg = self._buffer.lookup_transform_core(
                target_frame,
                source_frame,
                stamp_from_seconds(timestamp),
            )
        except Exception:
            return None
        return transform_stamped_to_rigid(msg)

    def lookup_required(self, target_frame: str, source_frame: str, timestamp: float) -> RigidTransform:
        transform = self.lookup(target_frame, source_frame, timestamp)
        if transform is None:
            raise KeyError(f"No transform from {target_frame} to {source_frame} at {timestamp:.6f}")
        return transform

    def lookup_nearest(
        self,
        target_frame: str,
        source_frame: str,
        timestamp: float,
        *,
        tolerance_sec: float,
    ) -> tuple[RigidTransform | None, float | None]:
        transform = self.lookup(target_frame, source_frame, timestamp)
        if transform is not None:
            return transform, timestamp
        for candidate_time in self._nearest_dynamic_times(timestamp, tolerance_sec):
            transform = self.lookup(target_frame, source_frame, candidate_time)
            if transform is not None:
                return transform, candidate_time
        return None, None

    def _nearest_dynamic_times(self, timestamp: float, tolerance_sec: float) -> list[float]:
        if not self._dynamic_times:
            return []
        if not self._dynamic_times_sorted:
            self._dynamic_times = sorted(set(self._dynamic_times))
            self._dynamic_times_sorted = True
        index = bisect_left(self._dynamic_times, timestamp)
        candidates: list[float] = []
        for step in range(16):
            before = index - 1 - step
            after = index + step
            if before >= 0:
                candidates.append(self._dynamic_times[before])
            if after < len(self._dynamic_times):
                candidates.append(self._dynamic_times[after])
        return [
            candidate
            for candidate in sorted(candidates, key=lambda value: abs(value - timestamp))
            if abs(candidate - timestamp) <= tolerance_sec
        ]


def _stamp_to_seconds(stamp) -> float:
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
