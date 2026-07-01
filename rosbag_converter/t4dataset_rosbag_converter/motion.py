from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoware_pointcloud_preprocessor.distortion_corrector import DistortionCorrector
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2

from .geometry import RigidTransform
from .geometry import identity


@dataclass(frozen=True)
class UndistortedCloud:
    cloud: PointCloud2
    status: Any | None = None
    timings: Any | None = None


class DistortionBank:
    def __init__(
        self,
        lidar_transforms: dict[str, RigidTransform],
        imu_transforms: dict[str, RigidTransform] | None = None,
        *,
        base_frame: str = "base_link",
        enabled: bool = True,
        use_3d: bool = False,
        use_imu: bool = True,
        update_azimuth_and_distance: bool = False,
    ) -> None:
        self.enabled = enabled
        self.use_imu = use_imu
        self.update_azimuth_and_distance = update_azimuth_and_distance
        self._base_frame = base_frame
        self._imu_transforms = imu_transforms or {}
        self._correctors: dict[str, DistortionCorrector] = {}
        for topic, transform in lidar_transforms.items():
            corrector = DistortionCorrector(use_3d_distortion_correction=use_3d)
            corrector.set_pointcloud_transform(transform.to_msg())
            corrector.set_imu_transform(identity(base_frame, base_frame).to_msg())
            self._correctors[topic] = corrector

    def process_twist(self, msg: TwistWithCovarianceStamped) -> None:
        for corrector in self._correctors.values():
            corrector.process_twist_message(msg)

    def process_imu(self, msg: Imu) -> None:
        frame_id = msg.header.frame_id or self._base_frame
        transform = self._imu_transforms.get(frame_id)
        if transform is None and frame_id == self._base_frame:
            transform = identity(self._base_frame, self._base_frame)
        for corrector in self._correctors.values():
            if transform is not None:
                corrector.set_imu_transform(transform.to_msg())
            corrector.process_imu_message(msg)

    def undistort(self, topic: str, cloud: PointCloud2) -> PointCloud2:
        if not self.enabled:
            return cloud
        corrector = self._correctors[topic]
        out, _status = corrector.undistort_pointcloud(
            cloud,
            use_imu=self.use_imu,
            update_azimuth_and_distance=self.update_azimuth_and_distance,
        )
        return out

    def undistort_with_status(self, topic: str, cloud: PointCloud2) -> UndistortedCloud:
        if not self.enabled:
            return UndistortedCloud(cloud=cloud)
        corrector = self._correctors[topic]
        timed = getattr(corrector, "undistort_pointcloud_timed", None)
        if timed is None:
            out, status = corrector.undistort_pointcloud(
                cloud,
                use_imu=self.use_imu,
                update_azimuth_and_distance=self.update_azimuth_and_distance,
            )
            return UndistortedCloud(cloud=out, status=status)
        out, _status, timings = timed(
            cloud,
            use_imu=self.use_imu,
            update_azimuth_and_distance=self.update_azimuth_and_distance,
        )
        return UndistortedCloud(cloud=out, status=_status, timings=timings)
