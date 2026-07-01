from __future__ import annotations

from pathlib import Path

from sensor_msgs.msg import PointCloud2


def save_lidar_pointcloud_pcd(
    path: Path,
    cloud: PointCloud2,
    *,
    num_lidar_feats: int = 7,
    encoding: str = "binary",
) -> None:
    if num_lidar_feats not in (5, 7):
        raise ValueError(f"num_lidar_feats must be 5 or 7, got {num_lidar_feats}")
    from pypcd4 import Encoding
    from pypcd4 import PointCloud

    path.parent.mkdir(parents=True, exist_ok=True)
    parsed = PointCloud.from_msg(cloud)
    arrays = [
        parsed.pc_data["x"],
        parsed.pc_data["y"],
        parsed.pc_data["z"],
        parsed.pc_data["intensity"],
        parsed.pc_data["channel"],
    ]
    if num_lidar_feats == 7:
        arrays.extend([parsed.pc_data["return_type"], parsed.pc_data["time_stamp"]])
    fields = ("x", "y", "z", "intensity", "ring", "return_type", "time_stamp")[: len(arrays)]
    types = ("<f4", "<f4", "<f4", "u1", "<u2", "u1", "<u4")[: len(arrays)]
    PointCloud.from_points(arrays, fields, types).save(
        path,
        encoding=Encoding(encoding),
    )
