from __future__ import annotations

from array import array

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

POINT_STEP = 32
FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
    PointField(name="return_type", offset=13, datatype=PointField.UINT8, count=1),
    PointField(name="channel", offset=14, datatype=PointField.UINT16, count=1),
    PointField(name="azimuth", offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name="elevation", offset=20, datatype=PointField.FLOAT32, count=1),
    PointField(name="distance", offset=24, datatype=PointField.FLOAT32, count=1),
    PointField(name="time_stamp", offset=28, datatype=PointField.UINT32, count=1),
]

POINT_DTYPE = np.dtype(
    {
        "names": [
            "x",
            "y",
            "z",
            "intensity",
            "return_type",
            "channel",
            "azimuth",
            "elevation",
            "distance",
            "time_stamp",
        ],
        "formats": [
            "<f4",
            "<f4",
            "<f4",
            "u1",
            "u1",
            "<u2",
            "<f4",
            "<f4",
            "<f4",
            "<u4",
        ],
        "offsets": [0, 4, 8, 12, 13, 14, 16, 20, 24, 28],
        "itemsize": POINT_STEP,
    }
)


def stamp_from_seconds(timestamp: float) -> Time:
    msg = Time()
    msg.sec = int(timestamp)
    msg.nanosec = int(round((timestamp - msg.sec) * 1e9))
    if msg.nanosec >= 1_000_000_000:
        msg.sec += 1
        msg.nanosec -= 1_000_000_000
    return msg


def stamp_to_seconds(stamp: Time) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def nebula_array_to_pointcloud(points: np.ndarray, *, stamp: Time, frame_id: str) -> PointCloud2:
    if points is None or points.size == 0:
        arr = np.zeros(0, dtype=POINT_DTYPE)
    else:
        points = np.asarray(points)
        arr = np.zeros(points.shape[0], dtype=POINT_DTYPE)
        arr["x"] = points[:, 0].astype(np.float32)
        arr["y"] = points[:, 1].astype(np.float32)
        arr["z"] = points[:, 2].astype(np.float32)
        arr["intensity"] = np.clip(points[:, 3], 0, 255).astype(np.uint8)
        arr["return_type"] = np.clip(points[:, 4], 0, 255).astype(np.uint8)
        arr["channel"] = np.clip(points[:, 5], 0, np.iinfo(np.uint16).max).astype(np.uint16)
        arr["azimuth"] = points[:, 6].astype(np.float32)
        arr["elevation"] = points[:, 7].astype(np.float32)
        arr["distance"] = points[:, 8].astype(np.float32)
        arr["time_stamp"] = np.clip(points[:, 9], 0, np.iinfo(np.uint32).max).astype(np.uint32)
    return structured_array_to_pointcloud(arr, stamp=stamp, frame_id=frame_id)


def structured_array_to_pointcloud(arr: np.ndarray, *, stamp: Time, frame_id: str) -> PointCloud2:
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(arr.shape[0])
    msg.fields = FIELDS
    msg.is_bigendian = False
    msg.point_step = POINT_STEP
    msg.row_step = POINT_STEP * msg.width
    msg.is_dense = True
    msg.data = array("B", arr.astype(POINT_DTYPE, copy=False).tobytes())
    return msg


def pointcloud_to_structured_array(cloud: PointCloud2) -> np.ndarray:
    if cloud.point_step != POINT_STEP:
        raise ValueError(f"Expected point_step {POINT_STEP}, got {cloud.point_step}")
    arr = np.frombuffer(cloud.data, dtype=POINT_DTYPE, count=cloud.width * cloud.height)
    return arr.copy()


_POINT_FIELD_FORMATS = {
    PointField.INT8: "i1",
    PointField.UINT8: "u1",
    PointField.INT16: "<i2",
    PointField.UINT16: "<u2",
    PointField.INT32: "<i4",
    PointField.UINT32: "<u4",
    PointField.FLOAT32: "<f4",
    PointField.FLOAT64: "<f8",
}


def normalize_pointcloud_layout(cloud: PointCloud2) -> PointCloud2:
    if cloud.point_step == POINT_STEP and {field.name for field in cloud.fields} >= set(POINT_DTYPE.names):
        return cloud

    count = int(cloud.width) * int(cloud.height)
    arr = np.zeros(count, dtype=POINT_DTYPE)
    fields = {field.name: field for field in cloud.fields}
    for name in POINT_DTYPE.names:
        field = fields.get(name)
        if field is None or field.count != 1:
            continue
        fmt = _POINT_FIELD_FORMATS.get(field.datatype)
        if fmt is None:
            continue
        source_dtype = np.dtype(
            {
                "names": [name],
                "formats": [fmt],
                "offsets": [field.offset],
                "itemsize": cloud.point_step,
            }
        )
        values = np.frombuffer(cloud.data, dtype=source_dtype, count=count)[name]
        target = arr[name]
        if np.issubdtype(target.dtype, np.integer):
            info = np.iinfo(target.dtype)
            values = np.clip(values, info.min, info.max)
        arr[name] = values.astype(target.dtype, copy=False)
    return structured_array_to_pointcloud(arr, stamp=cloud.header.stamp, frame_id=cloud.header.frame_id)


def pointcloud_to_lidar_features(cloud: PointCloud2, *, num_lidar_feats: int = 7) -> np.ndarray:
    if num_lidar_feats not in (5, 7):
        raise ValueError(f"num_lidar_feats must be 5 or 7, got {num_lidar_feats}")
    if cloud.point_step != POINT_STEP:
        raise ValueError(f"Expected point_step {POINT_STEP}, got {cloud.point_step}")
    arr = np.frombuffer(cloud.data, dtype=POINT_DTYPE, count=cloud.width * cloud.height)
    columns = [
        arr["x"],
        arr["y"],
        arr["z"],
        arr["intensity"],
        arr["channel"],
    ]
    if num_lidar_feats == 7:
        columns.extend([arr["return_type"], arr["time_stamp"]])
    return np.column_stack(columns).astype(np.float32, copy=False)


def transform_pointcloud_array(arr: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if out.size == 0:
        return out
    xyz = np.column_stack([out["x"], out["y"], out["z"], np.ones(out.size, dtype=np.float32)])
    transformed = (matrix @ xyz.T).T
    out["x"] = transformed[:, 0].astype(np.float32)
    out["y"] = transformed[:, 1].astype(np.float32)
    out["z"] = transformed[:, 2].astype(np.float32)
    return out
