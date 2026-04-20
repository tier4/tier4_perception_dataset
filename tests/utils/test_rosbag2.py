import struct

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

from perception_dataset.utils.rosbag2 import pointcloud_msg_to_numpy

POINT_FIELD_FORMATS = {
    PointField.INT8: ("b", 1),
    PointField.UINT8: ("B", 1),
    PointField.INT16: ("h", 2),
    PointField.UINT16: ("H", 2),
    PointField.INT32: ("i", 4),
    PointField.UINT32: ("I", 4),
    PointField.FLOAT32: ("f", 4),
    PointField.FLOAT64: ("d", 8),
}


def _make_pointcloud2(field_defs, points):
    fields = []
    offset = 0
    struct_format = "<"
    for name, datatype in field_defs:
        field_format, field_size = POINT_FIELD_FORMATS[datatype]
        fields.append(PointField(name=name, offset=offset, datatype=datatype, count=1))
        struct_format += field_format
        offset += field_size

    point_step = offset
    data = b"".join(struct.pack(struct_format, *point) for point in points)

    msg = PointCloud2()
    msg.height = 1
    msg.width = len(points)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * len(points)
    msg.data = data
    msg.is_dense = True
    return msg


def test_pointcloud_msg_to_numpy():
    msg = _make_pointcloud2(
        [
            ("x", PointField.FLOAT32),
            ("y", PointField.FLOAT32),
            ("z", PointField.FLOAT32),
            ("i", PointField.UINT8),
            ("channel", PointField.UINT16),
        ],
        [
            (1.0, 2.0, 3.0, 7, 10),
            (4.0, 5.0, 6.0, 8, 11),
        ],
    )

    points = pointcloud_msg_to_numpy(msg)

    assert points.dtype == np.float32
    assert points.shape == (2, 5)
    np.testing.assert_allclose(
        points,
        np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 10.0],
                [4.0, 5.0, 6.0, 8.0, 11.0],
            ],
            dtype=np.float32,
        ),
    )


def test_pointcloud_msg_to_numpy_with_extended_fields():
    msg = _make_pointcloud2(
        [
            ("x", PointField.FLOAT32),
            ("y", PointField.FLOAT32),
            ("z", PointField.FLOAT32),
            ("intensity", PointField.UINT8),
            ("ring", PointField.UINT16),
            ("return_type", PointField.INT8),
            ("timestamp", PointField.FLOAT32),
        ],
        [
            (1.0, 2.0, 3.0, 7, 10, 1, 0.01),
            (4.0, 5.0, 6.0, 8, 11, 2, 0.02),
        ],
    )

    points = pointcloud_msg_to_numpy(msg)

    assert points.dtype == np.float32
    assert points.shape == (2, 7)
    np.testing.assert_allclose(
        points,
        np.array(
            [
                [1.0, 2.0, 3.0, 7.0, 10.0, 1.0, 0.01],
                [4.0, 5.0, 6.0, 8.0, 11.0, 2.0, 0.02],
            ],
            dtype=np.float32,
        ),
    )


def test_compressed_msg_to_numpy():
    # TODO(yukke42): impl test_compressed_msg_to_numpy
    pass


def test_stamp_to_unix_timestamp():
    # TODO(yukke42): impl test_stamp_to_unix_timestamp
    pass


def test_unix_timestamp_to_stamp():
    # TODO(yukke42): impl test_unix_timestamp_to_stamp
    pass


def test_stamp_to_nusc_timestamp():
    # TODO(yukke42): impl test_stamp_to_nusc_timestamp
    pass
