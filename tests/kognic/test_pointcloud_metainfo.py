import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perception_dataset.kognic.openlabel_to_t4_converter import _lidar_point_count
from perception_dataset.utils.pointcloud import extract_pointclouds


@pytest.fixture(params=["count", "LIDAR_CONCAT", "LIDAR_FRONT"])
def process_cloud(request, tmp_path: Path):
    """Exercise metadata validation through point counting and both export paths."""

    def process(num_points, ranges, num_pts_feats=5):
        points = np.arange(num_points * num_pts_feats, dtype=np.float32).reshape(
            num_points, num_pts_feats
        )
        bin_path = tmp_path / "cloud.pcd.bin"
        info_path = tmp_path / "info.json"
        points.tofile(bin_path)
        stamp = {"sec": 1, "nanosec": 0}
        info_path.write_text(
            json.dumps({
                "stamp": stamp,
                "num_pts_feats": num_pts_feats,
                "sources": [
                    {
                        "sensor_token": f"sensor-{i}",
                        "idx_begin": start,
                        "length": length,
                        "stamp": stamp,
                    }
                    for i, (start, length) in enumerate(ranges)
                ],
            })
        )
        if request.param == "count":
            return _lidar_point_count(bin_path, info_path)
        record = SimpleNamespace(
            filename=bin_path.name, info_filename=info_path.name, timestamp=1_000_000
        )
        extract_pointclouds(
            tmp_path,
            tmp_path / "out",
            request.param,
            [{"LIDAR_CONCAT": record}],
            {request.param: "sensor-0"},
        )
        csv_path = tmp_path / "out" / "lidar" / request.param / "1000000000.csv"
        rows = csv_path.read_text().splitlines()[1:]
        selected = (
            points
            if request.param == "LIDAR_CONCAT"
            else points[ranges[0][0] : ranges[0][0] + ranges[0][1]]
        )
        if len(selected):
            np.testing.assert_allclose(
                [[float(value) for value in row.split(",")[1:]] for row in rows],
                selected[:, :4],
            )
        return len(rows)

    return process


@pytest.mark.parametrize("num_pts_feats", [5, 7])
def test_metainfo_controls_point_layout(process_cloud, num_pts_feats):
    """Load the declared binary schema and tolerate empty sensor contributions."""
    assert process_cloud(3, [(0, 3), (0, 0)], num_pts_feats) == 3


def test_empty_cloud_with_empty_source(process_cloud):
    """Keep empty frames valid in counting and CSV export."""
    assert process_cloud(0, [(0, 0)]) == 0


@pytest.mark.parametrize(
    ("num_points", "ranges", "message"),
    [
        (3, [(0, 2), (1, 1)], "Overlap detected"),
        (3, [(0, 1), (2, 1)], "Gap detected"),
        (3, [(0, 2)], "Incomplete coverage"),
        (3, [(0, 4)], "exceeds point cloud size"),
        (3, [(-1, 3)], "negative idx_begin"),
        (3, [(0, -1)], "negative length"),
        (0, [(0, 1)], "exceeds point cloud size"),
    ],
)
def test_invalid_metainfo_is_rejected(process_cloud, num_points, ranges, message):
    """Propagate devkit validation failures before counting or exporting points."""
    with pytest.raises(ValueError, match=message):
        process_cloud(num_points, ranges)
