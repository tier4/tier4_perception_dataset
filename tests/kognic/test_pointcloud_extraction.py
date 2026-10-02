from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from perception_dataset.constants import LIDAR_CONCAT_CHANNEL
from perception_dataset.utils.pointcloud import extract_pointclouds


class _FakeSensorPointCloud:
    def __init__(self, points: np.ndarray):
        self.points = points

    def num_points(self) -> int:
        return self.points.shape[1]


class _FakePointCloud:
    def __init__(self, points: np.ndarray, frame_index: int):
        self.points = points
        self.metainfo = SimpleNamespace(
            sources=[
                SimpleNamespace(
                    sensor_token="front_token",
                    stamp=SimpleNamespace(sec=frame_index + 1, nanosec=0),
                ),
                SimpleNamespace(
                    sensor_token="rear_token",
                    stamp=SimpleNamespace(sec=frame_index + 2, nanosec=0),
                ),
            ]
        )
        self._split = {
            "front_token": _FakeSensorPointCloud(
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
            ),
            "rear_token": _FakeSensorPointCloud(
                np.array([[5.0], [6.0], [7.0], [8.0]], dtype=np.float32)
            ),
        }

    def split_by_sensor(self):
        return self._split


def test_extract_pointclouds_loads_each_concat_frame_once(tmp_path):
    seq_path = tmp_path / "sequence"
    out_dir = tmp_path / "out"
    (seq_path / "data/LIDAR_CONCAT").mkdir(parents=True)
    (seq_path / "data/LIDAR_CONCAT_INFO").mkdir(parents=True)

    frame_records = []
    for i in range(2):
        (seq_path / f"data/LIDAR_CONCAT/{i}.pcd.bin").write_bytes(b"bin")
        (seq_path / f"data/LIDAR_CONCAT_INFO/{i}.json").write_text("{}")
        frame_records.append(
            {
                LIDAR_CONCAT_CHANNEL: SimpleNamespace(
                    filename=f"data/LIDAR_CONCAT/{i}.pcd.bin",
                    info_filename=f"data/LIDAR_CONCAT_INFO/{i}.json",
                    timestamp=1000 + i,
                    token=f"token-{i}",
                )
            }
        )

    call_count = {"value": 0}

    def _fake_from_file(filepath: str, metainfo_filepath: str | None = None):
        assert filepath.endswith(".pcd.bin")
        assert metainfo_filepath is not None
        frame_index = call_count["value"]
        call_count["value"] += 1
        return _FakePointCloud(
            np.array([[9.0], [10.0], [11.0], [12.0]], dtype=np.float32),
            frame_index=frame_index,
        )

    with patch(
        "perception_dataset.utils.pointcloud.LidarPointCloud.from_file",
        side_effect=_fake_from_file,
    ) as mocked_from_file:
        extract_pointclouds(
            seq_path=seq_path,
            out_dir=out_dir,
            lidar_channels=[LIDAR_CONCAT_CHANNEL, "LIDAR_FRONT", "LIDAR_REAR"],
            frame_records=frame_records,
            channel_to_token={
                LIDAR_CONCAT_CHANNEL: "concat_token",
                "LIDAR_FRONT": "front_token",
                "LIDAR_REAR": "rear_token",
            },
        )

    assert mocked_from_file.call_count == len(frame_records)
    assert len(list((out_dir / "lidar" / LIDAR_CONCAT_CHANNEL).glob("*.csv"))) == 2
    assert len(list((out_dir / "lidar" / "LIDAR_FRONT").glob("*.csv"))) == 2
    assert len(list((out_dir / "lidar" / "LIDAR_REAR").glob("*.csv"))) == 2
