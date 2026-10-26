from __future__ import annotations

from pathlib import Path

import numpy as np

from t4dataset_rosbag_converter.calibration import resolve_calibration
from t4dataset_rosbag_converter.camera_calibration import load_camera_calibrations
from t4dataset_rosbag_converter.config import load_config
from t4dataset_rosbag_converter.converter import _is_after_end
from t4dataset_rosbag_converter.converter import _pass_through_topics
from t4dataset_rosbag_converter.converter import _topics_to_read
from t4dataset_rosbag_converter.pcd import save_lidar_pointcloud_pcd
from t4dataset_rosbag_converter.pointcloud import POINT_DTYPE
from t4dataset_rosbag_converter.pointcloud import pointcloud_to_lidar_features
from t4dataset_rosbag_converter.pointcloud import stamp_from_seconds
from t4dataset_rosbag_converter.pointcloud import structured_array_to_pointcloud
from t4dataset_rosbag_converter.tf_manager import TfManager


ROOT = Path(__file__).resolve().parents[1]


def test_sample_config_extracts_lidar_and_t4_conversion() -> None:
    config = load_config(ROOT / "docs" / "sample_config.yaml")

    assert len(config.lidar_topics) == 8
    assert config.lidar_topics[0].name == "/sensing/lidar/left_lower/pandar_packets"
    assert config.lidar_topics[0].is_reset_topic is True
    assert config.lidar_topics[0].sensor_model == "PandarQT128"
    assert config.lidar_topics[1].sensor_model == "Pandar128E4X"
    assert config.conversion["lidar_sensor"]["channel"] == "LIDAR_CONCAT"
    assert config.conversion["lidar_sensor"]["write_individual_lidar"] is False
    assert len(config.conversion["camera_sensors"]) == 6
    assert "/diagnostics" in config.filter_topics
    assert "/sensing/lidar/front_upper/pandar_packets" in config.filter_topics
    assert "/perception/object_recognition/detection/rois0" in config.filter_topics
    assert "/sensing/gnss/fixed" not in config.filter_topics


def test_individual_params_resolution_finds_vehicle_specific_calibration() -> None:
    calibration = resolve_calibration(
        ROOT / "src" / "autoware_individual_params.j6_gen2",
        "j6_gen2_03",
        "aip_x2_gen2",
    )

    assert calibration.root.name == "aip_x2_gen2"
    assert calibration.pandar_csv("left_lower").exists()
    assert calibration.base_to_frame("left_lower/lidar_base_link").parent == "base_link"


def test_camera_calibration_resolves_sample_config_topics_for_j6_gen2_03() -> None:
    config = load_config(ROOT / "docs" / "sample_config.yaml")
    calibration = resolve_calibration(
        ROOT / "src" / "autoware_individual_params.j6_gen2",
        "j6_gen2_03",
        "aip_x2_gen2",
    )

    cameras = load_camera_calibrations(calibration.root, config.conversion["camera_sensors"])

    front = cameras["/sensing/camera/camera8/image_raw/compressed"]
    assert front.channel == "CAM_FRONT"
    assert front.camera_name == "camera8"
    assert front.frame_id == "top_front_center_left/camera_optical_link"
    assert front.width == 2880
    assert front.height == 1860

    tf_manager = TfManager(calibration, cameras.values())
    transform = tf_manager.lookup_required("base_link", front.frame_id, 0.0)
    assert transform.parent == "base_link"
    assert transform.child == front.frame_id
    assert transform.translation != (0.0, 0.0, 0.0)


def test_converter_reads_motion_topics_and_can_stop_after_end_time() -> None:
    config = load_config(ROOT / "docs" / "sample_config.yaml")
    topics = _topics_to_read(config)

    assert "/localization/twist_estimator/twist_with_covariance" in topics
    assert "/sensing/vehicle_velocity_converter/twist_with_covariance" in topics
    assert "/diagnostics" in topics
    assert "/tf" in topics
    assert config.conversion["lidar_sensor"]["topic"] not in _pass_through_topics(config)
    assert config.conversion["lidar_sensor"]["topic"] not in topics
    assert config.conversion["lidar_sensor"]["lidar_info_topic"] not in topics
    assert _is_after_end(1781242614000000000, 1781242613.36) is True
    assert _is_after_end(1781242613000000000, 1781242613.36) is False


def test_pointcloud_data_uses_ros_array_fast_path() -> None:
    cloud = structured_array_to_pointcloud(
        np.zeros(3, dtype=POINT_DTYPE),
        stamp=stamp_from_seconds(1.0),
        frame_id="base_link",
    )

    assert cloud.data.typecode == "B"
    assert len(cloud.data) == 3 * cloud.point_step


def test_save_lidar_pointcloud_pcd_preserves_compact_field_types(tmp_path: Path) -> None:
    arr = np.zeros(3, dtype=POINT_DTYPE)
    arr["intensity"] = [1, 2, 3]
    arr["return_type"] = [4, 5, 6]
    arr["channel"] = [7, 8, 9]
    arr["time_stamp"] = [10, 11, 12]
    cloud = structured_array_to_pointcloud(
        arr,
        stamp=stamp_from_seconds(1.0),
        frame_id="base_link",
    )

    path = tmp_path / "00000.pcd"
    save_lidar_pointcloud_pcd(path, cloud, num_lidar_feats=7)

    header = path.read_bytes().split(b"DATA", 1)[0].decode()
    assert "FIELDS x y z intensity ring return_type time_stamp" in header
    assert "SIZE 4 4 4 1 2 1 4" in header
    assert "TYPE F F F U U U U" in header


def test_pointcloud_to_lidar_features_keeps_legacy_float32_bin_layout() -> None:
    arr = np.zeros(2, dtype=POINT_DTYPE)
    arr["x"] = [1.0, 2.0]
    arr["intensity"] = [3, 4]
    arr["channel"] = [5, 6]
    arr["return_type"] = [7, 8]
    arr["time_stamp"] = [9, 10]
    cloud = structured_array_to_pointcloud(
        arr,
        stamp=stamp_from_seconds(1.0),
        frame_id="base_link",
    )

    features = pointcloud_to_lidar_features(cloud, num_lidar_feats=7)

    assert features.dtype == np.float32
    assert features.shape == (2, 7)
    assert features[0].tolist() == [1.0, 0.0, 0.0, 3.0, 5.0, 7.0, 9.0]
