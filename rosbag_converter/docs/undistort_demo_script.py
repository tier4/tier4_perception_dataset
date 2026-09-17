#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from typing import Iterable

from autoware_pointcloud_preprocessor.distortion_corrector import DistortionCorrector
from geometry_msgs.msg import TransformStamped, TwistWithCovarianceStamped
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Imu, PointCloud2
from tf2_msgs.msg import TFMessage

WORKSPACE = Path("/home/maxschmeller/autoware")
INPUT_BAG = Path(
    "/home/maxschmeller/.webauto/data/data/log_file/"
    "2a1835a0-ae9f-42c2-bd1c-04f159d43d79/"
    "cfa23601-97c4-4d47-a37e-edfc7e080d8c_2026-06-12-14-36-49_p0900_0.db3"
)
OUTPUT_BAG = INPUT_BAG.with_name(INPUT_BAG.stem + "_left_lower_undistorted.db3")
DECODED_DB = Path("/tmp/left_lower_decoded_pointclouds.db3")

PANDAR_TOPIC = "/sensing/lidar/left_lower/pandar_packets"
INPUT_CLOUD_TOPIC = "/input/sensing/lidar/left_lower/pointcloud_raw_ex"
OUTPUT_CLOUD_TOPIC = "/output/sensing/lidar/left_lower/pointcloud_raw_ex"
TF_STATIC_TOPIC = "/tf_static"
IMU_TOPIC = "/sensing/imu/imu_data"
TWIST_TOPIC = "/sensing/vehicle_velocity_converter/twist_with_covariance"

BASE_FRAME = "base_link"
LIDAR_FRAME = "left_lower/lidar"
LIDAR_BASE_FRAME = "left_lower/lidar_base_link"
SENSOR_KIT_FRAME = "sensor_kit_base_link"

HELPER_DIR = Path("/tmp/left_lower_pandar_decoder")
HELPER_INSTALL = Path("/tmp/left_lower_pandar_decoder_install")
HELPER_EXE = (
    HELPER_INSTALL / "left_lower_pandar_decoder/lib/left_lower_pandar_decoder/decode_left_lower"
)
CALIBRATION_DIR = (
    WORKSPACE / "src/sensor_component/external/nebula/src/nebula_hesai/"
    "nebula_hesai_decoders/calibration"
)


@dataclass(frozen=True)
class Event:
    source_id: int
    timestamp: int
    topic: str
    data: bytes


def run(cmd: list[str], *, cwd: Path = WORKSPACE) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_decoder_helper() -> None:
    if HELPER_EXE.exists():
        return

    src_dir = HELPER_DIR / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (HELPER_DIR / "package.xml").write_text(
        """<?xml version="1.0"?>
<package format="3">
  <name>left_lower_pandar_decoder</name>
  <version>0.0.0</version>
  <description>Temporary left_lower PandarScan decoder.</description>
  <maintainer email="max@example.com">max</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <depend>rclcpp</depend>
  <depend>pandar_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>nebula_core_common</depend>
  <depend>nebula_core_ros</depend>
  <depend>nebula_hesai_common</depend>
  <depend>nebula_hesai_decoders</depend>
</package>
""",
        encoding="utf-8",
    )
    (HELPER_DIR / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.16)
project(left_lower_pandar_decoder)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pandar_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nebula_core_common REQUIRED)
find_package(nebula_core_ros REQUIRED)
find_package(nebula_hesai_common REQUIRED)
find_package(nebula_hesai_decoders REQUIRED)

add_executable(decode_left_lower src/decode_left_lower.cpp)
target_compile_features(decode_left_lower PRIVATE cxx_std_17)
ament_target_dependencies(
  decode_left_lower
  rclcpp
  pandar_msgs
  sensor_msgs
  nebula_core_common
  nebula_core_ros
  nebula_hesai_common
  nebula_hesai_decoders)
target_link_libraries(decode_left_lower sqlite3)

install(TARGETS decode_left_lower DESTINATION lib/${PROJECT_NAME})
ament_package()
""",
        encoding="utf-8",
    )
    (src_dir / "decode_left_lower.cpp").write_text(
        r"""#include <nebula_core_ros/point_cloud_conversions.hpp>
#include <nebula_core_ros/rclcpp_logger.hpp>
#include <nebula_hesai_common/hesai_common.hpp>
#include <nebula_hesai_decoders/hesai_driver.hpp>
#include <pandar_msgs/msg/pandar_scan.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sqlite3.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class Stmt
{
public:
  Stmt(sqlite3 * db, const char * sql) : db_(db)
  {
    if (sqlite3_prepare_v2(db_, sql, -1, &stmt_, nullptr) != SQLITE_OK) {
      throw std::runtime_error(sqlite3_errmsg(db_));
    }
  }
  ~Stmt() { sqlite3_finalize(stmt_); }
  sqlite3_stmt * get() { return stmt_; }
  void reset()
  {
    sqlite3_reset(stmt_);
    sqlite3_clear_bindings(stmt_);
  }

private:
  sqlite3 * db_{};
  sqlite3_stmt * stmt_{};
};

void exec(sqlite3 * db, const char * sql)
{
  char * err = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string msg = err ? err : "sqlite error";
    sqlite3_free(err);
    throw std::runtime_error(msg);
  }
}

std::vector<uint8_t> serialize_cloud(const sensor_msgs::msg::PointCloud2 & cloud)
{
  rclcpp::SerializedMessage ser;
  rclcpp::Serialization<sensor_msgs::msg::PointCloud2>().serialize_message(&cloud, &ser);
  const auto & rcl = ser.get_rcl_serialized_message();
  return {rcl.buffer, rcl.buffer + rcl.buffer_length};
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    if (argc != 4) {
      std::cerr << "usage: decode_left_lower INPUT_BAG OUTPUT_DB CALIBRATION_DIR\n";
      return 2;
    }
    const std::string input = argv[1];
    const std::string output = argv[2];
    const std::string calib_dir = argv[3];
    fs::remove(output);

    sqlite3 * in_db = nullptr;
    sqlite3 * out_db = nullptr;
    if (sqlite3_open_v2(input.c_str(), &in_db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
      throw std::runtime_error(sqlite3_errmsg(in_db));
    }
    if (sqlite3_open(output.c_str(), &out_db) != SQLITE_OK) {
      throw std::runtime_error(sqlite3_errmsg(out_db));
    }
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> in_guard(in_db, sqlite3_close);
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> out_guard(out_db, sqlite3_close);

    exec(out_db, "create table decoded(id integer primary key, timestamp integer not null, data blob not null)");
    exec(out_db, "begin transaction");

    auto sensor_cfg = std::make_shared<nebula::drivers::HesaiSensorConfiguration>();
    sensor_cfg->sensor_model = nebula::drivers::sensor_model_from_string("PandarQT128");
    sensor_cfg->return_mode =
      nebula::drivers::return_mode_from_string_hesai("LastStrongest", sensor_cfg->sensor_model);
    sensor_cfg->frame_id = "left_lower/lidar";
    sensor_cfg->sync_angle = 0;
    sensor_cfg->cut_angle = 0.0;
    sensor_cfg->cloud_min_angle = 0;
    sensor_cfg->cloud_max_angle = 360;
    sensor_cfg->rotation_speed = 600;
    sensor_cfg->dual_return_distance_threshold = 0.1;
    sensor_cfg->min_range = 0.3;
    sensor_cfg->max_range = 300.0;
    sensor_cfg->packet_mtu_size = 1500;

    auto calibration = std::make_shared<nebula::drivers::HesaiCalibrationConfiguration>();
    calibration->calibration_file = calib_dir + "/PandarQT128.csv";
    if (calibration->load_from_file(calibration->calibration_file) != nebula::Status::OK) {
      throw std::runtime_error("failed to load PandarQT128 calibration");
    }
    auto driver = std::make_shared<nebula::drivers::HesaiDriver>(
      sensor_cfg, calibration,
      std::make_shared<nebula::drivers::loggers::RclcppLogger>("left_lower_pandar_decoder"),
      nullptr);

    Stmt topic_stmt(in_db, "select id from topics where name = '/sensing/lidar/left_lower/pandar_packets'");
    if (sqlite3_step(topic_stmt.get()) != SQLITE_ROW) {
      throw std::runtime_error("left_lower pandar topic not found");
    }
    const int topic_id = sqlite3_column_int(topic_stmt.get(), 0);

    Stmt select(
      in_db,
      "select id, timestamp, data from messages where topic_id = ? order by id");
    sqlite3_bind_int(select.get(), 1, topic_id);
    Stmt insert(out_db, "insert into decoded(id, timestamp, data) values (?, ?, ?)");

    rclcpp::Serialization<pandar_msgs::msg::PandarScan> scan_serialization;
    int count = 0;
    while (sqlite3_step(select.get()) == SQLITE_ROW) {
      const int message_id = sqlite3_column_int(select.get(), 0);
      const int64_t timestamp = sqlite3_column_int64(select.get(), 1);
      const auto * blob = static_cast<const uint8_t *>(sqlite3_column_blob(select.get(), 2));
      const int blob_size = sqlite3_column_bytes(select.get(), 2);

      rclcpp::SerializedMessage scan_ser(blob_size);
      auto & rcl_scan = scan_ser.get_rcl_serialized_message();
      std::memcpy(rcl_scan.buffer, blob, blob_size);
      rcl_scan.buffer_length = blob_size;

      pandar_msgs::msg::PandarScan scan;
      scan_serialization.deserialize_message(&scan_ser, &scan);

      std::optional<sensor_msgs::msg::PointCloud2> decoded_cloud;
      driver->set_pointcloud_callback(
        [&](const nebula::drivers::NebulaPointCloudPtr & pointcloud, double) {
          auto cloud = nebula::ros::to_ros_msg(*pointcloud);
          cloud.header = scan.header;
          if (cloud.header.frame_id.empty()) {
            cloud.header.frame_id = "left_lower/lidar";
          }
          decoded_cloud = std::move(cloud);
        });

      for (const auto & packet : scan.packets) {
        std::vector<uint8_t> packet_data(packet.data.begin(), packet.data.begin() + packet.size);
        driver->parse_cloud_packet(packet_data);
      }

      sensor_msgs::msg::PointCloud2 cloud;
      if (decoded_cloud) {
        cloud = std::move(*decoded_cloud);
      } else {
        cloud.header = scan.header;
        cloud.header.frame_id = "left_lower/lidar";
        cloud.height = 1;
        cloud.width = 0;
        cloud.is_bigendian = false;
        cloud.point_step = 0;
        cloud.row_step = 0;
        cloud.is_dense = true;
      }
      const auto serialized = serialize_cloud(cloud);

      sqlite3_bind_int(insert.get(), 1, message_id);
      sqlite3_bind_int64(insert.get(), 2, timestamp);
      sqlite3_bind_blob(insert.get(), 3, serialized.data(), static_cast<int>(serialized.size()), SQLITE_TRANSIENT);
      if (sqlite3_step(insert.get()) != SQLITE_DONE) {
        throw std::runtime_error(sqlite3_errmsg(out_db));
      }
      insert.reset();
      ++count;
    }

    exec(out_db, "commit");
    std::cout << "decoded " << count << " left_lower scans\n";
  } catch (const std::exception & e) {
    std::cerr << "error: " << e.what() << "\n";
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
""",
        encoding="utf-8",
    )

    env_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"source {WORKSPACE}/install/setup.bash && "
        "colcon --log-base /tmp/left_lower_pandar_decoder_log build "
        f"--base-paths {HELPER_DIR} "
        "--build-base /tmp/left_lower_pandar_decoder_build "
        f"--install-base {HELPER_INSTALL} "
        "--event-handlers console_direct+"
    )
    run(["bash", "-lc", env_cmd])


def decode_left_lower() -> None:
    ensure_decoder_helper()
    if DECODED_DB.exists():
        DECODED_DB.unlink()
    run([str(HELPER_EXE), str(INPUT_BAG), str(DECODED_DB), str(CALIBRATION_DIR)])


def quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]):
    x, y, z = v
    qv = (x, y, z, 0.0)
    q_conj = (-q[0], -q[1], -q[2], q[3])
    out = quat_multiply(quat_multiply(q, qv), q_conj)
    return out[:3]


def compose_transform(
    parent_to_mid: TransformStamped, mid_to_child: TransformStamped
) -> TransformStamped:
    out = TransformStamped()
    out.header = parent_to_mid.header
    out.child_frame_id = mid_to_child.child_frame_id

    q1 = (
        parent_to_mid.transform.rotation.x,
        parent_to_mid.transform.rotation.y,
        parent_to_mid.transform.rotation.z,
        parent_to_mid.transform.rotation.w,
    )
    q2 = (
        mid_to_child.transform.rotation.x,
        mid_to_child.transform.rotation.y,
        mid_to_child.transform.rotation.z,
        mid_to_child.transform.rotation.w,
    )
    t1 = parent_to_mid.transform.translation
    t2 = mid_to_child.transform.translation
    rt2 = quat_rotate(q1, (t2.x, t2.y, t2.z))

    out.transform.translation.x = t1.x + rt2[0]
    out.transform.translation.y = t1.y + rt2[1]
    out.transform.translation.z = t1.z + rt2[2]
    q = quat_multiply(q1, q2)
    out.transform.rotation.x = q[0]
    out.transform.rotation.y = q[1]
    out.transform.rotation.z = q[2]
    out.transform.rotation.w = q[3]
    return out


def identity_transform(parent: str, child: str) -> TransformStamped:
    out = TransformStamped()
    out.header.frame_id = parent
    out.child_frame_id = child
    out.transform.rotation.w = 1.0
    return out


def load_static_transforms(
    con: sqlite3.Connection,
) -> tuple[list[Event], dict[tuple[str, str], TransformStamped]]:
    topic_id = con.execute("select id from topics where name = ?", (TF_STATIC_TOPIC,)).fetchone()[
        0
    ]
    events: list[Event] = []
    transforms: dict[tuple[str, str], TransformStamped] = {}
    for msg_id, timestamp, data in con.execute(
        "select id, timestamp, data from messages where topic_id = ? order by id", (topic_id,)
    ):
        events.append(Event(msg_id, timestamp, TF_STATIC_TOPIC, bytes(data)))
        tf_message = deserialize_message(data, TFMessage)
        for transform in tf_message.transforms:
            transforms[(transform.header.frame_id, transform.child_frame_id)] = transform
    return events, transforms


def make_lidar_transform(transforms: dict[tuple[str, str], TransformStamped]) -> TransformStamped:
    base_to_kit = transforms[(BASE_FRAME, SENSOR_KIT_FRAME)]
    kit_to_lidar_base = transforms[(SENSOR_KIT_FRAME, LIDAR_BASE_FRAME)]
    lidar_base_to_lidar = transforms[(LIDAR_BASE_FRAME, LIDAR_FRAME)]
    return compose_transform(
        compose_transform(base_to_kit, kit_to_lidar_base), lidar_base_to_lidar
    )


def create_output_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    journal = path.with_name(path.name + "-journal")
    if journal.exists():
        journal.unlink()
    con = sqlite3.connect(path)
    con.execute(
        "create table schema(schema_version integer primary key, ros_distro text not null)"
    )
    con.execute(
        "create table metadata(id integer primary key, metadata_version integer not null, metadata text not null)"
    )
    con.execute(
        "create table topics(id integer primary key, name text not null, type text not null, "
        "serialization_format text not null, offered_qos_profiles text not null)"
    )
    con.execute(
        "create table messages(id integer primary key, topic_id integer not null, "
        "timestamp integer not null, data blob not null)"
    )
    con.execute("insert into schema(schema_version, ros_distro) values (?, ?)", (4, "humble"))
    topics = [
        (1, TF_STATIC_TOPIC, "tf2_msgs/msg/TFMessage"),
        (2, IMU_TOPIC, "sensor_msgs/msg/Imu"),
        (3, TWIST_TOPIC, "geometry_msgs/msg/TwistWithCovarianceStamped"),
        (4, INPUT_CLOUD_TOPIC, "sensor_msgs/msg/PointCloud2"),
        (5, OUTPUT_CLOUD_TOPIC, "sensor_msgs/msg/PointCloud2"),
    ]
    for topic_id, name, msg_type in topics:
        con.execute(
            "insert into topics(id, name, type, serialization_format, offered_qos_profiles) "
            "values (?, ?, ?, 'cdr', '')",
            (topic_id, name, msg_type),
        )
    con.commit()
    return con


def iter_motion_events(src: sqlite3.Connection, decoded: sqlite3.Connection) -> Iterable[Event]:
    rows: list[Event] = []
    for topic in (IMU_TOPIC, TWIST_TOPIC):
        topic_id = src.execute("select id from topics where name = ?", (topic,)).fetchone()[0]
        rows.extend(
            Event(msg_id, timestamp, topic, bytes(data))
            for msg_id, timestamp, data in src.execute(
                "select id, timestamp, data from messages where topic_id = ? order by id",
                (topic_id,),
            )
        )
    rows.extend(
        Event(msg_id, timestamp, INPUT_CLOUD_TOPIC, bytes(data))
        for msg_id, timestamp, data in decoded.execute(
            "select id, timestamp, data from decoded order by id"
        )
    )
    yield from sorted(rows, key=lambda event: event.source_id)


def write_message(
    con: sqlite3.Connection, next_id: int, topic_id: int, timestamp: int, data: bytes
) -> int:
    con.execute(
        "insert into messages(id, topic_id, timestamp, data) values (?, ?, ?, ?)",
        (next_id, topic_id, timestamp, data),
    )
    return next_id + 1


def main() -> None:
    decode_left_lower()

    src = sqlite3.connect(INPUT_BAG)
    decoded = sqlite3.connect(DECODED_DB)
    out = create_output_db(OUTPUT_BAG)
    corrector = DistortionCorrector(use_3d_distortion_correction=True)

    tf_static_events, transforms = load_static_transforms(src)
    corrector.set_pointcloud_transform(make_lidar_transform(transforms))
    corrector.set_imu_transform(identity_transform(BASE_FRAME, BASE_FRAME))

    topic_ids = {
        TF_STATIC_TOPIC: 1,
        IMU_TOPIC: 2,
        TWIST_TOPIC: 3,
        INPUT_CLOUD_TOPIC: 4,
        OUTPUT_CLOUD_TOPIC: 5,
    }

    next_id = 1
    counts = {topic: 0 for topic in topic_ids}
    status_counts: dict[str, int] = {}
    out.execute("begin transaction")

    for event in tf_static_events:
        next_id = write_message(
            out, next_id, topic_ids[TF_STATIC_TOPIC], event.timestamp, event.data
        )
        counts[TF_STATIC_TOPIC] += 1

    for event in iter_motion_events(src, decoded):
        if event.topic == IMU_TOPIC:
            imu = deserialize_message(event.data, Imu)
            corrector.process_imu_message(imu)
            next_id = write_message(
                out, next_id, topic_ids[IMU_TOPIC], event.timestamp, event.data
            )
            counts[IMU_TOPIC] += 1
        elif event.topic == TWIST_TOPIC:
            twist = deserialize_message(event.data, TwistWithCovarianceStamped)
            corrector.process_twist_message(twist)
            next_id = write_message(
                out, next_id, topic_ids[TWIST_TOPIC], event.timestamp, event.data
            )
            counts[TWIST_TOPIC] += 1
        elif event.topic == INPUT_CLOUD_TOPIC:
            cloud = deserialize_message(event.data, PointCloud2)
            next_id = write_message(
                out, next_id, topic_ids[INPUT_CLOUD_TOPIC], event.timestamp, event.data
            )
            counts[INPUT_CLOUD_TOPIC] += 1

            undistorted, status = corrector.undistort_pointcloud(
                cloud, use_imu=True, update_azimuth_and_distance=False
            )
            key = str(status.validity).split(".")[-1]
            status_counts[key] = status_counts.get(key, 0) + 1
            next_id = write_message(
                out,
                next_id,
                topic_ids[OUTPUT_CLOUD_TOPIC],
                event.timestamp,
                serialize_message(undistorted),
            )
            counts[OUTPUT_CLOUD_TOPIC] += 1
        else:
            raise RuntimeError(f"unexpected event topic: {event.topic}")

    out.commit()
    out.execute("vacuum")
    out.close()
    src.close()
    decoded.close()

    print(f"wrote {OUTPUT_BAG}")
    for topic, count in counts.items():
        print(f"{topic}: {count}")
    print(f"undistortion validity counts: {status_counts}")


if __name__ == "__main__":
    main()
