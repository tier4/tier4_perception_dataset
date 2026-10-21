import json
import struct

from autoware_adapi_v1_msgs.msg import RouteState
from autoware_planning_msgs.msg import LaneletPrimitive, LaneletRoute, LaneletSegment
from autoware_vehicle_msgs.msg import (
    GearReport,
    HazardLightsReport,
    TurnIndicatorsReport,
    VelocityReport,
)
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Imu, NavSatFix
from std_msgs.msg import Float32, String

from perception_dataset.t4_v1.bag_metadata import BagMetadata
from tests.t4_v1.bags import write_bag

SECOND = 1_763_000_000


def test_bag_metadata(tmp_path):
    stamp = Time(sec=SECOND)
    odometry = Odometry()
    odometry.header.stamp = stamp
    odometry.twist.twist.linear.x, odometry.twist.twist.angular.z = 3.0, 0.1
    imu = Imu()
    imu.header.stamp, imu.header.frame_id = stamp, "base_link"
    imu.linear_acceleration.z = 9.8
    fix = NavSatFix(latitude=35.5, longitude=139.7, altitude=37.0)
    fix.header.stamp = stamp
    velocity = VelocityReport(longitudinal_velocity=3.0)
    velocity.header.stamp = stamp
    route = LaneletRoute(
        segments=[
            LaneletSegment(
                preferred_primitive=LaneletPrimitive(id=7), primitives=[LaneletPrimitive(id=7)]
            )
        ]
    )
    route.header.stamp = stamp
    lidar = DiagnosticStatus(
        name="/sensing/lidar/front_upper/hesai_ros_wrapper_node: hesai_status",
        hardware_id="OT128: OT0001",
    )
    camera = CameraInfo(distortion_model="rational_polynomial")
    camera.header.stamp = stamp
    vehicle_info = String(data=json.dumps({"firmware_release_name": "20250715-v4.2.1"}))

    write_bag(
        tmp_path / "bag",
        [
            ("/localization/kinematic_state", "nav_msgs/msg/Odometry", odometry, SECOND),
            ("/sensing/imu/imu_data", "sensor_msgs/msg/Imu", imu, SECOND),
            ("/sensing/gnss/septentrio/nav_sat_fix", "sensor_msgs/msg/NavSatFix", fix, SECOND),
            # vehicle-specific topics: a stamped wheel angle [rad] (raw CDR, as the message
            # packages are not built) and pedals in percent (ERGA)
            (
                "/vehicle/status/steering_wheel_status",
                "tier4_vehicle_msgs/msg/SteeringWheelStatusStamped",
                b"\x00\x01\x00\x00" + struct.pack("<IIf", SECOND, 0, 1.5),
                SECOND,
            ),
            (
                "/vehicle/status/accel_pedal_position",
                "std_msgs/msg/Float32",
                Float32(data=25.0),
                SECOND,
            ),
            (
                "/vehicle/status/brake_pedal_position",
                "std_msgs/msg/Float32",
                Float32(data=0.0),
                SECOND,
            ),
            (
                "/vehicle/status/velocity_status",
                "autoware_vehicle_msgs/msg/VelocityReport",
                velocity,
                SECOND,
            ),
            (
                "/vehicle/status/gear_status",
                "autoware_vehicle_msgs/msg/GearReport",
                GearReport(stamp=stamp, report=22),
                SECOND,
            ),
            (
                "/vehicle/status/turn_indicators_status",
                "autoware_vehicle_msgs/msg/TurnIndicatorsReport",
                TurnIndicatorsReport(stamp=stamp, report=2),
                SECOND,
            ),
            (
                "/vehicle/status/hazard_lights_status",
                "autoware_vehicle_msgs/msg/HazardLightsReport",
                HazardLightsReport(stamp=stamp, report=1),
                SECOND,
            ),
            (
                "/planning/mission_planning/route",
                "autoware_planning_msgs/msg/LaneletRoute",
                route,
                SECOND,
            ),
            (
                "/api/routing/state",
                "autoware_adapi_v1_msgs/msg/RouteState",
                RouteState(stamp=Time(sec=SECOND + 5), state=3),
                SECOND + 5,
            ),
            (
                "/diagnostics",
                "diagnostic_msgs/msg/DiagnosticArray",
                DiagnosticArray(status=[lidar]),
                SECOND,
            ),
            ("/sensing/camera/camera0/camera_info", "sensor_msgs/msg/CameraInfo", camera, SECOND),
            ("/webauto/vehicle_info", "std_msgs/msg/String", vehicle_info, SECOND),
        ],
    )
    metadata = BagMetadata([str(tmp_path / "bag")])
    us = SECOND * 1_000_000

    near, far = metadata.ego_pose_fields([us + 10_000, us + 300_000])
    assert near["twist"] == [3.0, 0.0, 0.0, 0.1, 0.0, 0.0]
    assert near["acceleration"] == [0.0, 0.0, 9.8]
    assert near["geocoordinate"] == [35.5, 139.7, 37.0]
    assert far == dict.fromkeys(("twist", "covariance", "acceleration", "geocoordinate"))

    (state,) = metadata.vehicle_states([us + 10_000, us + 300_000])
    assert state["shift_state"] == "PARK"
    assert state["indicators"] == {"left": "on", "right": "off", "hazard": "off"}
    assert state["longitudinal_velocity"] == 3.0
    assert (state["steering_wheel_angle"], state["accel_pedal"], state["brake_pedal"]) == (
        1.5,
        0.25,
        0.0,
    )
    assert state["steer_pedal"] is None

    (route_row,) = metadata.routes()
    assert (route_row["valid_from"], route_row["valid_until"]) == (us, us + 5_000_000)
    assert route_row["segments"] == [{"preferred_primitive_id": 7, "primitive_ids": [7]}]

    assert metadata.lidar_hardware()["front_upper"].serial_number == "OT0001"
    assert metadata.distortion_models() == {"camera0": "rational_polynomial"}
    assert metadata.vehicle_info()["firmware_release_name"] == "20250715-v4.2.1"
