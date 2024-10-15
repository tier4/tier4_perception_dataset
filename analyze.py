import argparse
import os
from typing import List, Tuple

from builtin_interfaces.msg import Time
import matplotlib.pyplot as plt
import numpy as np

from perception_dataset.ros2.oxts_msgs.ins_handler import INSHandler, LocalizeMethod
from perception_dataset.ros2.vehicle_msgs.vehicle_status_handler import VehicleStatusHandler
from perception_dataset.utils.rosbag2 import stamp_to_unix_timestamp


def _analyze_ins(
    bag_dir: str,
    output_dir: str | None = None,
    localize_method: LocalizeMethod = LocalizeMethod.WITH_ODOMETRY,
    *,
    show: bool = False,
) -> None:
    """Analyze INS topis.

    Args:
        bag_dir (str): Directory path of rosbag.
        output_dir (str | None, optional): Output directory path to save figures.
            If None, it skips to save figures.
        localize_method (LocalizeMethod, optional): Method type of localization.
        show (bool, optional): Whether to show figures.
    """
    handler = INSHandler(bag_dir, localize_method=localize_method)

    # plot ego (=Odometry+IMU)
    odometries = handler.get_odometries()

    ego_timestamps: List[float] = []
    ego_translations: List[Tuple[float, float, float]] = []
    ego_linear_velocities: List[Tuple[float, float, float]] = []
    ego_angular_velocities: List[Tuple[float, float, float]] = []
    ego_accelerations: List[Tuple[float, float, float]] = []
    for msg in odometries:
        try:
            ego_state = handler.get_ego_state(msg.header.stamp)
        except Exception as e:
            print(e)
            continue

        ego_timestamps.append(stamp_to_unix_timestamp(ego_state.header.stamp))
        ego_translations.append(
            (
                ego_state.translation.x,
                ego_state.translation.y,
                ego_state.translation.z,
            )
        )
        ego_linear_velocities.append(
            (
                ego_state.twist.linear.x,
                ego_state.twist.linear.y,
                ego_state.twist.linear.z,
            )
        )
        ego_angular_velocities.append(
            (
                ego_state.twist.angular.x,
                ego_state.twist.angular.y,
                ego_state.twist.angular.z,
            )
        )
        ego_accelerations.append((ego_state.accel.x, ego_state.accel.y, ego_state.accel.z))

    ego_translations = np.array(ego_translations)
    ego_linear_velocities = np.array(ego_linear_velocities)
    ego_angular_velocities = np.array(ego_angular_velocities)
    ego_accelerations = np.array(ego_accelerations)

    # translation
    ego_trans_fig, ego_trans_axes = plt.subplots(nrows=1, ncols=3)
    ego_trans_axes[0].plot(
        ego_timestamps,
        ego_translations[:, 0],
        "--",
        label="EGO/Translation/x",
    )
    ego_trans_axes[1].plot(
        ego_timestamps,
        ego_translations[:, 1],
        "--",
        label="EGO/Translation/y",
    )
    ego_trans_axes[2].plot(
        ego_timestamps,
        ego_translations[:, 2],
        "--",
        label="EGO/Translation/z",
    )

    ego_trans_fig.legend()
    if output_dir is not None:
        ego_trans_fig.savefig(os.path.join(output_dir, "ego_translation.png"))

    # linear velocity
    ego_linear_vel_fig, ego_linear_vel_axes = plt.subplots(nrows=1, ncols=3)
    ego_linear_vel_axes[0].plot(
        ego_timestamps,
        ego_linear_velocities[:, 0],
        "--",
        label="EGO/LinearVelocity/x",
    )
    ego_linear_vel_axes[1].plot(
        ego_timestamps,
        ego_linear_velocities[:, 1],
        "--",
        label="EGO/LinearVelocity/y",
    )
    ego_linear_vel_axes[2].plot(
        ego_timestamps,
        ego_linear_velocities[:, 2],
        "--",
        label="EGO/LinearVelocity/z",
    )

    ego_linear_vel_fig.legend()
    if output_dir is not None:
        ego_linear_vel_fig.savefig(os.path.join(output_dir, "ego_linear_velocity.png"))

    # angular velocity
    ego_angular_vel_fig, ego_angular_vel_axes = plt.subplots(nrows=1, ncols=3)
    ego_angular_vel_axes[0].plot(
        ego_timestamps,
        ego_angular_velocities[:, 0],
        "--",
        label="EGO/AngularVelocity/Roll",
    )
    ego_angular_vel_axes[1].plot(
        ego_timestamps,
        ego_angular_velocities[:, 1],
        "--",
        label="EGO/AngularVelocity/Pitch",
    )
    ego_angular_vel_axes[2].plot(
        ego_timestamps,
        ego_angular_velocities[:, 2],
        "--",
        label="EGO/AngularVelocity/Yaw",
    )

    ego_angular_vel_fig.legend()
    if output_dir is not None:
        ego_angular_vel_fig.savefig(os.path.join(output_dir, "ego_angular_velocity.png"))

    # accel
    ego_accel_fig, ego_accel_axes = plt.subplots(nrows=1, ncols=3)
    ego_accel_axes[0].plot(
        ego_timestamps,
        ego_accelerations[:, 0],
        "--",
        label="EGO/Acceleration/x",
    )
    ego_accel_axes[1].plot(
        ego_timestamps,
        ego_accelerations[:, 1],
        "--",
        label="EGO/Acceleration/y",
    )
    ego_accel_axes[2].plot(
        ego_timestamps,
        ego_accelerations[:, 2],
        "--",
        label="EGO/Acceleration/z",
    )

    ego_accel_fig.legend()
    if output_dir is not None:
        ego_accel_fig.savefig(os.path.join(output_dir, "ego_acceleration.png"))

    # === plot IMU ===
    imus = handler.get_imus()

    imu_timestamps: List[float] = []
    angular_velocities: List[Tuple[float, float, float]] = []
    linear_accelerations: List[Tuple[float, float, float]] = []
    imu_stamps: List[Time] = []  # used to interpolate NavSatFix
    for msg in imus:
        imu_timestamps.append(stamp_to_unix_timestamp(msg.header.stamp))

        angular_velocities.append(
            (
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            )
        )

        linear_accelerations.append(
            (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            )
        )

        imu_stamps.append(msg.header.stamp)

    angular_velocities = np.array(angular_velocities)
    linear_accelerations = np.array(linear_accelerations)

    angular_vel_fig, angular_vel_axes = plt.subplots(nrows=1, ncols=3)
    angular_vel_axes[0].plot(
        imu_timestamps,
        angular_velocities[:, 0],
        "--",
        label="IMU/AngularVelocity/x",
    )
    angular_vel_axes[1].plot(
        imu_timestamps,
        angular_velocities[:, 1],
        "--",
        label="IMU/AngularVelocity/y",
    )
    angular_vel_axes[2].plot(
        imu_timestamps,
        angular_velocities[:, 2],
        "--",
        label="IMU/AngularVelocity/z",
    )

    angular_vel_fig.legend()
    if output_dir is not None:
        angular_vel_fig.savefig(os.path.join(output_dir, "imu_angular_velocity.png"))

    linear_acc_fig, linear_acc_axes = plt.subplots(nrows=1, ncols=3)
    linear_acc_axes[0].plot(
        imu_timestamps,
        linear_accelerations[:, 0],
        "--",
        label="IMU/LinearAcceleration/x",
    )
    linear_acc_axes[1].plot(
        imu_timestamps,
        linear_accelerations[:, 1],
        "--",
        label="IMU/LinearAcceleration/y",
    )
    linear_acc_axes[2].plot(
        imu_timestamps,
        linear_accelerations[:, 2],
        "--",
        label="IMU/LinearAcceleration/z",
    )

    linear_acc_fig.legend()
    if output_dir is not None:
        linear_acc_fig.savefig(os.path.join(output_dir, "imu_linear_acceleration.png"))

    # === plot NavSatFix ===
    nav_sat_fixes = handler.get_nav_sat_fixes()

    nav_sat_fix_timestamps: List[float] = []
    geo_coordinates: List[Tuple[float, float, float]] = []
    for msg in nav_sat_fixes:
        nav_sat_fix_timestamps.append(stamp_to_unix_timestamp(msg.header.stamp))

        geo_coordinates.append(
            (
                msg.latitude,
                msg.longitude,
                msg.altitude,
            )
        )

    nav_sat_fix_timestamps = np.array(nav_sat_fix_timestamps)
    geo_coordinates = np.array(geo_coordinates)

    # === plot interpolated NavSatFix ===
    interp_nav_sat_fixes = handler.get_interpolated_nav_sat_fixes(query_stamps=imu_stamps)

    interp_nav_sat_fix_timestamps: List[float] = []
    interp_geo_coordinates: List[Tuple[float, float, float]] = []
    for msg in interp_nav_sat_fixes:
        interp_nav_sat_fix_timestamps.append(stamp_to_unix_timestamp(msg.header.stamp))

        interp_geo_coordinates.append(
            (
                msg.latitude,
                msg.longitude,
                msg.altitude,
            )
        )

    interp_nav_sat_fix_timestamps = np.array(interp_nav_sat_fix_timestamps)
    interp_geo_coordinates = np.array(interp_geo_coordinates)

    geo_coords_fig, geo_coords_axes = plt.subplots(nrows=1, ncols=3)
    # latitude
    geo_coords_axes[0].plot(
        nav_sat_fix_timestamps,
        geo_coordinates[:, 0],
        "b--",
        label="NavSatFix/latitude",
    )
    geo_coords_axes[0].plot(
        interp_nav_sat_fix_timestamps,
        interp_geo_coordinates[:, 0],
        "r",
        label="NavSatFix/latitude/interp",
    )
    # longitude
    geo_coords_axes[1].plot(
        nav_sat_fix_timestamps,
        geo_coordinates[:, 1],
        "b--",
        label="NavSatFix/longitude",
    )
    geo_coords_axes[1].plot(
        interp_nav_sat_fix_timestamps,
        interp_geo_coordinates[:, 1],
        "r",
        label="NavSatFix/longitude/interp",
    )
    # altitude
    geo_coords_axes[2].plot(
        nav_sat_fix_timestamps,
        geo_coordinates[:, 2],
        "b--",
        label="NavSatFix/altitude",
    )
    geo_coords_axes[2].plot(
        interp_nav_sat_fix_timestamps,
        interp_geo_coordinates[:, 2],
        "r",
        label="NavSatFix/altitude/interp",
    )

    geo_coords_fig.legend()
    if output_dir is not None:
        geo_coords_fig.savefig(os.path.join(output_dir, "geo_coordinates.png"))

    if show:
        plt.show()

    plt.cla()
    plt.close()


def _analyze_vehicle_state(
    bag_dir: str,
    output_dir: str | None = None,
    *,
    show: bool = False,
) -> None:
    """Analyze vehicle state topics, and export data to figures.

    Args:
        bag_dir (str): Directory path of rosbag.
        output_dir (str | None): Directory path to save figures. If None, it skips to save figures.
        show (bool, optional): Whether to show figures.
    """
    handler = VehicleStatusHandler(bag_dir)

    # === plot actuation status ===
    actuation_statuses = handler.get_actuation_statuses()

    actuation_timestamps: List[float] = []
    accel_statuses: List[float] = []  # [%]
    brake_statuses: List[float] = []  # [%]
    steer_statuses: List[float] = []  # [%]
    for msg in actuation_statuses:
        actuation_timestamps.append(stamp_to_unix_timestamp(msg.header.stamp))

        accel_statuses.append(msg.status.accel_status)
        brake_statuses.append(msg.status.brake_status)
        steer_statuses.append(msg.status.steer_status)

    actuation_fig, actuation_axes = plt.subplots(nrows=1, ncols=3)
    actuation_axes[0].plot(
        actuation_timestamps,
        accel_statuses,
        "--",
        label="ActuationStatus/AccelStatus",
    )
    actuation_axes[1].plot(
        actuation_timestamps,
        brake_statuses,
        "--",
        label="ActuationStatus/BrakeStatus",
    )
    actuation_axes[2].plot(
        actuation_timestamps,
        steer_statuses,
        "--",
        label="ActuationStatus/SteerStatus",
    )

    actuation_fig.legend()
    if output_dir is not None:
        actuation_fig.savefig(os.path.join(output_dir, "actuation_status.png"))

    # === plot gear reports ===
    gear_reports = handler.get_gear_reports()

    gear_timestamps: List[float] = []
    gear_values: List[int] = []
    for msg in gear_reports:
        gear_timestamps.append(stamp_to_unix_timestamp(msg.stamp))
        gear_values.append(msg.report)

    gear_fig, gear_ax = plt.subplots(nrows=1, ncols=1)
    gear_ax.plot(gear_timestamps, gear_values, label="GearReport")

    gear_fig.legend()
    if output_dir is not None:
        gear_fig.savefig(os.path.join(output_dir, "gear_report.png"))

    # === plot steering reports ===
    steering_reports = handler.get_steering_reports()

    steering_timestamps: List[float] = []
    steering_tire_angles: List[float] = []
    for msg in steering_reports:
        steering_timestamps.append(stamp_to_unix_timestamp(msg.stamp))
        steering_tire_angles.append(msg.steering_tire_angle)

    steering_fig, steering_ax = plt.subplots(nrows=1, ncols=1)
    steering_ax.plot(
        steering_timestamps,
        steering_tire_angles,
        label="SteeringTireAngle",
    )

    steering_fig.legend()
    if output_dir is not None:
        steering_fig.savefig(os.path.join(output_dir, "steering_tire_angle.png"))

    # === plot steering wheel statuses ===
    steering_wheel_statuses = handler.get_steering_wheel_statuses()

    steering_wheel_timestamps: List[float] = []
    steering_wheel_angles: List[float] = []
    for msg in steering_wheel_statuses:
        steering_wheel_timestamps.append(stamp_to_unix_timestamp(msg.stamp))
        steering_wheel_angles.append(msg.data)

    steering_wheel_fig, steering_wheel_ax = plt.subplots(nrows=1, ncols=1)
    steering_wheel_ax.plot(
        steering_wheel_timestamps,
        steering_wheel_angles,
        label="SteeringWheelAngle",
    )

    steering_wheel_fig.legend()
    if output_dir is not None:
        steering_wheel_fig.savefig(os.path.join(output_dir, "steering_wheel_angle.png"))

    # === plot turn indicators statuses ===
    turn_indicators_reports = handler.get_turn_indicators_reports()

    turn_indicators_timestamps: List[float] = []
    turn_indicators: List[int] = []
    for msg in turn_indicators_reports:
        turn_indicators_timestamps.append(stamp_to_unix_timestamp(msg.stamp))
        turn_indicators.append(msg.report)

    turn_indicators_fig, turn_indicators_ax = plt.subplots(nrows=1, ncols=1)
    turn_indicators_ax.plot(
        turn_indicators_timestamps,
        turn_indicators,
        label="TurnIndicators",
    )

    turn_indicators_fig.legend()
    if output_dir is not None:
        turn_indicators_fig.savefig(os.path.join(output_dir, "turn_indicators.png"))

    # === plot hazard lights statuses ===
    hazard_lights_reports = handler.get_hazard_lights_reports()

    hazard_lights_timestamps: List[float] = []
    hazard_lights: List[int] = []
    for msg in hazard_lights_reports:
        hazard_lights_timestamps.append(stamp_to_unix_timestamp(msg.stamp))
        hazard_lights.append(msg.report)

    hazard_lights_fig, hazard_lights_ax = plt.subplots(nrows=1, ncols=1)
    hazard_lights_ax.plot(
        hazard_lights_timestamps,
        hazard_lights,
        label="HazardLights",
    )

    hazard_lights_fig.legend()
    if output_dir is not None:
        hazard_lights_fig.savefig(os.path.join(output_dir, "hazard_lights.png"))

    # === plot velocity reports ===
    velocity_reports = handler.get_velocity_reports()

    velocity_timestamps: List[float] = []
    longitudinal_velocities: List[float] = []
    lateral_velocities: List[float] = []
    heading_rates: List[float] = []
    for msg in velocity_reports:
        velocity_timestamps.append(stamp_to_unix_timestamp(msg.header.stamp))
        longitudinal_velocities.append(msg.longitudinal_velocity)
        lateral_velocities.append(msg.lateral_velocity)
        heading_rates.append(msg.heading_rate)

    vel_report_fig, vel_report_ax = plt.subplots(nrows=1, ncols=3)
    vel_report_ax[0].plot(
        velocity_timestamps,
        lateral_velocities,
        "--",
        label="VelocityReport/LateralVelocity",
    )
    vel_report_ax[1].plot(
        velocity_timestamps,
        longitudinal_velocities,
        "--",
        label="VelocityReport/LongitudinalVelocity",
    )
    vel_report_ax[2].plot(
        velocity_timestamps,
        heading_rates,
        "--",
        label="VelocityReport/HeadingRate",
    )

    vel_report_fig.legend()
    if output_dir is not None:
        vel_report_fig.savefig(os.path.join(output_dir, "velocity_report.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input rosbag directory path.")
    parser.add_argument("-o", "--output", type=str, help="Output plot directory path.")
    parser.add_argument("--show", action="store_true", help="Whether to show plot results.")

    args = parser.parse_args()

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    _analyze_ins(bag_dir=args.input, output_dir=args.output, show=args.show)
    _analyze_vehicle_state(bag_dir=args.input, output_dir=args.output, show=args.show)


if __name__ == "__main__":
    main()
