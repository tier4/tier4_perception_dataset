from __future__ import annotations

import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw Pandar rosbag2 data to T4 dataset.")
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML config path.")
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        type=Path,
        dest="input_bags",
        help="Input rosbag2 directory. Repeat for ordered multi-bag conversion.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output base directory.")
    parser.add_argument(
        "--individual-params-root",
        type=Path,
        default=Path("src/autoware_individual_params.j6_gen2"),
        help="Path to autoware_individual_params repository or its individual_params directory.",
    )
    parser.add_argument("--vehicle-id", required=True, help="Vehicle ID, e.g. j6_gen2_16.")
    parser.add_argument("--vehicle-model", default="j6_gen2", help="Vehicle model name.")
    parser.add_argument("--sensor-model", default="aip_x2_gen2", help="Sensor model name.")
    parser.add_argument("--start-time", type=float, default=None, help="Absolute bag storage start time.")
    parser.add_argument("--end-time", type=float, default=None, help="Absolute bag storage end time.")
    parser.add_argument(
        "--pcd-output-format",
        choices=("bin", "pcd", "both"),
        default=None,
        help="LiDAR sample output format. Overrides conversion.pcd_output_format.",
    )
    parser.add_argument(
        "--output-rosbag-storage-id",
        choices=("sqlite3", "mcap"),
        default=None,
        help="Storage backend for the output rosbag. Defaults to the first input bag backend.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print aggregate timing after this many LiDAR messages.",
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        help="Print aggregate conversion timing summary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-message decode, undistort, and concat trace logs.",
    )
    parser.add_argument(
        "--lidar-undistort-mode",
        choices=("config", "2d", "3d", "none"),
        default="config",
        help="Diagnostic override for LiDAR distortion correction. Defaults to config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print resolved inputs.")
    args = parser.parse_args()

    from .converter import RuntimeOptions
    from .converter import convert

    options = RuntimeOptions(
        config_path=args.config,
        input_bags=args.input_bags,
        output_base=args.output,
        individual_params_root=args.individual_params_root,
        vehicle_id=args.vehicle_id,
        vehicle_model=args.vehicle_model,
        sensor_model=args.sensor_model,
        start_time=args.start_time,
        end_time=args.end_time,
        dry_run=args.dry_run,
        progress_interval=args.progress_interval,
        debug_timing=args.debug_timing,
        verbose=args.verbose,
        pcd_output_format=args.pcd_output_format,
        output_rosbag_storage_id=args.output_rosbag_storage_id,
        lidar_undistort_mode=args.lidar_undistort_mode,
    )
    convert(options)


if __name__ == "__main__":
    main()
