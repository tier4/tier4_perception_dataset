import argparse
import contextlib
import io
import os
import os.path as osp
from typing import Any

import yaml

from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams
from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.utils.logger import configure_logger
import perception_dataset.utils.misc as misc_utils
import perception_dataset.utils.rosbag2 as rosbag2_utils

logger = configure_logger(modname=__name__)


def _resolve_bag_path(params: Rosbag2ConverterParams) -> str:
    if params.input_bag_path:
        return params.input_bag_path

    if not osp.isdir(params.input_base):
        raise FileNotFoundError(f"input_base does not exist: {params.input_base}")

    bag_dirs = []
    for entry in sorted(os.listdir(params.input_base)):
        bag_dir = osp.join(params.input_base, entry)
        if osp.isdir(bag_dir) and osp.exists(osp.join(bag_dir, "metadata.yaml")):
            bag_dirs.append(bag_dir)

    if len(bag_dirs) == 0:
        raise FileNotFoundError(f"No rosbag directories found under {params.input_base}")
    if len(bag_dirs) > 1:
        raise ValueError(
            "Multiple rosbag directories found. Please pass --bag-path to select one explicitly: "
            f"{bag_dirs}"
        )
    return bag_dirs[0]


def _load_timestamps(
    reader: Rosbag2Reader,
    topic: str,
    start_time: Any,
    offset_sec: float = 0.0,
) -> list[float]:
    return [
        rosbag2_utils.stamp_to_unix_timestamp(msg.header.stamp) - offset_sec
        for msg in reader.read_messages(topics=[topic], start_time=start_time)
    ]


def _summarize_camera(
    params: Rosbag2ConverterParams,
    reader: Rosbag2Reader,
    lidar_timestamps: list[float],
    num_load_lidar_frames: int,
    camera_sensor: dict,
    camera_start_timestamp: float,
) -> dict[str, Any]:
    cam_start_time = rosbag2_utils.unix_timestamp_to_stamp(
        camera_start_timestamp - 2 * params.system_scan_period_sec - params.max_camera_jitter_sec
    )
    image_timestamps = _load_timestamps(
        reader=reader,
        topic=camera_sensor["topic"],
        start_time=cam_start_time,
        offset_sec=1e-3 * float(camera_sensor["delay_msec"]),
    )

    with contextlib.redirect_stdout(io.StringIO()):
        if (
            params.system_scan_period_sec == params.camera_scan_period_sec
            or not params.accept_frame_drop
        ):
            synced_frame_info = misc_utils.get_lidar_camera_synced_frame_info(
                image_timestamp_list=image_timestamps,
                lidar_timestamp_list=lidar_timestamps,
                system_scan_period_sec=min(
                    params.system_scan_period_sec, params.camera_scan_period_sec
                ),
                max_camera_jitter_sec=params.max_camera_jitter_sec,
                num_load_frames=num_load_lidar_frames,
                msg_display_interval=10**9,
            )
        else:
            num_load_cam_frames = int(
                num_load_lidar_frames
                * params.system_scan_period_sec
                / params.camera_scan_period_sec
            )
            synced_frame_info = misc_utils.get_lidar_camera_frame_info_async(
                image_timestamp_list=image_timestamps,
                lidar_timestamp_list=lidar_timestamps,
                max_camera_jitter=params.max_camera_jitter_sec,
                camera_scan_period=params.camera_scan_period_sec,
                num_load_image_frames=num_load_cam_frames,
                num_load_lidar_frames=num_load_lidar_frames,
                msg_display_interval=10**9,
            )

    matched = sum(
        1
        for image_index, lidar_index, _ in synced_frame_info
        if image_index is not None and lidar_index is not None
    )
    image_drops = sum(
        1
        for image_index, lidar_index, _ in synced_frame_info
        if image_index is None and lidar_index is not None
    )
    lidar_drops = sum(
        1
        for image_index, lidar_index, _ in synced_frame_info
        if image_index is not None and lidar_index is None
    )

    diffs = [
        image_timestamps[image_index] - lidar_timestamps[lidar_index]
        for image_index, lidar_index, _ in synced_frame_info
        if image_index is not None and lidar_index is not None
    ]

    return {
        "channel": camera_sensor["channel"],
        "topic": camera_sensor["topic"],
        "raw_images_after_start": len(image_timestamps),
        "matched_frames": matched,
        "camera_drops_vs_lidar": image_drops,
        "lidar_without_camera_match": lidar_drops,
        "effective_output_frames": matched + image_drops,
        "camera_drop_rate_vs_lidar": (
            image_drops / num_load_lidar_frames if num_load_lidar_frames else 0.0
        ),
        "mean_cam_minus_lidar_sec": sum(diffs) / len(diffs) if diffs else None,
        "min_cam_minus_lidar_sec": min(diffs) if diffs else None,
        "max_cam_minus_lidar_sec": max(diffs) if diffs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to conversion config yaml")
    parser.add_argument(
        "--bag-path",
        default="",
        help="optional explicit rosbag directory; otherwise auto-resolve from input_base",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    param_args = {
        "task": config_dict["task"],
        "scene_description": config_dict["description"]["scene"],
        **config_dict["conversion"],
    }
    if args.bag_path:
        param_args["input_bag_path"] = args.bag_path
    params = Rosbag2ConverterParams(**param_args)
    bag_path = _resolve_bag_path(params)

    reader = Rosbag2Reader(
        bag_path,
        with_world_frame_conversion=(params.world_frame_id != "base_link"),
        with_sensor_frame_conversion=(
            len(params.camera_sensors) > 0 or len(params.radar_sensors) > 0
        ),
    )

    start_timestamp = (
        params.start_timestamp_sec
        if params.start_timestamp_sec > 0.0
        else reader.start_timestamp + params.skip_timestamp
    )
    lidar_start_time = rosbag2_utils.unix_timestamp_to_stamp(start_timestamp)
    lidar_timestamps = _load_timestamps(
        reader=reader,
        topic=params.lidar_sensor["topic"],
        start_time=lidar_start_time,
        offset_sec=params.lidar_latency_sec,
    )

    if len(lidar_timestamps) == 0:
        raise ValueError(f"No lidar frames found for topic {params.lidar_sensor['topic']}")

    camera_topic_counts = [reader.get_topic_count(camera["topic"]) for camera in params.camera_sensors]
    num_frames_in_bag = min(
        camera_topic_counts + [reader.get_topic_count(params.lidar_sensor["topic"])]
    )
    num_frames_to_skip = int(params.skip_timestamp * 10)
    max_num_load_frames = num_frames_in_bag - num_frames_to_skip - 1
    num_load_lidar_frames = (
        max_num_load_frames
        if params.num_load_frames <= 0 or params.num_load_frames > max_num_load_frames
        else params.num_load_frames
    )
    num_load_lidar_frames = min(num_load_lidar_frames, len(lidar_timestamps))
    lidar_timestamps = lidar_timestamps[:num_load_lidar_frames]

    camera_start_timestamp = start_timestamp
    results = [
        _summarize_camera(
            params=params,
            reader=reader,
            lidar_timestamps=lidar_timestamps,
            num_load_lidar_frames=num_load_lidar_frames,
            camera_sensor=camera_sensor,
            camera_start_timestamp=camera_start_timestamp,
        )
        for camera_sensor in params.camera_sensors
    ]

    print(f"bag_path: {bag_path}")
    print(f"lidar_topic: {params.lidar_sensor['topic']}")
    print(f"num_load_lidar_frames: {num_load_lidar_frames}")
    print("")
    print("channel raw_images matched camera_drops output_frames drop_rate mean_dt_sec min_dt_sec max_dt_sec")
    for result in results:
        mean_dt = (
            f"{result['mean_cam_minus_lidar_sec']:.6f}"
            if result["mean_cam_minus_lidar_sec"] is not None
            else "n/a"
        )
        min_dt = (
            f"{result['min_cam_minus_lidar_sec']:.6f}"
            if result["min_cam_minus_lidar_sec"] is not None
            else "n/a"
        )
        max_dt = (
            f"{result['max_cam_minus_lidar_sec']:.6f}"
            if result["max_cam_minus_lidar_sec"] is not None
            else "n/a"
        )
        print(
            f"{result['channel']} "
            f"{result['raw_images_after_start']} "
            f"{result['matched_frames']} "
            f"{result['camera_drops_vs_lidar']} "
            f"{result['effective_output_frames']} "
            f"{result['camera_drop_rate_vs_lidar']:.3%} "
            f"{mean_dt} {min_dt} {max_dt}"
        )


if __name__ == "__main__":
    main()
