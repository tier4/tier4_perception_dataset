import os
from typing import List

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME


def unix_timestamp_to_nusc_timestamp(timestamp: float) -> int:
    return int(timestamp * 1e6)


def nusc_timestamp_to_unix_timestamp(timestamp: int) -> float:
    return float(timestamp) * 1e-6


def get_sample_data_filename(sensor_channel: str, frame_index: int, fileformat: str):
    filename = os.path.join(
        T4_FORMAT_DIRECTORY_NAME.DATA.value,
        sensor_channel,
        f"{frame_index:05}.{fileformat}",
    )
    return filename


def get_lidar_camera_synced_frame_info(
    image_timestamp_list: List[float],
    lidar_timestamp_list: List[float],
    camera_latency_sec: float = 0.0,
    accept_frame_drop: bool = False,
    timestamp_diff: float = 0.15,
    num_load_frames: int = 0,
    msg_display_interval: int = 100,
):
    synced_frame_info_list: List[
        int, int, float
    ] = []  # [image_index, lidar_frame_index, dummy_timestamp (None if image is not dropped)]

    current_image_index: int = 0
    for lidar_index, lidar_timestamp in enumerate(lidar_timestamp_list):
        if lidar_index >= num_load_frames:
            break
        image_timestamp = image_timestamp_list[current_image_index]

        while image_timestamp - lidar_timestamp < timestamp_diff + camera_latency_sec - 0.1:
            if not accept_frame_drop:
                raise ValueError(
                    f"LiDAR message may be dropped at image_timestamp={image_timestamp}"
                )
            current_image_index += 1
            image_timestamp = image_timestamp_list[current_image_index]

        if image_timestamp - lidar_timestamp > timestamp_diff + camera_latency_sec:
            if not accept_frame_drop:
                raise ValueError(
                    f"Image message may be dropped at lidar_timestamp={lidar_timestamp}"
                )
            dummy_timestamp = image_timestamp - 0.1
            synced_frame_info_list.append([None, lidar_index, dummy_timestamp])
            continue

        synced_frame_info_list.append([current_image_index, lidar_index, None])
        current_image_index += 1

        if lidar_index % msg_display_interval == 0:
            print(
                f"frame{lidar_index}, stamp = {image_timestamp}, diff cam - lidar = {image_timestamp - lidar_timestamp:0.3f} sec"
            )
    return synced_frame_info_list
