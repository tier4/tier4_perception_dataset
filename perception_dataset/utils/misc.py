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
    ] = []  # [image_index, lidar_frame_index, dummy_timestamp (None if not dummy)]
    lidar_frame_index: int = 0
    prev_frame_unix_timestamp = lidar_timestamp_list[0] + camera_latency_sec

    for image_index, image_unix_timestamp in enumerate(image_timestamp_list):
        if lidar_frame_index >= num_load_frames:
            break

        # Get LiDAR data
        lidar_unix_timestamp = lidar_timestamp_list[lidar_frame_index]

        # Address image drop
        while (image_unix_timestamp - prev_frame_unix_timestamp) > timestamp_diff:
            # Add dummy image timestamp in synced_frame_info_list
            dummy_image_timestamp = image_unix_timestamp
            while (dummy_image_timestamp - prev_frame_unix_timestamp) > timestamp_diff:
                dummy_image_timestamp -= 0.1
            synced_frame_info_list.append([None, lidar_frame_index, dummy_image_timestamp])

            # Increment LiDAR information
            lidar_frame_index += 1
            prev_frame_unix_timestamp = dummy_image_timestamp
            if lidar_frame_index >= num_load_frames:
                return synced_frame_info_list

            lidar_unix_timestamp = lidar_timestamp_list[lidar_frame_index]

        time_diff_from_lidar = image_unix_timestamp - lidar_unix_timestamp
        if not accept_frame_drop and time_diff_from_lidar > (camera_latency_sec + timestamp_diff):
            raise ValueError(
                f"Topic message may be dropped at [{lidar_frame_index}]: lidar_timestamp={lidar_unix_timestamp} image_timestamp={image_unix_timestamp}"
            )

        if lidar_frame_index % msg_display_interval == 0:
            print(
                f"frame{lidar_frame_index}, stamp = {image_unix_timestamp}, diff cam - lidar = {time_diff_from_lidar:0.3f} sec"
            )

        synced_frame_info_list.append([image_index, lidar_frame_index, None])
        lidar_frame_index += 1
        prev_frame_unix_timestamp = image_unix_timestamp

    return synced_frame_info_list
