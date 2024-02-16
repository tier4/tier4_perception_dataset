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
    lidar_to_camera_latency_sec: float = 0.0,
    system_scan_period_sec: float = 0.1,
    max_camera_jitter_sec: float = 0.03,
    num_load_frames: int = 0,
    msg_display_interval: int = 100,
):
    """
    Get synced frame info list for lidar and camera.
    LiDAR scan with t_lidar and image with t_image are synced if
        t_image - t_lidar >= lidar_to_camera_latency_sec - (system_scan_period_sec - max_camera_jitter_sec)
            and
        t_image - t_lidar <= lidar_to_camera_latency_sec + (system_scan_period_sec - max_camera_jitter_sec)

    Args:
        image_timestamp_list: image timestamp list
        lidar_timestamp_list: lidar timestamp list
        lidar_to_camera_latency_sec: camera latency in seconds between the header.stamp and shutter trigger
        system_scan_period_sec: system scan period in seconds
        num_load_frames: the number of frames to be loaded. if the value isn't positive, read all messages.
        msg_display_interval: display interval for messages
    Return:
        synced_frame_info_list: synced frame info list
            [[image_index, lidar_frame_index, dummy_image_timestamp (None if image is not dropped)]]
    """
    synced_frame_info_list: List[int, int, float] = []

    # calculate nominal delay between lidar and camera

    current_image_index: int = 0
    for lidar_index, lidar_timestamp in enumerate(lidar_timestamp_list):
        if lidar_index >= num_load_frames:
            break
        if current_image_index >= len(image_timestamp_list):
            break
        image_timestamp = image_timestamp_list[current_image_index]

        while image_timestamp - lidar_timestamp < lidar_to_camera_latency_sec - (
            system_scan_period_sec - max_camera_jitter_sec
        ):
            current_image_index += 1
            image_timestamp = image_timestamp_list[current_image_index]

        if image_timestamp - lidar_timestamp > lidar_to_camera_latency_sec + (
            system_scan_period_sec - max_camera_jitter_sec
        ):
            # Image is dropped
            dummy_timestamp = image_timestamp - system_scan_period_sec
            synced_frame_info_list.append([None, lidar_index, dummy_timestamp])
            continue

        synced_frame_info_list.append([current_image_index, lidar_index, None])
        current_image_index += 1

        if lidar_index % msg_display_interval == 0:
            print(
                f"frame{lidar_index}, stamp = {image_timestamp}, diff cam - lidar = {image_timestamp - lidar_timestamp:0.3f} sec"
            )
    return synced_frame_info_list
