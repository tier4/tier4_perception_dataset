import os
from typing import List, Optional, Tuple

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
) -> List[Tuple[Optional[int], Optional[int], Optional[float]]]:
    """
    Get synced frame info list for lidar and camera.
    LiDAR scan with t_lidar and image with t_image are synced if
        (T_image - T_lidar - T_latency) > -(T_system/2 + max_camera_jitter)
            and
        (T_image - T_lidar - T_latency) <= (T_system/2 + max_camera_jitter)

    Args:
        image_timestamp_list: image timestamp list
        lidar_timestamp_list: lidar timestamp list
        lidar_to_camera_latency_sec: camera latency in seconds between the header.stamp and shutter trigger
        system_scan_period_sec: system scan period in seconds
        num_load_frames: the number of frames to be loaded. if the value isn't positive, read all messages.
        msg_display_interval: display interval for messages
    Returns:
        List[Tuple[Optional[int], Optional[int], Optional[float]]]:
            A list where each entry represents a matched or unmatched frame pair:
            - image_index: index of the image frame (or None if missing)
            - lidar_index: index of the LiDAR frame (or None if missing)
            - dummy_timestamp: an adjusted timestamp when image frame is missing; None if image is present
    """
    synced_frame_info_list: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []

    # calculate nominal delay between lidar and camera

    current_image_index: int = 0
    for lidar_index, lidar_timestamp in enumerate(lidar_timestamp_list):
        if lidar_index >= num_load_frames:
            break
        if current_image_index >= len(image_timestamp_list):
            break
        image_timestamp = image_timestamp_list[current_image_index]

        # T_image > T_lidar + T_latency - (T_system/2 + max_camera_jitter)
        while image_timestamp - lidar_timestamp < lidar_to_camera_latency_sec - (
            system_scan_period_sec / 2 + max_camera_jitter_sec
        ):
            # increase image index until the condition is met
            current_image_index += 1
            if current_image_index >= len(image_timestamp_list):
                # set dummy timestamp
                image_timestamp = (
                    lidar_timestamp
                    + lidar_to_camera_latency_sec
                    + max_camera_jitter_sec
                    + system_scan_period_sec
                )
                break
            image_timestamp = image_timestamp_list[current_image_index]

        # T_image <= T_lidar + T_latency + (T_system/2 + max_camera_jitter)
        if image_timestamp - lidar_timestamp >= lidar_to_camera_latency_sec + (
            system_scan_period_sec / 2 + max_camera_jitter_sec
        ):
            # If the image timestamp is larger than above condition, assume the image is dropped
            dummy_timestamp = image_timestamp - system_scan_period_sec
            synced_frame_info_list.append((None, lidar_index, dummy_timestamp))
            continue

        synced_frame_info_list.append([current_image_index, lidar_index, None])
        current_image_index += 1

        if lidar_index % msg_display_interval == 0:
            print(
                f"frame{lidar_index}, stamp = {image_timestamp}, diff cam - lidar = {image_timestamp - lidar_timestamp:0.3f} sec"
            )
    return synced_frame_info_list


def get_lidar_camera_frame_info_async(
    image_timestamp_list: List[float],
    lidar_timestamp_list: List[float],
    lidar_to_camera_latency: float = 0.0,
    max_camera_jitter: float = 0.03,
    camera_scan_period: float = 0.1,
    num_load_image_frames: int = 0,
    num_load_lidar_frames: int = 0,
    msg_display_interval: int = 100,
) -> List[Tuple[Optional[int], Optional[int], Optional[float]]]:
    """
    Asynchronously synchronizes camera image frames with LiDAR frames based on their timestamps.

    This function assumes that the camera frame period is equal to or shorter than the LiDAR frame period.
    The first frame is always aligned to a LiDAR timestamp for synchronization. After that,
    camera and LiDAR frames proceed independently, and the function attempts to pair them based on
    their expected time relationship while tolerating latency and jitter.

    Args:
        image_timestamp_list (List[float]): List of timestamps for camera image frames (in seconds).
        lidar_timestamp_list (List[float]): List of timestamps for LiDAR frames (in seconds).
        lidar_to_camera_latency (float): Latency (in seconds) from the LiDAR timestamp to the camera shutter trigger.
        max_camera_jitter (float): Maximum expected jitter in the camera timestamps (in seconds).
        camera_scan_period (float): Expected time interval between camera captures (in seconds).
        num_load_image_frames (int): Number of image frames to process. If <= 0, process all frames.
        num_load_lidar_frames (int): Number of LiDAR frames to process. If <= 0, process all frames.
        msg_display_interval (int): Number of frames between printed progress messages.

    Returns:
        List[Tuple[Optional[int], Optional[int], Optional[float]]]:
            A list where each entry represents a matched or unmatched frame pair:
            - image_index: index of the image frame (or None if missing)
            - lidar_index: index of the LiDAR frame (or None if missing)
            - dummy_timestamp: an adjusted timestamp for the image frame when it is missing; None if the image is present.
    """
    synced_frame_info_list: List[Tuple[Optional[int], Optional[int], Optional[float]]] = []

    lidar_index = 0
    image_index = 0
    lidar_load_completed = False
    image_load_completed = False
    first_frame_loaded = False
    half_camera_period_with_jitter = camera_scan_period / 2 + max_camera_jitter

    while lidar_index < num_load_lidar_frames or image_index < num_load_image_frames:
        if lidar_index < num_load_lidar_frames:
            lidar_timestamp = lidar_timestamp_list[lidar_index]
        else:
            lidar_load_completed = True
            lidar_timestamp = lidar_timestamp_list[-1]

        if image_index < num_load_image_frames:
            image_timestamp = image_timestamp_list[image_index]
        else:
            image_load_completed = True
            image_timestamp = image_timestamp_list[-1]

        adjusted_image_timestamp = image_timestamp - lidar_to_camera_latency

        if lidar_load_completed:
            # All LiDAR frames are loaded
            synced_frame_info_list.append((image_index, None, None))
            image_index += 1
        elif image_load_completed:
            # All image frames are loaded
            dummy_timestamp = lidar_timestamp + lidar_to_camera_latency
            synced_frame_info_list.append((None, lidar_index, dummy_timestamp))
            lidar_index += 1
        elif adjusted_image_timestamp < lidar_timestamp - half_camera_period_with_jitter:
            # Image timestamp is too early; assume LiDAR frame is missing
            if first_frame_loaded:
                synced_frame_info_list.append((image_index, None, None))
                image_index += 1
            else:
                # Drop current image timestamp and try next to find the first aligned frame
                image_timestamp_list = image_timestamp_list[1:]
        elif adjusted_image_timestamp > lidar_timestamp + half_camera_period_with_jitter:
            # LiDAR timestamp is too early; assume image frame is missing
            dummy_timestamp = lidar_timestamp + lidar_to_camera_latency
            synced_frame_info_list.append((None, lidar_index, dummy_timestamp))
            lidar_index += 1
            first_frame_loaded = True
        else:
            # Both image and LiDAR frames are available and matched
            synced_frame_info_list.append((image_index, lidar_index, None))
            image_index += 1
            lidar_index += 1
            first_frame_loaded = True

        if image_index % msg_display_interval == 0:
            print(
                f"frame{lidar_index}, stamp = {image_timestamp}, diff cam - lidar = {image_timestamp - lidar_timestamp:0.3f} sec"
            )

    return synced_frame_info_list
