import os

from perception_dataset.constants import T4_FORMAT_DIRECTORY_NAME


def unix_timestamp_to_nusc_timestamp(timestamp: float) -> int:
    return int(timestamp * 1e6)


def nusc_timestamp_to_unix_timestamp(timestamp: int) -> float:
    return float(timestamp) * 1e-6


def get_sample_data_filename(sensor_channel: str, frame_index: int, fileformat: str):
    filename = os.path.join(
        T4_FORMAT_DIRECTORY_NAME.DATA.value,
        sensor_channel,
        f"{frame_index:04}.{fileformat}",
    )
    return filename
