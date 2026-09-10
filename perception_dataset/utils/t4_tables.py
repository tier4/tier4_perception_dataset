"""Lookup helpers over the raw T4 annotation tables.

Several converters need to resolve which sensor channel a ``sample_data``
record belongs to (via its ``calibrated_sensor``) and to pick the lidar channel
that carries the per-frame ego pose. The joins are small but easy to get subtly
wrong, so they live here.
"""

from typing import Dict, List, Optional

from perception_dataset.constants import LIDAR_CONCAT_CHANNEL


def channel_by_calibrated_sensor(
    sensor: list, calibrated_sensor: list
) -> Dict[str, Optional[str]]:
    """Map calibrated-sensor tokens to their sensor channel names.

    Args:
        sensor (list): Sensor table records.
        calibrated_sensor (list): Calibrated-sensor table records.

    Returns:
        Dict[str, Optional[str]]: A mapping from each calibrated-sensor token to
            its channel name, or ``None`` when its sensor token is unknown.
    """
    token_to_channel = {s["token"]: s["channel"] for s in sensor}
    return {c["token"]: token_to_channel.get(c["sensor_token"]) for c in calibrated_sensor}


def records_for_channel(
    sample_data: list, channel_by_calib: Dict[str, Optional[str]], channel: str
) -> List[dict]:
    """Get the sample-data records for a sensor channel.

    Args:
        sample_data (list): Sample-data table records.
        channel_by_calib (Dict[str, Optional[str]]): Mapping from
            calibrated-sensor tokens to channel names.
        channel (str): Sensor channel to select.

    Returns:
        List[dict]: Matching sample-data records sorted by timestamp.
    """
    return sorted(
        (
            record
            for record in sample_data
            if channel_by_calib.get(record["calibrated_sensor_token"]) == channel
        ),
        key=lambda record: record["timestamp"],
    )


def select_lidar_channel(
    sensor: list, channel_by_calib: Dict[str, Optional[str]], sample_data: list
) -> str:
    """Select the lidar channel used to retrieve ego poses.

    Args:
        sensor (list): Sensor table records.
        channel_by_calib (Dict[str, Optional[str]]): Mapping from
            calibrated-sensor tokens to channel names.
        sample_data (list): Sample-data table records.

    Returns:
        str: ``LIDAR_CONCAT`` when present; otherwise, the first lidar channel
            alphabetically, or ``LIDAR_CONCAT`` when no lidar sensor exists.
    """
    channels = {channel_by_calib.get(r["calibrated_sensor_token"]) for r in sample_data}
    if LIDAR_CONCAT_CHANNEL in channels:
        return LIDAR_CONCAT_CHANNEL
    lidar_channels = sorted(s["channel"] for s in sensor if s.get("modality") == "lidar")
    return lidar_channels[0] if lidar_channels else LIDAR_CONCAT_CHANNEL
