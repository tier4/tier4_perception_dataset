"""Lookup helpers over the raw T4 annotation tables, shared by the Kognic
converters.

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
    """Map each ``calibrated_sensor`` token to its sensor channel name."""
    token_to_channel = {s["token"]: s["channel"] for s in sensor}
    return {
        c["token"]: token_to_channel.get(c["sensor_token"]) for c in calibrated_sensor
    }


def records_for_channel(
    sample_data: list, channel_by_calib: Dict[str, Optional[str]], channel: str
) -> List[dict]:
    """Return the *channel*'s ``sample_data`` records sorted by timestamp."""
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
    """Pick the lidar channel that carries ego poses (LIDAR_CONCAT if present)."""
    channels = {channel_by_calib.get(r["calibrated_sensor_token"]) for r in sample_data}
    if LIDAR_CONCAT_CHANNEL in channels:
        return LIDAR_CONCAT_CHANNEL
    lidar_channels = sorted(s["channel"] for s in sensor if s.get("modality") == "lidar")
    return lidar_channels[0] if lidar_channels else LIDAR_CONCAT_CHANNEL
