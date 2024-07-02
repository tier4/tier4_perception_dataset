from __future__ import annotations

from numbers import Number

__all__ = ("us2sec", "sec2us")


def us2sec(timestamp: Number) -> float:
    """Convert timestamp from micro seconds [us] to seconds [s].

    Args:
        timestamp (Number): Timestamp in [us].

    Returns:
        Timestamp in [s].
    """
    return 1e-6 * timestamp


def sec2us(timestamp: Number) -> float:
    """Convert timestamp from seconds [s] to micro seconds [us].

    Args:
        timestamp (Number): Timestamp in [s].

    Returns:
        Timestamp in [us].
    """
    return 1e6 * timestamp
