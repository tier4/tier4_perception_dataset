from __future__ import annotations

import struct
from typing import Any

__all__ = ["parse_traffic_signals"]


def parse_traffic_signals(buffer: bytes) -> dict[str, Any] | None:
    """Return a `TrafficLightGroupArray` as recorded by Autoware v4.2.1, read from raw CDR.

    v4.2.1 has no `predictions` field, so the message classes built from newer
    autoware_msgs can't deserialize it.

    Args:
        buffer (bytes): Serialized message, with its 4-byte encapsulation header.

    Returns:
        `{"stamp": (sec, nanosec), "groups": [(group_id, [(color, shape, status, confidence)])]}`,
        or None if the buffer is not exactly this layout.
    """
    reader = _CdrReader(buffer)
    try:
        stamp = (reader.u32(), reader.u32())
        groups = []
        for _ in range(reader.u32()):
            group_id = reader.i64()
            elements = [
                (reader.u8(), reader.u8(), reader.u8(), reader.f32()) for _ in range(reader.u32())
            ]
            groups.append((group_id, elements))
    except struct.error:
        return None
    return {"stamp": stamp, "groups": groups} if reader.at_end() else None


class _CdrReader:
    """Little-endian CDR reader; alignment counts from the end of the encapsulation header."""

    def __init__(self, buffer: bytes) -> None:
        self.buffer = buffer
        self.position = 4

    def _read(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        self.position += -(self.position - 4) % size
        (value,) = struct.unpack_from(fmt, self.buffer, self.position)
        self.position += size
        return value

    def u8(self) -> int:
        return self._read("<B")

    def u32(self) -> int:
        return self._read("<I")

    def i64(self) -> int:
        return self._read("<q")

    def f32(self) -> float:
        return self._read("<f")

    def at_end(self) -> bool:
        return self.position == len(self.buffer)
