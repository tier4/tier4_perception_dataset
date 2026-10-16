from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from t4_devkit.schema.v1 import Hardware
import yaml

__all__ = ["VehicleModel", "VehicleTable"]


class VehicleModel:
    """One vehicle model of a `vehicles.yaml`."""

    def __init__(self, name: str, entry: dict[str, Any]) -> None:
        self.name = name
        self.entry = entry

    def matches(self, vehicle_name: str) -> bool:
        """Return whether a vehicle name matches one of the model's `vehicle_names` patterns."""
        return any(fnmatch(vehicle_name, pattern) for pattern in self.entry["vehicle_names"])

    def vehicle_fields(self) -> dict[str, Any]:
        """Return the 1.0 `vehicle` fields of the model.

        Dimensions are derived from `vehicle_info`; they are left out when the model has none.
        """
        fields = {"model": self.name, "sensor_rig_id": self.entry.get("sensor_kit")}
        info = self.entry.get("vehicle_info")
        if not info:
            return fields
        length = info["front_overhang"] + info["wheel_base"] + info["rear_overhang"]
        width = info["left_overhang"] + info["wheel_tread"] + info["right_overhang"]
        return fields | {
            "length": round(length, 6),
            "width": round(width, 6),
            "height": info["vehicle_height"],
            "wheel_base": info["wheel_base"],
            "rear_axle_to_center": round(length / 2 - info["rear_overhang"], 6),
        }

    def camera_hardware(self, camera: str) -> Hardware | None:
        """Return the hardware of a camera (`camera0`, ...): its own entry, else `default`.

        Args:
            camera (str): Camera name, as in its topic.

        Returns:
            The hardware, or None if the model lists neither.
        """
        cameras = self.entry.get("cameras") or {}
        hardware = cameras.get(camera, cameras.get("default"))
        if hardware is None:
            return None
        return Hardware(
            make=hardware.get("make"),
            model=hardware.get("model"),
            serial_number=None,
            firmware_version=None,
        )


class VehicleTable:
    """Vehicle models of a `vehicles.yaml`: the public sample or the perception-devops one."""

    def __init__(self, models: list[VehicleModel]) -> None:
        self.models = models

    @classmethod
    def load(cls, path: str | Path) -> VehicleTable:
        """Return the table read from a `vehicles.yaml`."""
        with open(path) as f:
            entries = yaml.safe_load(f) or {}
        return cls([VehicleModel(name, entry) for name, entry in entries.items()])

    def find(self, vehicle_name: str) -> VehicleModel | None:
        """Return the model of a vehicle name, or None if no model matches.

        Raises:
            ValueError: If several models match.
        """
        found = [model for model in self.models if model.matches(vehicle_name)]
        if len(found) > 1:
            raise ValueError(
                f"{vehicle_name} matches several models: {[model.name for model in found]}"
            )
        return found[0] if found else None
