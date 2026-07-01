from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .geometry import RigidTransform, compose, identity, load_transform_yaml


@dataclass(frozen=True)
class CalibrationSet:
    root: Path
    default_root: Path
    vehicle_id: str
    sensor_model: str
    base_frame: str = "base_link"
    sensor_kit_frame: str = "sensor_kit_base_link"
    transforms: dict[tuple[str, str], RigidTransform] | None = None

    def pandar_csv(self, lidar_name: str) -> Path:
        path = self.root / "pandar" / f"{lidar_name}.csv"
        if not path.exists():
            path = self.default_root / "pandar" / f"{lidar_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing Pandar calibration CSV: {path}")
        return path

    def get_transform(self, parent: str, child: str) -> RigidTransform | None:
        transforms = self.transforms or {}
        if (parent, child) in transforms:
            return transforms[(parent, child)]
        if (child, parent) in transforms:
            return transforms[(child, parent)].inverse()
        return None

    def base_to_frame(self, frame_id: str) -> RigidTransform:
        if frame_id == self.base_frame:
            return identity(self.base_frame, self.base_frame)
        direct = self.get_transform(self.base_frame, frame_id)
        if direct is not None:
            return direct
        via_sensor_kit = self.get_transform(self.sensor_kit_frame, frame_id)
        base_to_sensor_kit = self.get_transform(self.base_frame, self.sensor_kit_frame)
        if via_sensor_kit is not None and base_to_sensor_kit is not None:
            return compose(base_to_sensor_kit, via_sensor_kit)
        raise KeyError(f"No transform from {self.base_frame} to {frame_id} in {self.root}")

    def base_to_lidar(self, lidar_name: str) -> RigidTransform:
        base_link = f"{lidar_name}/lidar_base_link"
        lidar_frame = f"{lidar_name}/lidar"
        base_to_lidar_base = self.base_to_frame(base_link)
        lidar_base_to_lidar = self.get_transform(base_link, lidar_frame)
        if lidar_base_to_lidar is None:
            # Individual params often stop at lidar_base_link. The packet frame in the reference
            # converter is the base link, so use that as a deterministic fallback.
            return base_to_lidar_base
        return compose(base_to_lidar_base, lidar_base_to_lidar)


def resolve_calibration(
    individual_params_root: str | Path,
    vehicle_id: str,
    sensor_model: str,
    *,
    base_frame: str = "base_link",
    sensor_kit_frame: str = "sensor_kit_base_link",
) -> CalibrationSet:
    params_root = Path(individual_params_root)
    config_root = params_root / "individual_params" / "config"
    if not config_root.exists():
        config_root = params_root / "config"
    vehicle_root = config_root / vehicle_id / sensor_model
    if not vehicle_root.exists():
        vehicle_root = config_root / "default" / sensor_model
    if not vehicle_root.exists():
        raise FileNotFoundError(
            f"Could not resolve individual params for vehicle_id={vehicle_id}, "
            f"sensor_model={sensor_model} under {params_root}"
        )

    transforms: dict[tuple[str, str], RigidTransform] = {}
    for filename in ("sensors_calibration.yaml", "sensor_kit_calibration.yaml"):
        path = vehicle_root / filename
        if not path.exists():
            continue
        for children in load_transform_yaml(path).values():
            for transform in children.values():
                transforms[(transform.parent, transform.child)] = transform

    return CalibrationSet(
        root=vehicle_root,
        default_root=config_root / "default" / sensor_model,
        vehicle_id=vehicle_id,
        sensor_model=sensor_model,
        base_frame=base_frame,
        sensor_kit_frame=sensor_kit_frame,
        transforms=transforms,
    )
