from __future__ import annotations

import json
import os.path as osp
import re
from typing import Any

from t4_devkit.lanelet import LaneletParser
from t4_devkit.schema.v1 import (
    DistortionModel,
    Manifest,
    Route,
    SchemaName,
    VehicleState,
)
from t4_devkit.schema.v1.migrate_v0 import V0Migrator

from perception_dataset.t4_v1.bag_metadata import BagMetadata
from perception_dataset.t4_v1.vehicle_table import VehicleTable

__all__ = ["Backfill", "backfill_datasets"]


def backfill_datasets(conversion: dict[str, Any]) -> None:
    """Migrate each dataset of the task config's `conversion` block, with metadata backfill.

    Reads `<input_base>/<dataset_id>/<version>` and writes
    `<output_base>/<dataset_id>/<dataset_version>`. The bags are the dataset's `input_bag/`
    (if any) and its `bags`.
    """
    vehicle_table = VehicleTable.load(conversion["vehicle_table"])
    for metadata in conversion["datasets"]:
        dataset_id = metadata["dataset_id"]
        data_root = osp.join(conversion["input_base"], dataset_id, str(metadata["version"]))
        output_root = osp.join(
            conversion["output_base"], dataset_id, str(metadata["dataset_version"])
        )
        input_bag = osp.join(data_root, "input_bag")
        bag_dirs = [input_bag] if osp.isdir(input_bag) else []
        backfill = Backfill(
            data_root, output_root, metadata, vehicle_table, bag_dirs + metadata.get("bags", [])
        )
        backfill.run(
            dataset_id,
            metadata["dataset_version"],
            keep_images=conversion.get("keep_images", False),
        )


class Backfill:
    """Migrate a 0.x dataset to 1.0, filling the metadata 0.x never stored.

    - Web.Auto values (project, map ids, log files, ...) come from the task config, as given.
    - Vehicle model and dimensions, camera hardware: the vehicle table.
    - Ego twist / acceleration / GNSS, vehicle state, routes, LiDAR hardware, distortion models
      and the Autoware version: the rosbags.

    `traffic_light` is not filled from the rosbags: recognition results are model output, not
    ground truth.

    Values without a source keep their 1.0 placeholders; `notes` (written to
    `manifest.migration_notes`) say what was filled and what was dropped.
    """

    def __init__(
        self,
        data_root: str,
        output_root: str,
        metadata: dict[str, Any],
        vehicle_table: VehicleTable,
        bag_dirs: list[str],
    ) -> None:
        """
        Args:
            data_root (str): 0.x dataset root, `<DATASET_ID>/<VERSION>`.
            output_root (str): 1.0 dataset root to create.
            metadata (dict[str, Any]): The dataset's block of the task config.
            vehicle_table (VehicleTable): Vehicle models.
            bag_dirs (list[str]): Rosbags of the recording (`input_bag/` and extra ones).
        """
        self.data_root = data_root
        self.metadata = metadata
        self.vehicle_table = vehicle_table
        self.migrator = V0Migrator(data_root, output_root)
        self.bags = BagMetadata(bag_dirs)
        self.notes: list[str] = []
        self.converter_params = self._converter_params()
        self.topics = self._channel_topics()
        samples = self.migrator.tables(SchemaName.SAMPLE)
        self.sample_times = sorted(sample.timestamp for sample in samples)

    def run(self, dataset_id: str, version: int, *, keep_images: bool = False) -> Manifest:
        """Fill every table, then migrate.

        Returns:
            The manifest written.
        """
        self.fill_vehicle()
        self.fill_map()
        self.fill_sensors()
        self.fill_ego_poses()
        self.fill_vehicle_states()
        lanelet_map = self._lanelet_map()
        self.fill_routes(lanelet_map)
        return self.migrator.migrate(
            dataset_id, version, keep_images=keep_images, manifest_fields=self.manifest_fields()
        )

    def fill_vehicle(self) -> None:
        """Set the vehicle name and, if the vehicle table has its model, model and dimensions."""
        name = self.metadata.get("vehicle_name")
        if name is None:
            self.notes.append("vehicle: no vehicle_name given, placeholder kept")
            return
        model = self.vehicle_table.find(name)
        for vehicle in self.migrator.tables(SchemaName.VEHICLE):
            vehicle.name = name
            if model is None:
                continue
            for field, value in model.vehicle_fields().items():
                setattr(vehicle, field, value)
        if model is None:
            self.notes.append(f"vehicle: {name} not in the vehicle table, placeholder kept")
        elif "length" not in model.vehicle_fields():
            self.notes.append(f"vehicle: no dimensions for {model.name}, placeholders kept")

    def fill_map(self) -> None:
        """Set the map ids and name, and the log location, from the config."""
        map_info = self.metadata.get("map") or {}
        for map_record in self.migrator.tables(SchemaName.MAP):
            for field in ("map_id", "version_id", "name"):
                if map_info.get(field) is not None:
                    setattr(map_record, field, str(map_info[field]))
        for log in self.migrator.tables(SchemaName.LOG):
            log.location = self.metadata.get("location", log.location)

    def fill_sensors(self) -> None:
        """Set LiDAR hardware (`/diagnostics`, channel `LIDAR_<POSITION>`), camera hardware
        (vehicle table, by the camera in the channel's topic) and camera distortion models
        (`camera_info`, unless the 0.x converter undistorted the images)."""
        lidars = self.bags.lidar_hardware()
        distortion_models = self.bags.distortion_models()
        if self.converter_params.get("_undistort_image"):
            self.notes.append(
                "cameras: images were undistorted by the 0.x converter, distortion model kept"
            )
            distortion_models = {}
        model = self.vehicle_table.find(self.metadata.get("vehicle_name") or "")
        for sensor in self.migrator.tables(SchemaName.SENSOR):
            if sensor.lidar is not None:
                position = sensor.channel.removeprefix("LIDAR_").lower()
                sensor.hardware = lidars.get(position, sensor.hardware)
                continue
            if sensor.camera is None or sensor.channel not in self.topics:
                continue
            camera = self.topics[sensor.channel].split("/")[3]  # /sensing/camera/<camera>/...
            if model is not None:
                sensor.hardware = model.camera_hardware(camera) or sensor.hardware
            distortion_model = distortion_models.get(camera)
            if distortion_model is None or distortion_model == sensor.camera.distortion_model:
                continue
            if DistortionModel(distortion_model).num_coefficients != len(sensor.camera.distortion):
                self.notes.append(
                    f"{sensor.channel}: camera_info says {distortion_model}, which does not fit "
                    f"its {len(sensor.camera.distortion)} coefficients; kept "
                    f"{sensor.camera.distortion_model.value}"
                )
                continue
            sensor.camera.distortion_model = distortion_model

    def fill_ego_poses(self) -> None:
        """Set ego pose twist, covariance, acceleration and geocoordinate where 0.x has none."""
        poses = self.migrator.tables(SchemaName.EGO_POSE)
        fields = self.bags.ego_pose_fields([pose.timestamp for pose in poses])
        filled = dict.fromkeys(("twist", "covariance", "acceleration", "geocoordinate"), 0)
        for pose, values in zip(poses, fields):
            for field, value in values.items():
                if getattr(pose, field) is None and value is not None:
                    setattr(pose, field, value)
                    filled[field] += 1
        self.notes.append(
            f"ego_pose: filled from rosbags ({len(poses)} poses): "
            + ", ".join(f"{field} {count}" for field, count in filled.items())
        )
        self.notes.extend(self.bags.notes)

    def fill_vehicle_states(self) -> None:
        """Replace `vehicle_state` with one row per sample from the rosbags' vehicle status."""
        rows = self.bags.vehicle_states(self.sample_times)
        if not rows:
            return
        states = self.migrator.tables(SchemaName.VEHICLE_STATE)
        if states:
            self.notes.append(f"vehicle_state: {len(states)} 0.x rows replaced by rosbag rows")
        states[:] = [VehicleState.new(row) for row in rows]

    def fill_routes(self, lanelet_map: LaneletParser | None) -> None:
        """Add the routes valid during the dataset; drop those with lanelets not in its map."""
        first, last = self.sample_times[0], self.sample_times[-1]
        log, map_record = self.migrator.tables(SchemaName.LOG)[0], self._map_record()
        if map_record is None:
            self.notes.append("route: none added, the dataset has no map")
            return
        lanelets = None
        if lanelet_map is not None:
            lanelets = {
                int(id)
                for id, relation in lanelet_map.relations.items()
                if relation.tags.get("type") == "lanelet"
            }
        routes = self.migrator.tables(SchemaName.ROUTE)
        for row in self.bags.routes():
            if row["valid_from"] > last or (row["valid_until"] or last + 1) <= first:
                continue
            ids = {
                i
                for s in row["segments"]
                for i in (s["preferred_primitive_id"], *s["primitive_ids"])
            }
            missing = ids - lanelets if lanelets is not None else set()
            if missing:
                self.notes.append(
                    f"route {row['route_uuid']}: dropped, {len(missing)} of {len(ids)} lanelets "
                    "not in the dataset's map"
                )
                continue
            routes.append(Route.new(dict(row, log_token=log.token, map_token=map_record.token)))

    def manifest_fields(self) -> dict[str, Any]:
        """Return the manifest fields: config values, topics of each channel, Autoware version
        and the notes."""
        fields = {
            key: self.metadata[key]
            for key in ("project_id", "source_project_id", "is_discontinuous")
            if key in self.metadata
        }
        if "provenance" in self.metadata:
            provenance = self.metadata["provenance"]
            fields["provenance"] = {
                "rosbag": provenance["rosbag"],
                "topics": self.topics,
                "process_config_id": provenance.get("process_config_id"),
                "process_config_version_id": provenance.get("process_config_version_id"),
            }
        firmware = (self.bags.vehicle_info() or {}).get("firmware_release_name")
        # a release name has no spaces (e.g. "20250715-v4.2.1"); some vehicles report free
        # text such as "a mix of versions"
        if firmware is not None and re.fullmatch(r"\S+", firmware):
            fields["generators"] = [{"name": "autoware", "version": firmware}]
        elif firmware is not None:
            self.notes.append(f"generators: Autoware version not recorded ({firmware!r})")
        fields["migration_notes"] = self.notes
        return fields

    def _converter_params(self) -> dict[str, Any]:
        """Return the 0.x converter parameters from `status.json` ({} without one)."""
        path = osp.join(self.data_root, "status.json")
        if not osp.exists(path):
            return {}
        with open(path) as f:
            return json.load(f).get("rosbag2_to_non_annotated_t4_converter", {})

    def _channel_topics(self) -> dict[str, str]:
        """Return the topic of each channel, from the 0.x converter parameters."""
        params = self.converter_params
        sensors = [params.get("_lidar_sensor") or {}, *params.get("_camera_sensors", [])]
        return {sensor["channel"]: sensor["topic"] for sensor in sensors if "channel" in sensor}

    def _map_record(self) -> Any:
        maps = self.migrator.tables(SchemaName.MAP)
        return maps[0] if maps else None

    def _lanelet_map(self) -> LaneletParser | None:
        """Return the dataset's lanelet2 map, or None (routes are then not checked)."""
        map_record = self._map_record()
        if map_record is None or not osp.exists(osp.join(self.data_root, map_record.filename)):
            self.notes.append("route: no lanelet2 map, lanelet ids not checked")
            return None
        return LaneletParser(osp.join(self.data_root, map_record.filename))
