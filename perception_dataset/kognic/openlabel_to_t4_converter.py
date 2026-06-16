"""Kognic OpenLABEL annotations -> T4 annotation tables.

Enriches a *non-annotated* T4 dataset with the 3D box annotations downloaded
from Kognic (see ``perception_dataset.kognic.download_annotation``). It
populates the otherwise-empty annotation tables:

    instance.json  category.json  attribute.json
    visibility.json  sample_annotation.json

This is the inverse of ``T4ToOpenLabelConverter``: that converter writes T4
boxes out as Kognic cuboids (per-frame ego/base_link frame, yaw 0 facing +y,
post-multiplied by Rz(-90 deg)); here we read those cuboids back and undo the
transform to recover global-frame T4 boxes.

The dataset is enriched **in place**: the populated annotation tables are
written back into each scene's ``annotation/`` directory.

Layout::

    <output_base>/<scene>/                (T4 dataset, enriched in place)
        annotation/  data/
    <annotation_base>/
        <scene>.json  or  <scene_uuid>.json   (downloaded OpenLABEL)

"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from t4_devkit.schema.tables import (
    Attribute,
    Category,
    Instance,
    SampleAnnotation,
    Visibility,
)

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import LIDAR_CONCAT_CHANNEL
from perception_dataset.t4_dataset.table_handler import TableHandler
from perception_dataset.utils.calculate_num_points import calculate_num_points
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

# Kognic cuboids face +y at yaw 0 while T4/nuScenes boxes face +x. This is the
# same correction applied (forward) by T4ToOpenLabelConverter; here we invert it.
ROTATION_T4_TO_KOGNIC = Rotation.from_euler("z", -90, degrees=True)

# TODO: the visbility mapping is a best effort based on the occlusion_state values in T4 dataset and kognic format.
# It should be revisited and standarized in the future. 
_OCCLUSION_TO_VISIBILITY = {
    "none": "full",
    "light": "most",
    "most": "partial",
    "full": "none",
}

# TODO: Kognic property names that T4 stores under a different attribute name. The
# value (e.g. ``with_rider``/``without_rider``) is preserved as-is. Inverse of
# ``_T4_ATTRIBUTE_NAME_TO_KOGNIC`` in ``t4_to_openlabel.py``.
# It should be revisited and standarized in the future. 
_KOGNIC_ATTRIBUTE_NAME_TO_T4 = {
    "rider_state": "two_wheel_vehicle_state",
}


class OpenLabelToT4Converter(AbstractConverter[None]):
    """Merge downloaded Kognic OpenLABEL annotations into a non-annotated T4 dataset."""

    def __init__(
        self,
        output_base: str,
        annotation_base: str,
        iso_rotated_cuboids: bool = False,
        category_map: Optional[Dict[str, str]] = None,
        include_attributes: bool = True,
    ):
        super().__init__(output_base, output_base)
        self._annotation_base = Path(annotation_base)
        self._iso_rotated_cuboids = iso_rotated_cuboids
        self._category_map = category_map or {}
        self._include_attributes = include_attributes

    # ------------------------------------------------------------------
    # AbstractConverter contract
    # ------------------------------------------------------------------

    def convert(self) -> None:
        start = time.time()
        output_base = Path(self._output_base)

        openlabels = self._index_openlabels()
        if not openlabels:
            logger.warning(f"No OpenLABEL annotation files found under {self._annotation_base}")
            return

        scenes = self._find_t4_scenes(output_base)
        if not scenes:
            logger.warning(f"No T4 scenes found under {output_base}")
            return

        for scene_dir in scenes:
            openlabel_path = self._match_openlabel(scene_dir, openlabels)
            if openlabel_path is None:
                logger.warning(f"No matching OpenLABEL annotation for scene {scene_dir.name}; skipping")
                continue

            self._convert_one_scene(scene_dir, openlabel_path)

        logger.info(f"Elapsed: {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Discovery / matching
    # ------------------------------------------------------------------

    @staticmethod
    def _is_t4_scene(path: Path) -> bool:
        return (path / "annotation" / "sample.json").exists()

    def _find_t4_scenes(self, dataset_base: Path) -> List[Path]:
        if self._is_t4_scene(dataset_base):
            return [dataset_base]
        # A scene is any directory (at any depth) holding annotation/sample.json.
        scenes = {
            sample_json.parent.parent
            for sample_json in dataset_base.rglob("annotation/sample.json")
        }
        return sorted(scenes)

    def _index_openlabels(self) -> Dict[str, Path]:
        """Map every plausible scene identifier to its OpenLABEL file path."""
        index: Dict[str, Path] = {}
        if not self._annotation_base.exists():
            return index

        files = (
            [self._annotation_base]
            if self._annotation_base.is_file()
            else sorted(self._annotation_base.rglob("*.json"))
        )
        for path in files:
            index.setdefault(path.stem, path)
            metadata = self._read_metadata(path)
            for key in ("dataset_id", "source_filename", "scene_uuid", "input_external_id"):
                value = metadata.get(key)
                if value:
                    index.setdefault(str(value), path)
            scene_metadata = metadata.get("scene_metadata") or {}
            for key in ("dataset_id", "source_filename", "inner_uuid"):
                value = scene_metadata.get(key)
                if value:
                    index.setdefault(str(value), path)
        return index

    @staticmethod
    def _read_metadata(path: Path) -> dict:
        try:
            with open(path) as f:
                return json.load(f).get("openlabel", {}).get("metadata", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def _match_openlabel(self, scene_dir: Path, openlabels: Dict[str, Path]) -> Optional[Path]:
        """Match a scene to its OpenLABEL file by the scene dir name or any of
        its ancestor dir names up to the dataset root.

        T4 datasets are commonly nested as ``<root>/<scene_id>/<version>/``,
        so the matching identifier is often an ancestor (e.g. ``<scene_id>``)
        rather than the leaf (e.g. version ``0``).
        """
        root = Path(self._output_base).resolve()
        current = scene_dir.resolve()
        while True:
            if current.name in openlabels:
                return openlabels[current.name]
            if current == root or current.parent == current:
                return None
            current = current.parent

    # ------------------------------------------------------------------
    # Scene conversion
    # ------------------------------------------------------------------

    def _convert_one_scene(self, scene_dir: Path, openlabel_path: Path) -> None:
        logger.info(f"[BEGIN] {scene_dir} + {openlabel_path.name}")

        with open(openlabel_path) as f:
            openlabel = json.load(f)["openlabel"]

        sample_index, lidar_channel = self._build_sample_index(scene_dir)
        if not sample_index.by_timestamp_us:
            logger.warning(f"No samples found in {scene_dir}; skipping")
            return

        objects = openlabel.get("objects", {})
        frames = openlabel.get("frames", {})

        tables = self._init_annotation_tables()
        # object_uuid -> instance token; reused across frames.
        instance_tokens: Dict[str, str] = {}
        # instance token -> ordered list of (frame_idx, sample_annotation token)
        instance_annotations: Dict[str, List[Tuple[int, str]]] = {}

        placed = 0
        skipped = 0
        for frame_key, frame in sorted(frames.items(), key=lambda kv: int(kv[0])):
            match = sample_index.match(frame, frame_key, lidar_channel)
            if match is None:
                logger.warning(
                    f"OpenLABEL frame {frame_key} could not be matched to a T4 sample; "
                    f"dropping {len(frame.get('objects', {}))} object(s)"
                )
                skipped += len(frame.get("objects", {}))
                continue
            sample_token, ego_pose = match

            for object_uuid, frame_object in frame.get("objects", {}).items():
                object_data = frame_object.get("object_data", {})
                cuboids = object_data.get("cuboid", [])
                if not cuboids:
                    continue

                obj = objects.get(object_uuid, {})
                category_name = self._category_map.get(obj.get("type", ""), obj.get("type", "unknown"))
                instance_token = self._get_or_create_instance(
                    tables, instance_tokens, object_uuid, category_name
                )

                attribute_tokens = self._collect_attribute_tokens(tables, object_data)
                visibility_token = self._visibility_token(tables, object_data)
                translation, size, rotation = self._cuboid_to_t4_box(cuboids[0]["val"], ego_pose)

                annotation_token = tables["sample_annotation"].insert_into_table(
                    sample_token=sample_token,
                    instance_token=instance_token,
                    attribute_tokens=attribute_tokens,
                    visibility_token=visibility_token,
                    translation=translation,
                    size=size,
                    rotation=rotation,
                    num_lidar_pts=0,  # filled in by _populate_num_lidar_pts after _save
                    num_radar_pts=0,
                    next="",
                    prev="",
                )
                instance_annotations.setdefault(instance_token, []).append(
                    (int(frame_key), annotation_token)
                )
                placed += 1

        self._finalize_instances(tables, instance_annotations)
        self._save(scene_dir, tables)
        self._populate_num_lidar_pts(scene_dir, tables["sample_annotation"], lidar_channel)

        logger.info(
            f"[DONE]  {scene_dir}: {placed} annotation(s), {len(instance_tokens)} instance(s) "
            f"(skipped {skipped})"
        )

    # ------------------------------------------------------------------
    # Frame <-> sample mapping
    # ------------------------------------------------------------------

    def _build_sample_index(self, scene_dir: Path) -> Tuple["_SampleIndex", str]:
        """Index T4 samples by lidar timestamp so OpenLABEL frames can be matched.

        OpenLABEL frames carry their lidar stream ``uri`` (the absolute-ns
        capture timestamp) and an ``external_id`` (original scene frame index).
        Annotation requests often cover only a subsampled set of scene frames,
        so a positional ``frame k -> sample k`` mapping is unsafe; matching by
        timestamp (with external-id as a fallback) is exact.
        """
        sample = self._load_table(scene_dir, "sample.json")
        sample_data = self._load_table(scene_dir, "sample_data.json")
        sensor = self._load_table(scene_dir, "sensor.json")
        calibrated_sensor = self._load_table(scene_dir, "calibrated_sensor.json")
        ego_pose = self._load_table(scene_dir, "ego_pose.json")

        lidar_channel = self._lidar_channel(sensor, calibrated_sensor, sample_data)
        token_to_channel = {s["token"]: s["channel"] for s in sensor}
        channel_by_calib = {
            c["token"]: token_to_channel.get(c["sensor_token"]) for c in calibrated_sensor
        }
        ego_pose_by_token = {ep["token"]: ep for ep in ego_pose}

        ego_pose_by_sample: Dict[str, dict] = {}
        for record in sample_data:
            if channel_by_calib.get(record["calibrated_sensor_token"]) == lidar_channel:
                ego_pose_by_sample[record["sample_token"]] = ego_pose_by_token.get(
                    record["ego_pose_token"]
                )

        ordered = sorted(sample, key=lambda s: s["timestamp"])
        by_timestamp_us = {
            s["timestamp"]: (s["token"], ego_pose_by_sample.get(s["token"])) for s in ordered
        }
        by_order = [(s["token"], ego_pose_by_sample.get(s["token"])) for s in ordered]
        return _SampleIndex(by_timestamp_us, by_order), lidar_channel

    @staticmethod
    def _lidar_channel(sensor: list, calibrated_sensor: list, sample_data: list) -> str:
        """Pick the lidar channel that carries ego poses (LIDAR_CONCAT if present)."""
        token_to_channel = {s["token"]: s["channel"] for s in sensor}
        channel_by_calib = {
            c["token"]: token_to_channel.get(c["sensor_token"]) for c in calibrated_sensor
        }
        channels = {
            channel_by_calib.get(r["calibrated_sensor_token"]) for r in sample_data
        }
        if LIDAR_CONCAT_CHANNEL in channels:
            return LIDAR_CONCAT_CHANNEL
        lidar_channels = sorted(
            s["channel"] for s in sensor if s.get("modality") == "lidar"
        )
        return lidar_channels[0] if lidar_channels else LIDAR_CONCAT_CHANNEL

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _cuboid_to_t4_box(
        self, val: List[float], ego_pose: dict
    ) -> Tuple[List[float], List[float], List[float]]:
        """Undo ``T4ToOpenLabelConverter._cuboid_val``.

        ``val`` is ``[x, y, z, qx, qy, qz, qw, sx, sy, sz]`` in the per-frame
        ego/base_link frame. Returns ``(translation, size, rotation)`` in the
        T4 global frame, with rotation as a wxyz quaternion and size as
        ``[width, length, height]``.
        """
        rotation_ego = Rotation.from_quat(_quat_wxyz_to_xyzw(ego_pose["rotation"]))

        position_ego = np.asarray(val[0:3], dtype=np.float64)
        translation = rotation_ego.apply(position_ego) + np.asarray(
            ego_pose["translation"], dtype=np.float64
        )

        rotation_cuboid = Rotation.from_quat([val[3], val[4], val[5], val[6]])
        if self._iso_rotated_cuboids:
            # ISO8855 cuboids already face +x (T4 convention); no yaw correction.
            rotation_box = rotation_ego * rotation_cuboid
        else:
            rotation_box = rotation_ego * rotation_cuboid * ROTATION_T4_TO_KOGNIC.inv()
        qx, qy, qz, qw = rotation_box.as_quat()

        width, length, height = (float(v) for v in val[7:10])
        return (
            [float(translation[0]), float(translation[1]), float(translation[2])],
            [width, length, height],
            [float(qw), float(qx), float(qy), float(qz)],
        )

    # ------------------------------------------------------------------
    # Table building
    # ------------------------------------------------------------------

    @staticmethod
    def _init_annotation_tables() -> Dict[str, TableHandler]:
        return {
            "category": TableHandler(Category),
            "instance": TableHandler(Instance),
            "attribute": TableHandler(Attribute),
            "visibility": TableHandler(Visibility),
            "sample_annotation": TableHandler(SampleAnnotation),
        }

    def _get_or_create_instance(
        self,
        tables: Dict[str, TableHandler],
        instance_tokens: Dict[str, str],
        object_uuid: str,
        category_name: str,
    ) -> str:
        if object_uuid in instance_tokens:
            return instance_tokens[object_uuid]

        # Reuse the existing category for this name, else create one with the
        # next sequential index. The index doubles as the rerun ClassId during
        # visualization, which must be a non-negative uint16 (-1 overflows).
        category_token = tables["category"].get_token_from_field(
            field_name="name", field_value=category_name
        )
        if not category_token:
            category_token = tables["category"].insert_into_table(
                name=category_name,
                description="",
                index=len(tables["category"].to_records()),
                has_orientation=True,
                has_number=False,
            )
        instance_token = tables["instance"].insert_into_table(
            category_token=category_token,
            instance_name=f"{category_name}:{object_uuid}",
            nbr_annotations=0,
            first_annotation_token="",
            last_annotation_token="",
        )
        instance_tokens[object_uuid] = instance_token
        return instance_token

    def _collect_attribute_tokens(
        self, tables: Dict[str, TableHandler], object_data: dict
    ) -> List[str]:
        if not self._include_attributes:
            return []

        tokens: List[str] = []
        for text in object_data.get("text", []):
            name = _KOGNIC_ATTRIBUTE_NAME_TO_T4.get(text["name"], text["name"])
            tokens.append(self._attribute_token(tables, f"{name}.{text['val']}"))
        for boolean in object_data.get("boolean", []):
            name = _KOGNIC_ATTRIBUTE_NAME_TO_T4.get(boolean["name"], boolean["name"])
            value = "true" if boolean.get("val") else "false"
            tokens.append(self._attribute_token(tables, f"{name}.{value}"))
        for vec in object_data.get("vec", []):
            name = _KOGNIC_ATTRIBUTE_NAME_TO_T4.get(vec["name"], vec["name"])
            for value in vec.get("val", []):
                tokens.append(self._attribute_token(tables, f"{name}.{value}"))
        return tokens

    @staticmethod
    def _attribute_token(tables: Dict[str, TableHandler], name: str) -> str:
        return tables["attribute"].insert_into_table(
            reuse_if_duplicate=True, name=name, description=""
        )

    def _visibility_token(self, tables: Dict[str, TableHandler], object_data: dict) -> str:
        occlusion = next(
            (t["val"] for t in object_data.get("text", []) if t["name"] == "occlusion_state"),
            None,
        )
        level = _OCCLUSION_TO_VISIBILITY.get(occlusion, "unavailable") if occlusion else "unavailable"
        return tables["visibility"].insert_into_table(
            reuse_if_duplicate=True, level=level, description=""
        )

    @staticmethod
    def _finalize_instances(
        tables: Dict[str, TableHandler],
        instance_annotations: Dict[str, List[Tuple[int, str]]],
    ) -> None:
        sample_annotation = tables["sample_annotation"]
        for instance_token, annotations in instance_annotations.items():
            ordered = [token for _, token in sorted(annotations, key=lambda fa: fa[0])]
            for i in range(1, len(ordered)):
                sample_annotation.update_record_from_token(ordered[i - 1], next=ordered[i])
                sample_annotation.update_record_from_token(ordered[i], prev=ordered[i - 1])
            tables["instance"].update_record_from_token(
                instance_token,
                nbr_annotations=len(ordered),
                first_annotation_token=ordered[0],
                last_annotation_token=ordered[-1],
            )

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    @staticmethod
    def _load_table(scene_dir: Path, name: str) -> list:
        path = scene_dir / "annotation" / name
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _populate_num_lidar_pts(
        scene_dir: Path, sample_annotation: TableHandler, lidar_channel: str
    ) -> None:
        """Count lidar points inside each box and write them back in place.

        Reuses ``calculate_num_points``, which reloads the dataset from disk via
        ``Tier4``; this therefore runs *after* ``_save``. If the scene's lidar
        point clouds are unavailable the counts are left at their default of 0
        rather than failing the whole conversion.
        """
        try:
            calculate_num_points(str(scene_dir), lidar_channel, sample_annotation)
        except Exception as e:  # noqa: BLE001 - lidar data may not be present
            logger.warning(
                f"Could not compute num_lidar_pts for {scene_dir} "
                f"(channel {lidar_channel}); leaving counts at 0: {e}"
            )
            return
        sample_annotation.save_json(str(scene_dir / "annotation"))

    @staticmethod
    def _save(scene_dir: Path, tables: Dict[str, TableHandler]) -> None:
        """Write the populated annotation tables back into the scene in place."""
        anno_dir = scene_dir / "annotation"
        anno_dir.mkdir(parents=True, exist_ok=True)
        for table in tables.values():
            table.save_json(str(anno_dir))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@dataclass
class _SampleIndex:
    """Resolve an OpenLABEL frame to its T4 (sample_token, ego_pose)."""

    by_timestamp_us: Dict[int, Tuple[str, Optional[dict]]]
    by_order: List[Tuple[str, Optional[dict]]]

    def match(
        self, frame: dict, frame_key: str, lidar_channel: str
    ) -> Optional[Tuple[str, dict]]:
        candidate = self._by_uri_timestamp(frame, lidar_channel) or self._by_external_id(frame)
        if candidate is None or candidate[1] is None:
            return None
        return candidate  # type: ignore[return-value]

    def _by_uri_timestamp(
        self, frame: dict, lidar_channel: str
    ) -> Optional[Tuple[str, Optional[dict]]]:
        stream = frame.get("frame_properties", {}).get("streams", {}).get(lidar_channel, {})
        uri = stream.get("uri")
        if not uri:
            return None
        try:
            ts_us = int(Path(uri).stem) // 1000
        except ValueError:
            return None
        return self.by_timestamp_us.get(ts_us)

    def _by_external_id(self, frame: dict) -> Optional[Tuple[str, Optional[dict]]]:
        external_id = frame.get("frame_properties", {}).get("external_id")
        try:
            idx = int(external_id)
        except (TypeError, ValueError):
            return None
        return self.by_order[idx] if 0 <= idx < len(self.by_order) else None


def _quat_wxyz_to_xyzw(quat: list) -> List[float]:
    return [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]
