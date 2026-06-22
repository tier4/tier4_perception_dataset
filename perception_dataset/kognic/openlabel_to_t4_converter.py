"""Kognic OpenLABEL annotations -> T4 annotation tables.

Enriches a *non-annotated* T4 dataset with the annotations downloaded from
Kognic (see ``perception_dataset.kognic.download_annotation``). The annotation
type is auto-detected per scene; two kinds are supported:

3D cuboids (object detection)
    Populates the otherwise-empty annotation tables::

        instance.json  category.json  attribute.json
        visibility.json  sample_annotation.json

    This is the inverse of ``T4ToOpenLabelConverter``: that converter writes T4
    boxes out as Kognic cuboids (per-frame ego/base_link frame, yaw 0 facing +y,
    post-multiplied by Rz(-90 deg)); here we read those cuboids back and undo
    the transform to recover global-frame T4 boxes.

Point-cloud segmentation (``3DPointCloudSegmentation`` / ``semseg``)
    Writes T4 lidarseg: ``lidarseg.json`` plus one ``lidarseg/<version>/<token>.bin``
    of per-point ``uint8`` class indices per frame (one label per point in the
    matching ``LIDAR_CONCAT`` ``.pcd.bin``, in order), and adds the ontology
    classes to ``category.json`` keyed by their ontology id (index ``0`` =
    ``background``). Labels are decoded from Kognic run-length encoding
    (``#<count>V<class_id>``); a trailing run of unlabelled points omitted by the
    RLE is restored as ``background`` (0) and appended at the end.

OpenLABEL frames are matched to T4 samples by the LiDAR stream's URI timestamp
(authoritative when present; the frame ``external_id`` is a positional fallback
only when no timestamp is available).

The dataset is enriched **in place**: the populated tables (and any lidarseg
files) are written back into each scene's ``annotation/`` directory.

Layout::

    <output_base>/<scene>/                (T4 dataset, enriched in place)
        annotation/  data/  [lidarseg/]
    <annotation_base>/
        <scene>.json  or  <scene_uuid>.json   (downloaded OpenLABEL)

"""

import bisect
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from t4_devkit.schema.tables import (
    Attribute,
    Category,
    Instance,
    LidarSeg,
    SampleAnnotation,
    Visibility,
)

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.kognic.openlabel import (
    cuboid_val_to_t4_box,
    occlusion_to_visibility_level,
    to_t4_attribute_name,
)
from perception_dataset.t4_dataset.table_handler import TableHandler
from perception_dataset.utils.calculate_num_points import calculate_num_points
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.pointcloud import detect_point_stride
from perception_dataset.utils.t4_tables import (
    channel_by_calibrated_sensor,
    select_lidar_channel,
)

logger = configure_logger(modname=__name__)


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
                logger.warning(
                    f"No matching OpenLABEL annotation for scene {scene_dir.name}; skipping"
                )
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

        if _is_segmentation(openlabel):
            self._convert_segmentation(scene_dir, openlabel, sample_index, lidar_channel)
            return

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
                category_name = self._category_map.get(
                    obj.get("type", ""), obj.get("type", "unknown")
                )
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

        channel_by_calib = channel_by_calibrated_sensor(sensor, calibrated_sensor)
        lidar_channel = select_lidar_channel(sensor, channel_by_calib, sample_data)
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

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _cuboid_to_t4_box(
        self, val: List[float], ego_pose: dict
    ) -> Tuple[List[float], List[float], List[float]]:
        return cuboid_val_to_t4_box(val, ego_pose, self._iso_rotated_cuboids)

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
            name = to_t4_attribute_name(text["name"])
            tokens.append(self._attribute_token(tables, f"{name}.{text['val']}"))
        for boolean in object_data.get("boolean", []):
            name = to_t4_attribute_name(boolean["name"])
            value = "true" if boolean.get("val") else "false"
            tokens.append(self._attribute_token(tables, f"{name}.{value}"))
        for vec in object_data.get("vec", []):
            name = to_t4_attribute_name(vec["name"])
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
        level = occlusion_to_visibility_level(occlusion)
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
    # Point-cloud segmentation (3DPointCloudSegmentation -> T4 lidarseg)
    # ------------------------------------------------------------------

    def _convert_segmentation(
        self,
        scene_dir: Path,
        openlabel: dict,
        sample_index: "_SampleIndex",
        lidar_channel: str,
    ) -> None:
        """Convert OpenLABEL point-cloud segmentation into T4 lidarseg tables.

        Writes ``lidarseg.json`` plus one ``<token>.bin`` of per-point uint8
        class indices per frame under ``<scene>/lidarseg/<version>/``, and adds
        the ontology classes (with their ``index``) to ``category.json``. The
        layout mirrors ``annotation_files_generator._convert_lidarseg_scene_annotations``.
        """
        frames = openlabel.get("frames", {})

        # Ontology id -> class name; the id doubles as the T4 category index and
        # the per-point label value stored in the .bin file.
        ontology = _segmentation_ontology(openlabel)
        if not ontology:
            logger.warning(f"No segmentation ontology found in annotation; skipping {scene_dir}")
            return

        # Lidar sample_data record keyed by the sample it belongs to.
        lidar_sd_by_sample: Dict[str, dict] = {
            sd["sample_token"]: sd
            for sd in self._load_table(scene_dir, "sample_data.json")
            if lidar_channel in sd["filename"]
        }

        category_table = TableHandler(Category)
        # Reserve index 0 for points the annotator left unlabelled.
        category_table.insert_into_table(
            name="background", description="unlabelled / background points", index=0
        )
        for index in sorted(ontology):
            category_table.insert_into_table(name=ontology[index], description="", index=index)

        lidarseg_table = TableHandler(LidarSeg)
        anno_dir = scene_dir / "annotation"
        version_name = anno_dir.name
        lidarseg_relative = Path("lidarseg") / version_name
        lidarseg_dir = scene_dir / lidarseg_relative
        # Clear stale .bin files: each run mints fresh tokens, so re-running
        # would otherwise accumulate orphaned files not referenced by lidarseg.json.
        shutil.rmtree(lidarseg_dir, ignore_errors=True)
        lidarseg_dir.mkdir(parents=True, exist_ok=True)

        placed = 0
        skipped = 0
        for frame_key, frame in sorted(frames.items(), key=lambda kv: int(kv[0])):
            rle = _frame_segmentation_rle(frame)
            if rle is None:
                continue

            match = sample_index.match(frame, frame_key, lidar_channel)
            if match is None:
                logger.warning(
                    f"OpenLABEL frame {frame_key} could not be matched to a T4 sample; "
                    f"dropping its segmentation"
                )
                skipped += 1
                continue
            sample_token = match[0]

            sample_data = lidar_sd_by_sample.get(sample_token)
            if sample_data is None:
                logger.warning(
                    f"No {lidar_channel} sample_data for the sample matched by frame "
                    f"{frame_key}; skipping its segmentation"
                )
                skipped += 1
                continue

            labels = _decode_rle_labels(rle)
            num_points = _lidar_point_count(scene_dir / sample_data["filename"])
            if num_points is None:
                logger.warning(
                    f"Could not read {sample_data['filename']} for frame {frame_key}; "
                    f"skipping its segmentation"
                )
                skipped += 1
                continue
            if labels.shape[0] > num_points:
                # More labels than points means the annotated cloud is not this
                # extraction at all (a genuine data mismatch); aligning is unsafe.
                logger.warning(
                    f"Segmentation has more labels than points for frame {frame_key} "
                    f"({sample_data['filename']}): {labels.shape[0]} labels vs "
                    f"{num_points} points. The annotated cloud differs from this T4 "
                    f"extraction; skipping this frame."
                )
                skipped += 1
                continue
            if labels.shape[0] < num_points:
                # Kognic RLE encodes labels sequentially from point 0 and omits a
                # trailing run of unlabelled points; restore them as background (0).
                # NOTE: missing points are treated as 0 and added at the end.
                pad = num_points - labels.shape[0]
                logger.warning(
                    f"Frame {frame_key}: RLE covers {labels.shape[0]}/{num_points} points; "
                    f"padding {pad} trailing point(s) as background (class 0)."
                )
                labels = np.concatenate([labels, np.zeros(pad, dtype=np.uint8)])

            token = lidarseg_table.insert_into_table(
                filename="", sample_data_token=sample_data["token"]
            )
            labels.tofile(lidarseg_dir / f"{token}.bin")
            lidarseg_table.update_record_from_token(
                token, filename=str(lidarseg_relative / f"{token}.bin")
            )
            placed += 1

        category_table.save_json(str(anno_dir))
        lidarseg_table.save_json(str(anno_dir))

        logger.info(
            f"[DONE]  {scene_dir}: {placed} lidarseg frame(s), "
            f"{len(ontology)} categor(y/ies) (skipped {skipped})"
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

    # Max |Δ| (µs) between an OpenLABEL capture time and a T4 sample timestamp
    # still treated as the same frame. T4 sample timestamps are produced via a
    # lossy float64 path (``int((sec + nanosec * 1e-9) * 1e6)``), so the µs value
    # can differ from a direct ns->µs conversion by ~1; 1 ms is far below the
    # ~100 ms frame period yet absorbs that rounding error.
    _MATCH_TOLERANCE_US = 1000

    def __post_init__(self):
        self._sorted_us: List[int] = sorted(self.by_timestamp_us)

    def match(self, frame: dict, frame_key: str, lidar_channel: str) -> Optional[Tuple[str, dict]]:
        # The lidar uri timestamp is the ground truth: when present it is
        # authoritative, so a frame whose capture time has no nearby sample is
        # genuinely unmatched (e.g. annotation and point clouds from different
        # recordings). Only fall back to the positional external_id when no
        # usable timestamp is available, since that mapping is unreliable.
        ts_ns = self._uri_timestamp_ns(frame, lidar_channel)
        if ts_ns is not None:
            candidate = self._nearest(round(ts_ns / 1000))
        else:
            candidate = self._by_external_id(frame)
        if candidate is None or candidate[1] is None:
            return None
        return candidate  # type: ignore[return-value]

    def _uri_timestamp_ns(self, frame: dict, lidar_channel: str) -> Optional[int]:
        streams = frame.get("frame_properties", {}).get("streams", {})
        stream = self._select_lidar_stream(streams, lidar_channel)
        uri = stream.get("uri") if stream else None
        return _parse_uri_timestamp_ns(uri) if uri else None

    @staticmethod
    def _select_lidar_stream(streams: dict, lidar_channel: str) -> Optional[dict]:
        """Find the lidar stream entry in a frame's ``streams`` mapping.

        OpenLABEL exports key the lidar stream as ``"lidar"``; other paths may
        use the T4 channel name (e.g. ``LIDAR_CONCAT``). Try both, then fall
        back to any lidar-like key.
        """
        for key in (lidar_channel, "lidar"):
            if key in streams:
                return streams[key]
        for key, value in streams.items():
            if "lidar" in key.lower():
                return value
        return None

    def _nearest(self, ts_us: int) -> Optional[Tuple[str, Optional[dict]]]:
        """Return the sample whose timestamp is closest to ``ts_us`` within tolerance."""
        if not self._sorted_us:
            return None
        i = bisect.bisect_left(self._sorted_us, ts_us)
        best: Optional[int] = None
        for j in (i - 1, i):
            if 0 <= j < len(self._sorted_us):
                cand = self._sorted_us[j]
                if best is None or abs(cand - ts_us) < abs(best - ts_us):
                    best = cand
        if best is None or abs(best - ts_us) > self._MATCH_TOLERANCE_US:
            return None
        return self.by_timestamp_us[best]

    def _by_external_id(self, frame: dict) -> Optional[Tuple[str, Optional[dict]]]:
        external_id = frame.get("frame_properties", {}).get("external_id")
        try:
            idx = int(external_id)
        except (TypeError, ValueError):
            return None
        return self.by_order[idx] if 0 <= idx < len(self.by_order) else None


def _parse_uri_timestamp_ns(uri: str) -> Optional[int]:
    """Extract the capture-time nanoseconds from a stream uri.

    Camera uris are ``<ns>.<ext>``; lidar uris carry a frame-index prefix,
    e.g. ``550_<ns>.csv``. Take the last underscore-separated numeric token.
    """
    token = Path(uri).stem.rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return None


# ``#<run_length>V<class_id>`` repeated; the Kognic RLE encoding of per-point labels.
_RLE_TOKEN = re.compile(r"#(\d+)V(\d+)")


def _is_segmentation(openlabel: dict) -> bool:
    """True if this OpenLABEL carries point-cloud segmentation labels."""
    if openlabel.get("metadata", {}).get("annotation_type") == "semseg":
        return True
    return any(
        _frame_segmentation_rle(frame) is not None
        for frame in openlabel.get("frames", {}).values()
    )


def _segmentation_ontology(openlabel: dict) -> Dict[int, str]:
    """Map ontology class id -> class name (used as T4 category index -> name)."""
    ontology: Dict[int, str] = {}
    for entry in openlabel.get("ontologies", {}).values():
        for class_id, name in entry.get("classifications", {}).items():
            try:
                ontology[int(class_id)] = name
            except (TypeError, ValueError):
                continue
    return ontology


def _frame_segmentation_rle(frame: dict) -> Optional[str]:
    """Return the RLE-encoded lidar label string for a frame, if present."""
    for frame_object in frame.get("objects", {}).values():
        for binary in frame_object.get("object_data", {}).get("binary", []):
            if binary.get("name") == "labels" and binary.get("encoding") == "rle":
                return binary.get("val")
    return None


def _decode_rle_labels(val: str) -> np.ndarray:
    """Expand a ``#<count>V<class>`` RLE string into per-point uint8 labels."""
    pairs = _RLE_TOKEN.findall(val)
    counts = np.fromiter((int(c) for c, _ in pairs), dtype=np.int64, count=len(pairs))
    classes = np.fromiter((int(v) for _, v in pairs), dtype=np.int64, count=len(pairs))
    if classes.size and classes.max() > np.iinfo(np.uint8).max:
        raise ValueError(f"Segmentation class id {classes.max()} does not fit in uint8")
    return np.repeat(classes, counts).astype(np.uint8)


def _lidar_point_count(bin_path: Path) -> Optional[int]:
    """Number of points in a LIDAR_CONCAT ``.pcd.bin``, or None if unreadable."""
    if not bin_path.exists():
        return None
    floats = np.fromfile(bin_path, dtype=np.float32)
    if floats.size == 0:
        return 0
    return floats.size // detect_point_stride(floats, bin_path)
