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

Like ``DeepenToT4Converter``, each scene is first copied from the
non-annotated dataset (``input_base``) into ``output_base`` and the
annotations are written into the copy. When ``input_bag_base`` is given, a
time/topic filtered copy of each scene's rosbag is placed next to the
annotations as ``input_bag`` (same as the Deepen flow).

Layout::

    <input_base>/<scene>/                 (non-annotated T4 dataset)
        annotation/  data/
    <input_bag_base>/<scene>/             (optional source rosbag)
    <output_base>/<scene>/                (annotated T4 dataset)
        annotation/  data/  [lidarseg/]  [input_bag/]
    <annotation_base>/
        <scene>.json  or  <scene_uuid>.json   (downloaded OpenLABEL)

"""

import bisect
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple, Union

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
import perception_dataset.utils.misc as misc_utils
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
        input_base: str,
        output_base: str,
        annotation_base: str,
        input_bag_base: Optional[str] = None,
        topic_list: Union[Dict[str, List[str]], List[str], None] = None,
        overwrite_mode: bool = False,
        iso_rotated_cuboids: bool = False,
        category_map: Optional[Dict[str, str]] = None,
        include_attributes: bool = True,
    ):
        """Initialize the converter.

        Args:
            input_base (str): Base directory containing non-annotated T4 scenes.
            output_base (str): Destination directory for annotated T4 scenes.
            annotation_base (str): OpenLABEL file or directory to import.
            input_bag_base (Optional[str]): Optional source rosbag directory.
            topic_list (Union[Dict[str, List[str]], List[str], None]): Rosbag
                topics to preserve.
            overwrite_mode (bool): Whether existing output scenes may be replaced.
            iso_rotated_cuboids (bool): Whether cuboids use the T4 forward axis.
            category_map (Optional[Dict[str, str]]): Kognic-to-T4 category map.
            include_attributes (bool): Whether to import object attributes.
        """
        super().__init__(input_base, output_base)
        self._annotation_base = Path(annotation_base)
        self._input_bag_base: Optional[str] = input_bag_base
        self._topic_list: Union[Dict[str, List[str]], List[str]] = topic_list or []
        self._overwrite_mode = overwrite_mode
        self._iso_rotated_cuboids = iso_rotated_cuboids
        self._category_map = category_map or {}
        self._include_attributes = include_attributes

    # ------------------------------------------------------------------
    # AbstractConverter contract
    # ------------------------------------------------------------------

    def convert(self) -> None:
        """Convert all matched OpenLABEL and T4 scenes.

        Returns:
            None
        """
        start = time.time()
        input_base = Path(self._input_base)

        openlabels = self._index_openlabels()
        if not openlabels:
            logger.warning(f"No OpenLABEL annotation files found under {self._annotation_base}")
            return

        scenes = self._find_t4_scenes(input_base)
        if not scenes:
            logger.warning(f"No T4 scenes found under {input_base}")
            return

        for scene_dir in scenes:
            openlabel_path = self._match_openlabel(scene_dir, openlabels)
            if openlabel_path is None:
                logger.warning(
                    f"No matching OpenLABEL annotation for scene {scene_dir.name}; skipping"
                )
                continue

            output_dir = self._prepare_output_scene(scene_dir)
            self._convert_one_scene(output_dir, openlabel_path)

        logger.info(f"Elapsed: {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Output preparation (copy data / make filtered rosbag, Deepen-style)
    # ------------------------------------------------------------------

    def _prepare_output_scene(self, scene_dir: Path) -> Path:
        """Copy the non-annotated scene (and optionally its rosbag) into the output.

        Mirrors ``DeepenToT4Converter.convert``: the scene keeps its path
        relative to ``input_base`` under ``output_base``.

        Args:
            scene_dir (Path): Source T4 scene directory.

        Returns:
            Path: Prepared output scene directory.

        Raises:
            ValueError: If input and output resolve to the same scene, or an
                output exists while overwrite mode is disabled.
        """
        input_base = Path(self._input_base).resolve()
        scene_dir = scene_dir.resolve()
        relative = scene_dir.relative_to(input_base)
        output_dir = Path(self._output_base).resolve() / relative
        if output_dir == scene_dir:
            raise ValueError(
                f"input_base and output_base resolve to the same scene ({scene_dir}); "
                f"they must differ"
            )

        if output_dir.exists():
            logger.warning(f"{output_dir} already exists.")
            if not self._overwrite_mode:
                raise ValueError("If you want to overwrite files, use --overwrite option.")
        shutil.rmtree(output_dir, ignore_errors=True)
        self._copy_data(scene_dir, output_dir)

        if self._input_bag_base is not None:
            scene_name = relative.parts[0] if relative.parts else scene_dir.name
            input_bag_dir = Path(self._input_bag_base) / scene_name
            self._make_rosbag(scene_dir, input_bag_dir, output_dir)
        return output_dir

    @staticmethod
    def _copy_data(input_dir: Path, output_dir: Path) -> None:
        """Copy T4 data files into an output scene.

        Args:
            input_dir (Path): Source scene directory.
            output_dir (Path): Destination scene directory.

        Returns:
            None
        """
        logger.info(f"Copying {input_dir} to {output_dir} ... ")
        for item in os.listdir(input_dir):
            if item not in ["annotation", "data", "status.json"]:
                # Skip non t4-format files
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            src_path = input_dir / item
            dest_path = output_dir / item
            if src_path.is_dir():
                logger.info(f"Copying directory {src_path} to {dest_path} ...")
                shutil.copytree(src_path, dest_path)
            else:
                logger.info(f"Copying file {src_path} to {dest_path} ...")
                shutil.copy2(src_path, dest_path)
            if item == "data" and (input_dir / "anonymized_data").exists():
                # Overwrite data with anonymized_data if exists
                shutil.copytree(input_dir / "anonymized_data", dest_path, dirs_exist_ok=True)
        logger.info("Done!")

    @staticmethod
    def _find_start_end_time(t4_dataset_dir: Path) -> Tuple[float, float]:
        """Get the scene time range for rosbag filtering.

        Args:
            t4_dataset_dir (Path): T4 scene directory.

        Returns:
            Tuple[float, float]: Unix start and end times with two-second margins.
        """
        from t4_devkit import Tier4

        t4_dataset = Tier4(data_root=str(t4_dataset_dir), verbose=False)
        timestamps = [sample.timestamp for sample in t4_dataset.sample]
        start_sec = misc_utils.nusc_timestamp_to_unix_timestamp(min(timestamps)) - 2.0
        end_sec = misc_utils.nusc_timestamp_to_unix_timestamp(max(timestamps)) + 2.0
        return start_sec, end_sec

    def _make_rosbag(self, scene_dir: Path, input_bag_dir: Path, output_dir: Path) -> None:
        """Copy a filtered rosbag into an output scene.

        Args:
            scene_dir (Path): Source T4 scene used to derive the time range.
            input_bag_dir (Path): Source rosbag directory.
            output_dir (Path): Destination scene directory.

        Returns:
            None
        """
        # Imported lazily: pulls in ROS 2 dependencies not needed otherwise.
        from perception_dataset.rosbag2.rosbag2_converter import Rosbag2Converter

        if not input_bag_dir.exists():
            logger.warning(f"Input rosbag {input_bag_dir} not found; skipping input_bag")
            return
        output_bag_dir = output_dir / "input_bag"

        logger.info(f"Copying {input_bag_dir} to {output_bag_dir} ... ")
        start_sec, end_sec = self._find_start_end_time(scene_dir)
        output_bag_dir_temp = output_dir / input_bag_dir.name
        converter = Rosbag2Converter(
            str(input_bag_dir),
            str(output_bag_dir_temp),
            self._topic_list,
            start_sec,
            end_sec,
        )
        converter.convert()
        if output_bag_dir_temp != output_bag_dir:
            shutil.move(str(output_bag_dir_temp), str(output_bag_dir))

    # ------------------------------------------------------------------
    # Discovery / matching
    # ------------------------------------------------------------------

    @staticmethod
    def _is_t4_scene(path: Path) -> bool:
        """Check whether a directory contains a T4 scene.

        Args:
            path (Path): Candidate directory.

        Returns:
            bool: ``True`` when ``annotation/sample.json`` exists.
        """
        return (path / "annotation" / "sample.json").exists()

    def _find_t4_scenes(self, dataset_base: Path) -> List[Path]:
        """Find T4 scenes at or below a dataset directory.

        Args:
            dataset_base (Path): Directory to search.

        Returns:
            List[Path]: Sorted T4 scene directories.
        """
        if self._is_t4_scene(dataset_base):
            return [dataset_base]
        # A scene is any directory (at any depth) holding annotation/sample.json.
        scenes = {
            sample_json.parent.parent
            for sample_json in dataset_base.rglob("annotation/sample.json")
        }
        return sorted(scenes)

    def _index_openlabels(self) -> Dict[str, Path]:
        """Index OpenLABEL files by plausible scene identifiers.

        Returns:
            Dict[str, Path]: Annotation paths keyed by file and metadata IDs.
        """
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
        """Read OpenLABEL metadata without failing on malformed files.

        Args:
            path (Path): OpenLABEL JSON path.

        Returns:
            dict: Metadata mapping, or an empty mapping when unreadable.
        """
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

        Args:
            scene_dir (Path): T4 scene directory.
            openlabels (Dict[str, Path]): Indexed OpenLABEL paths.

        Returns:
            Optional[Path]: Matching OpenLABEL path, if found.
        """
        root = Path(self._input_base).resolve()
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
        """Import one OpenLABEL annotation into a T4 scene.

        Args:
            scene_dir (Path): Prepared output T4 scene.
            openlabel_path (Path): Source OpenLABEL JSON file.

        Returns:
            None
        """
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

        tables = self._init_annotation_tables(scene_dir)
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

        Args:
            scene_dir (Path): T4 scene directory.

        Returns:
            Tuple[_SampleIndex, str]: Sample lookup index and selected lidar
                channel.
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
        """Convert a Kognic cuboid to T4 box geometry.

        Args:
            val (List[float]): Kognic cuboid values.
            ego_pose (dict): T4 ego pose for the frame.

        Returns:
            Tuple[List[float], List[float], List[float]]: Translation, size,
                and quaternion in T4 conventions.
        """
        return cuboid_val_to_t4_box(val, ego_pose, self._iso_rotated_cuboids)

    # ------------------------------------------------------------------
    # Table building
    # ------------------------------------------------------------------

    @classmethod
    def _init_annotation_tables(cls, scene_dir: Path) -> Dict[str, TableHandler]:
        """Create table handlers used by a box-annotation import.

        Args:
            scene_dir (Path): T4 scene directory.

        Returns:
            Dict[str, TableHandler]: Table handlers keyed by table name.
        """
        return {
            "category": cls._load_category_table(scene_dir),
            "instance": TableHandler(Instance),
            "attribute": TableHandler(Attribute),
            "visibility": TableHandler(Visibility),
            "sample_annotation": TableHandler(SampleAnnotation),
        }

    @staticmethod
    def _load_category_table(scene_dir: Path) -> TableHandler:
        """Seed the category table from the scene's existing ``category.json``.

        A scene may receive several conversions (e.g. a bbox OpenLABEL and a
        semseg OpenLABEL); each run saves ``category.json``, so starting from
        the on-disk table lets the runs compose instead of clobbering each
        other's categories.

        Args:
            scene_dir (Path): T4 scene directory.

        Returns:
            TableHandler: Existing or empty category table handler.
        """
        path = scene_dir / "annotation" / "category.json"
        if path.exists():
            return TableHandler.from_json(Category, str(path))
        return TableHandler(Category)

    def _get_or_create_instance(
        self,
        tables: Dict[str, TableHandler],
        instance_tokens: Dict[str, str],
        object_uuid: str,
        category_name: str,
    ) -> str:
        """Get or create the T4 instance for an OpenLABEL object.

        Args:
            tables (Dict[str, TableHandler]): Mutable annotation tables.
            instance_tokens (Dict[str, str]): Object-to-instance token cache.
            object_uuid (str): OpenLABEL object UUID.
            category_name (str): T4 category name.

        Returns:
            str: Existing or newly created instance token.
        """
        if object_uuid in instance_tokens:
            return instance_tokens[object_uuid]

        # Reuse the existing category for this name, else create one with the
        # next sequential index. The index doubles as the rerun ClassId during
        # visualization, which must be a non-negative uint16 (-1 overflows).
        category_token = tables["category"].get_token_from_field(
            field_name="name", field_value=category_name
        )
        if not category_token:
            # max+1 rather than len(): the seeded table may hold semseg
            # categories whose indices (ontology ids) are sparse, and a lidarseg
            # index must never be shared by two categories.
            next_index = (
                max(
                    (r.index for r in tables["category"].to_records() if r.index is not None),
                    default=-1,
                )
                + 1
            )
            category_token = tables["category"].insert_into_table(
                name=category_name,
                description="",
                index=next_index,
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
        """Convert OpenLABEL properties to T4 attribute tokens.

        Args:
            tables (Dict[str, TableHandler]): Mutable annotation tables.
            object_data (dict): OpenLABEL object-data mapping.

        Returns:
            List[str]: T4 attribute tokens.
        """
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
        """Get or create a T4 attribute token.

        Args:
            tables (Dict[str, TableHandler]): Mutable annotation tables.
            name (str): Fully qualified T4 attribute name.

        Returns:
            str: Attribute token.
        """
        return tables["attribute"].insert_into_table(
            reuse_if_duplicate=True, name=name, description=""
        )

    def _visibility_token(self, tables: Dict[str, TableHandler], object_data: dict) -> str:
        """Get or create the visibility token for OpenLABEL object data.

        Args:
            tables (Dict[str, TableHandler]): Mutable annotation tables.
            object_data (dict): OpenLABEL object-data mapping.

        Returns:
            str: T4 visibility token.
        """
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
        """Link per-instance annotations and update instance summaries.

        Args:
            tables (Dict[str, TableHandler]): Mutable annotation tables.
            instance_annotations (Dict[str, List[Tuple[int, str]]]): Frame and
                annotation tokens keyed by instance token.

        Returns:
            None
        """
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

        Args:
            scene_dir (Path): T4 scene directory.
            openlabel (dict): Parsed OpenLABEL document body.
            sample_index (_SampleIndex): Frame-to-sample lookup index.
            lidar_channel (str): T4 lidar channel receiving segmentation.

        Returns:
            None
        """
        frames = openlabel.get("frames", {})

        # Ontology id -> class name; the id doubles as the T4 category index and
        # the per-point label value stored in the .bin file.
        ontology = _segmentation_ontology(openlabel)
        if not ontology:
            logger.warning(f"No segmentation ontology found in annotation; skipping {scene_dir}")
            return
        if max(ontology) > np.iinfo(np.uint8).max:
            raise ValueError(
                f"Segmentation ontology id {max(ontology)} does not fit in uint8 lidarseg labels"
            )
        # RLE label value -> ontology id (countable objects are encoded as
        # per-instance classification_ids rather than ontology ids).
        value_map = _segmentation_value_map(openlabel, ontology)

        # Lidar sample_data record keyed by the sample it belongs to.
        lidar_sd_by_sample: Dict[str, dict] = {
            sd["sample_token"]: sd
            for sd in self._load_table(scene_dir, "sample_data.json")
            if lidar_channel in sd["filename"]
        }

        category_table = self._load_category_table(scene_dir)
        # Reserve index 0 for points the annotator left unlabelled. Skip names
        # already present (seeded from a previous conversion / rerun).
        if category_table.get_token_from_field("name", "background") is None:
            category_table.insert_into_table(
                name="background", description="unlabelled / background points", index=0
            )
        for index in sorted(ontology):
            if category_table.get_token_from_field("name", ontology[index]) is None:
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

        channel_by_sensor_token = {
            s["token"]: s["channel"] for s in self._load_table(scene_dir, "sensor.json")
        }

        placed = 0
        skipped = 0
        for frame_key, frame in sorted(frames.items(), key=lambda kv: int(kv[0])):
            rles = _frame_segmentation_rles(frame)
            if not rles:
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

            num_points = _lidar_point_count(scene_dir / sample_data["filename"])
            if num_points is None:
                logger.warning(
                    f"Could not read {sample_data['filename']} for frame {frame_key}; "
                    f"skipping its segmentation"
                )
                skipped += 1
                continue

            if len(rles) == 1:
                # A single blob covers the whole (fused) cloud, whatever its
                # stream tag; only per-source splits need LIDAR_CONCAT_INFO.
                labels = self._single_stream_labels(
                    next(iter(rles.values())),
                    value_map,
                    num_points,
                    frame_key,
                    sample_data["filename"],
                )
            else:
                labels = self._stitch_stream_labels(
                    scene_dir,
                    sample_data,
                    rles,
                    value_map,
                    num_points,
                    channel_by_sensor_token,
                    frame_key,
                )
            if labels is None:
                skipped += 1
                continue

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

    @staticmethod
    def _single_stream_labels(
        rle: str,
        value_map: Dict[int, int],
        num_points: int,
        frame_key: str,
        filename: str,
    ) -> Optional[np.ndarray]:
        """Decode labels for a frame represented by one lidar stream.

        Args:
            rle (str): Kognic run-length encoded labels.
            value_map (Dict[int, int]): Raw-label to category-index mapping.
            num_points (int): Expected point count.
            frame_key (str): Frame identifier used in diagnostics.
            filename (str): Point-cloud filename used in diagnostics.

        Returns:
            Optional[np.ndarray]: One uint8 label per point, or ``None`` when
                the annotation contains more labels than the cloud has points.
        """
        labels = _remap_labels(_decode_rle_labels(rle), value_map, frame_key)
        if labels.shape[0] > num_points:
            # More labels than points means the annotated cloud is not this
            # extraction at all (a genuine data mismatch); aligning is unsafe.
            logger.warning(
                f"Segmentation has more labels than points for frame {frame_key} "
                f"({filename}): {labels.shape[0]} labels vs {num_points} points. "
                f"The annotated cloud differs from this T4 extraction; skipping this frame."
            )
            return None
        if labels.shape[0] < num_points:
            # Kognic RLE encodes labels sequentially from point 0 and omits a
            # trailing run of unlabelled points; restore them as background (0).
            pad = num_points - labels.shape[0]
            logger.warning(
                f"Frame {frame_key}: RLE covers {labels.shape[0]}/{num_points} points; "
                f"padding {pad} trailing point(s) as background (class 0)."
            )
            labels = np.concatenate([labels, np.zeros(pad, dtype=np.uint8)])
        return labels

    def _stitch_stream_labels(
        self,
        scene_dir: Path,
        sample_data: dict,
        rles: Dict[Optional[str], str],
        value_map: Dict[int, int],
        num_points: int,
        channel_by_sensor_token: Dict[str, str],
        frame_key: str,
    ) -> Optional[np.ndarray]:
        """Labels for a frame annotated per source lidar stream.

        The upload split LIDAR_CONCAT into per-sensor Kognic streams using the
        LIDAR_CONCAT_INFO slice table (``sensor_token``/``idx_begin``/``length``),
        so each stream's RLE is written back into its slice of the concat cloud.

        Args:
            scene_dir (Path): T4 scene directory.
            sample_data (dict): Fused-lidar sample-data record.
            rles (Dict[Optional[str], str]): Encoded labels keyed by stream.
            value_map (Dict[int, int]): Raw-label to category-index mapping.
            num_points (int): Total fused-cloud point count.
            channel_by_sensor_token (Dict[str, str]): Sensor-token to channel map.
            frame_key (str): Frame identifier used in diagnostics.

        Returns:
            Optional[np.ndarray]: Stitched uint8 labels, or ``None`` when source
                metadata is missing or inconsistent.
        """
        info_filename = sample_data.get("info_filename")
        info_path = scene_dir / info_filename if info_filename else None
        if info_path is None or not info_path.exists():
            logger.warning(
                f"Frame {frame_key}: segmentation is split per lidar stream but "
                f"LIDAR_CONCAT_INFO is missing for {sample_data['filename']}; skipping"
            )
            return None
        with open(info_path) as f:
            sources = json.load(f)["sources"]

        labels = np.zeros(num_points, dtype=np.uint8)
        matched_streams = set()
        for source in sources:
            channel = channel_by_sensor_token.get(source["sensor_token"])
            idx_begin, length = int(source["idx_begin"]), int(source["length"])
            if length == 0:
                continue
            rle = rles.get(channel)
            if rle is None:
                logger.warning(
                    f"Frame {frame_key}: no RLE labels for stream {channel}; leaving "
                    f"its {length} point(s) as background"
                )
                continue
            if idx_begin + length > num_points:
                logger.warning(
                    f"Frame {frame_key}: LIDAR_CONCAT_INFO slice for {channel} "
                    f"([{idx_begin}, {idx_begin + length})) exceeds the {num_points}-point "
                    f"cloud; skipping this frame"
                )
                return None
            stream_labels = _remap_labels(
                _decode_rle_labels(rle), value_map, f"{frame_key}/{channel}"
            )
            if stream_labels.shape[0] > length:
                logger.warning(
                    f"Frame {frame_key}: stream {channel} has {stream_labels.shape[0]} labels "
                    f"for a {length}-point slice; the annotated cloud differs from this T4 "
                    f"extraction; skipping this frame"
                )
                return None
            if stream_labels.shape[0] < length:
                pad = length - stream_labels.shape[0]
                logger.warning(
                    f"Frame {frame_key}: stream {channel} RLE covers "
                    f"{stream_labels.shape[0]}/{length} points; padding {pad} trailing "
                    f"point(s) as background (class 0)."
                )
                stream_labels = np.concatenate([stream_labels, np.zeros(pad, dtype=np.uint8)])
            labels[idx_begin : idx_begin + length] = stream_labels
            matched_streams.add(channel)

        unmatched = set(rles) - matched_streams
        if unmatched:
            logger.warning(
                f"Frame {frame_key}: RLE stream(s) {sorted(str(s) for s in unmatched)} have no "
                f"matching LIDAR_CONCAT_INFO source; their labels were dropped"
            )
        return labels

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    @staticmethod
    def _load_table(scene_dir: Path, name: str) -> list:
        """Load an optional T4 annotation table.

        Args:
            scene_dir (Path): T4 scene directory.
            name (str): Annotation-table filename.

        Returns:
            list: Parsed records, or an empty list when the file is absent.
        """
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

        Args:
            scene_dir (Path): T4 scene directory.
            sample_annotation (TableHandler): Sample-annotation table.
            lidar_channel (str): Lidar channel used for point counting.

        Returns:
            None
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
        """Write populated annotation tables into a scene.

        Args:
            scene_dir (Path): T4 scene directory.
            tables (Dict[str, TableHandler]): Tables to save.

        Returns:
            None
        """
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
        """Build the sorted timestamp index used for nearest matching."""
        self._sorted_us: List[int] = sorted(self.by_timestamp_us)

    def match(self, frame: dict, frame_key: str, lidar_channel: str) -> Optional[Tuple[str, dict]]:
        """Match an OpenLABEL frame to a T4 sample and ego pose.

        Args:
            frame (dict): OpenLABEL frame mapping.
            frame_key (str): OpenLABEL frame key used for diagnostics.
            lidar_channel (str): Preferred T4 lidar channel.

        Returns:
            Optional[Tuple[str, dict]]: Sample token and ego pose, if matched.
        """
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
        """Extract a lidar capture timestamp from a frame.

        Args:
            frame (dict): OpenLABEL frame mapping.
            lidar_channel (str): Preferred T4 lidar channel.

        Returns:
            Optional[int]: Capture timestamp in nanoseconds, if available.
        """
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

        Args:
            streams (dict): OpenLABEL frame stream mapping.
            lidar_channel (str): Preferred T4 lidar channel.

        Returns:
            Optional[dict]: Selected stream mapping, if found.
        """
        for key in (lidar_channel, "lidar"):
            if key in streams:
                return streams[key]
        for key, value in streams.items():
            if "lidar" in key.lower():
                return value
        return None

    def _nearest(self, ts_us: int) -> Optional[Tuple[str, Optional[dict]]]:
        """Find the sample nearest a timestamp within tolerance.

        Args:
            ts_us (int): Capture timestamp in microseconds.

        Returns:
            Optional[Tuple[str, Optional[dict]]]: Sample token and ego pose.
        """
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
        """Match a frame by its positional external ID.

        Args:
            frame (dict): OpenLABEL frame mapping.

        Returns:
            Optional[Tuple[str, Optional[dict]]]: Sample token and ego pose.
        """
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

    Args:
        uri (str): OpenLABEL stream URI.

    Returns:
        Optional[int]: Capture timestamp in nanoseconds, if parseable.
    """
    token = Path(uri).stem.rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return None


# ``#<run_length>V<class_id>`` repeated; the Kognic RLE encoding of per-point labels.
_RLE_TOKEN = re.compile(r"#(\d+)V(\d+)")


def _is_segmentation(openlabel: dict) -> bool:
    """Check whether an OpenLABEL document contains point-cloud segmentation.

    Args:
        openlabel (dict): Parsed OpenLABEL document body.

    Returns:
        bool: ``True`` when segmentation metadata or frame labels are present.
    """
    if openlabel.get("metadata", {}).get("annotation_type") == "semseg":
        return True
    return any(_frame_segmentation_rles(frame) for frame in openlabel.get("frames", {}).values())


def _segmentation_ontology(openlabel: dict) -> Dict[int, str]:
    """Extract segmentation class names by ontology ID.

    Args:
        openlabel (dict): Parsed OpenLABEL document body.

    Returns:
        Dict[int, str]: Class names keyed by numeric ontology ID.
    """
    ontology: Dict[int, str] = {}
    for entry in openlabel.get("ontologies", {}).values():
        for class_id, name in entry.get("classifications", {}).items():
            try:
                ontology[int(class_id)] = name
            except (TypeError, ValueError):
                continue
    return ontology


def _frame_segmentation_rles(frame: dict) -> Dict[Optional[str], str]:
    """Map lidar stream name -> RLE label string for a frame.

    Multi-lidar scenes carry one ``3DPointCloudSegmentation`` blob per source
    lidar (tagged with a ``stream`` text attribute); single-lidar scenes carry
    one untagged blob, keyed here as ``None``.

    Args:
        frame (dict): OpenLABEL frame mapping.

    Returns:
        Dict[Optional[str], str]: RLE strings keyed by lidar stream.
    """
    rles: Dict[Optional[str], str] = {}
    for frame_object in frame.get("objects", {}).values():
        for binary in frame_object.get("object_data", {}).get("binary", []):
            if binary.get("name") == "labels" and binary.get("encoding") == "rle":
                stream = next(
                    (
                        text.get("val")
                        for text in binary.get("attributes", {}).get("text", [])
                        if text.get("name") == "stream"
                    ),
                    None,
                )
                rles[stream] = binary.get("val")
    return rles


def _decode_rle_labels(val: str) -> np.ndarray:
    """Expand a Kognic RLE string into per-point label values.

    Args:
        val (str): Repeated ``#<count>V<class>`` tokens.

    Returns:
        np.ndarray: Decoded integer class values.
    """
    pairs = _RLE_TOKEN.findall(val)
    counts = np.fromiter((int(c) for c, _ in pairs), dtype=np.int64, count=len(pairs))
    classes = np.fromiter((int(v) for _, v in pairs), dtype=np.int64, count=len(pairs))
    return np.repeat(classes, counts)


def _segmentation_value_map(openlabel: dict, ontology: Dict[int, str]) -> Dict[int, int]:
    """Map an RLE label value to its T4 category index (ontology id).

    Kognic semseg RLEs mix two value spaces: stuff classes are encoded
    directly as ontology ids, while countable objects are encoded as the
    per-instance ``classification_id`` declared on the top-level object,
    whose ``type`` names the ontology class.

    Args:
        openlabel (dict): Parsed OpenLABEL document body.
        ontology (Dict[int, str]): Class names keyed by ontology ID.

    Returns:
        Dict[int, int]: Raw RLE values mapped to T4 category indices.
    """
    name_to_id = {name: class_id for class_id, name in ontology.items()}
    value_map = {class_id: class_id for class_id in ontology}
    for obj in openlabel.get("objects", {}).values():
        class_id = next(
            (
                num.get("val")
                for num in obj.get("object_data", {}).get("num", [])
                if num.get("name") == "classification_id"
            ),
            None,
        )
        if class_id is None:
            continue
        index = name_to_id.get(obj.get("type"))
        if index is None:
            logger.warning(
                f"Object {obj.get('name')} has type '{obj.get('type')}' not present in the "
                f"segmentation ontology; its points will be mapped to background"
            )
            continue
        value_map[int(class_id)] = index
    return value_map


def _remap_labels(labels: np.ndarray, value_map: Dict[int, int], frame_key: str) -> np.ndarray:
    """Convert raw RLE values into uint8 T4 category indices.

    Args:
        labels (np.ndarray): Decoded raw label values.
        value_map (Dict[int, int]): Raw-value to category-index mapping.
        frame_key (str): Frame identifier used in diagnostics.

    Returns:
        np.ndarray: Remapped uint8 category indices.
    """
    unmapped = sorted(set(np.unique(labels).tolist()) - set(value_map) - {0})
    if unmapped:
        logger.warning(
            f"Frame {frame_key}: {len(unmapped)} RLE label value(s) have no ontology/object "
            f"mapping (e.g. {unmapped[:5]}); mapping them to background (0)"
        )
    size = int(max(labels.max(initial=0), max(value_map, default=0))) + 1
    lut = np.zeros(size, dtype=np.uint8)
    for value, index in value_map.items():
        lut[value] = index
    return lut[labels]


def _lidar_point_count(bin_path: Path) -> Optional[int]:
    """Count points in a fused-lidar binary file.

    Args:
        bin_path (Path): Path to a ``.pcd.bin`` file.

    Returns:
        Optional[int]: Point count, or ``None`` when the file is missing.
    """
    if not bin_path.exists():
        return None
    floats = np.fromfile(bin_path, dtype=np.float32)
    if floats.size == 0:
        return 0
    return floats.size // detect_point_stride(floats, bin_path)
