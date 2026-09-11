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

OpenLABEL frames are matched to T4 samples by the LiDAR stream's URI timestamp.
Uploads that split ``LIDAR_CONCAT`` into per-sensor streams name their files
with each source sensor's own capture stamp, so those stamps are indexed from
``LIDAR_CONCAT_INFO`` alongside the sample timestamps.

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
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from t4_devkit import Tier4
from t4_devkit.common.serialize import serialize_dataclass
from t4_devkit.dataclass import LidarPointCloud, PointCloudMetainfo
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
from perception_dataset.utils.t4_tables import (
    channel_by_calibrated_sensor,
    select_lidar_channel,
)

logger = configure_logger(modname=__name__)

# Points the annotator left unlabelled, plus any point a decoded RLE omits or
# fails to map, are written as this class. Segmentation owns the low indices so
# this stays 0 no matter which annotation is imported first.
BACKGROUND_CATEGORY_NAME = "background"
BACKGROUND_CATEGORY_INDEX = 0
BACKGROUND_CATEGORY_DESCRIPTION = "unlabelled / background points"


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
        self._t4_table_cache: Dict[Tuple[Path, str], list] = {}

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
            annotation_paths = self._match_openlabels(scene_dir, openlabels)
            if not annotation_paths:
                logger.warning(
                    f"No matching OpenLABEL annotation for scene {scene_dir.name}; skipping"
                )
                continue

            output_dir = self._prepare_output_scene(scene_dir)
            for openlabel_path in annotation_paths:
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
                output exists while overwrite mode is disabled, or overwriting
                the output would delete a directory containing the source scene.
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
            if scene_dir.is_relative_to(output_dir.resolve()):
                raise ValueError(
                    f"Cannot overwrite output directory {output_dir}: it contains "
                    f"the source scene {scene_dir}; overwriting would delete the source."
                )
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
            if item not in ["annotation", "data", "lidarseg", "status.json"]:
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

    def _index_openlabels(self) -> Dict[str, List[Path]]:
        """Index OpenLABEL files by plausible scene identifiers.

        A scene sometimes has several annotations (e.g. one cuboid and one
        semseg request), so each identifier maps to every file claiming it
        rather than to the first one seen.

        Returns:
            Dict[str, List[Path]]: Annotation paths keyed by file and metadata IDs.
        """
        index: Dict[str, List[Path]] = {}
        if not self._annotation_base.exists():
            return index

        def add(key: str, path: Path) -> None:
            paths = index.setdefault(key, [])
            if path not in paths:
                paths.append(path)

        files = (
            [self._annotation_base]
            if self._annotation_base.is_file()
            else sorted(self._annotation_base.rglob("*.json"))
        )
        for path in files:
            add(path.stem, path)
            metadata = self._read_metadata(path)
            for key in ("dataset_id", "source_filename", "scene_uuid", "input_external_id"):
                value = metadata.get(key)
                if value:
                    add(str(value), path)
            scene_metadata = metadata.get("scene_metadata") or {}
            for key in ("dataset_id", "source_filename", "inner_uuid"):
                value = scene_metadata.get(key)
                if value:
                    add(str(value), path)
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

    def _match_openlabels(self, scene_dir: Path, openlabels: Dict[str, List[Path]]) -> List[Path]:
        """Match a scene to its OpenLABEL files by the scene dir name or any of
        its ancestor dir names up to the dataset root.

        T4 datasets are commonly nested as ``<root>/<scene_id>/<version>/``,
        so the matching identifier is often an ancestor (e.g. ``<scene_id>``)
        rather than the leaf (e.g. version ``0``).

        Args:
            scene_dir (Path): T4 scene directory.
            openlabels (Dict[str, List[Path]]): Indexed OpenLABEL paths.

        Returns:
            List[Path]: One annotation per type, boxes first.
        """
        root = Path(self._input_base).resolve()
        current = scene_dir.resolve()
        while True:
            if current.name in openlabels:
                return self._select_by_type(scene_dir, openlabels[current.name])
            if current == root or current.parent == current:
                return []
            current = current.parent

    def _select_by_type(self, scene_dir: Path, candidates: List[Path]) -> List[Path]:
        """Reduce candidate annotations to one per annotation type.

        Boxes are applied before segmentation so the box categories seed
        ``category.json`` before the ontology classes extend it. Competing
        files of the same type have no defined precedence, so that type is
        skipped and the caller is told to name one explicitly.

        Args:
            scene_dir (Path): T4 scene the candidates were matched to.
            candidates (List[Path]): Annotation paths claiming this scene.

        Returns:
            List[Path]: Selected annotation paths in application order.
        """
        by_type: Dict[str, List[Path]] = {}
        for path in candidates:
            by_type.setdefault(self._annotation_type(path), []).append(path)

        selected: List[Path] = []
        for annotation_type in ("boxes", "segmentation"):
            paths = by_type.get(annotation_type, [])
            if len(paths) > 1:
                logger.error(
                    f"Scene {scene_dir.name} matches {len(paths)} {annotation_type} "
                    f"annotations ({', '.join(p.name for p in paths)}); skipping this type. "
                    f"Point annotation_base at a single file to choose one."
                )
                continue
            selected.extend(paths)
        return selected

    @staticmethod
    def _annotation_type(path: Path) -> str:
        """Classify an OpenLABEL file as boxes or segmentation.

        Args:
            path (Path): OpenLABEL JSON path.

        Returns:
            str: ``"segmentation"`` or ``"boxes"``.
        """
        try:
            with open(path) as f:
                openlabel = json.load(f).get("openlabel", {})
        except (json.JSONDecodeError, OSError):
            return "boxes"
        return "segmentation" if _is_segmentation(openlabel) else "boxes"

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

        OpenLABEL frames carry their lidar stream ``uri``, whose stem is the
        absolute-ns capture timestamp of the exported cloud. For a fused
        ``LIDAR_CONCAT`` upload that is the ``sample_data`` timestamp, but a
        per-sensor split upload uses each source's own stamp, which can sit tens
        of milliseconds away; both are therefore indexed onto the same sample.

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

        lidar_sd_by_sample = self._select_lidar_sample_data(
            sample, sample_data, channel_by_calib, lidar_channel
        )
        entry_by_sample = {
            sample_token: (sample_token, ego_pose_by_token.get(record["ego_pose_token"]))
            for sample_token, record in lidar_sd_by_sample.items()
        }
        # ``extract_pointclouds`` names the exported cloud after
        # ``sample_data.timestamp``, so keying on that record makes the match
        # bit-exact; ``sample.timestamp`` is only an alias for it.
        by_timestamp_us = {
            record["timestamp"]: entry_by_sample[sample_token]
            for sample_token, record in lidar_sd_by_sample.items()
        }
        for s in sample:
            if s["token"] in entry_by_sample:
                by_timestamp_us.setdefault(s["timestamp"], entry_by_sample[s["token"]])
        self._index_source_timestamps(
            scene_dir, lidar_sd_by_sample, entry_by_sample, by_timestamp_us
        )
        return _SampleIndex(by_timestamp_us, lidar_sd_by_sample), lidar_channel

    @staticmethod
    def _select_lidar_sample_data(
        sample: List[dict],
        sample_data: List[dict],
        channel_by_calib: Dict[str, Optional[str]],
        lidar_channel: str,
    ) -> Dict[str, dict]:
        """Resolve the one lidar ``sample_data`` record backing each sample.

        A sample owns its keyframe record *and* the intermediate sweeps that
        follow it, all sharing its ``sample_token``. The keyframe is the record
        whose timestamp is the sample's; ``is_key_frame`` breaks the tie when no
        timestamp matches. A sample whose record cannot be pinned down is left
        out of the index rather than resolved by table order, since the sweeps
        of one sample can be metres apart.

        Args:
            sample (List[dict]): Sample table records.
            sample_data (List[dict]): Sample-data table records.
            channel_by_calib (Dict[str, Optional[str]]): Calibrated-sensor token
                to channel mapping.
            lidar_channel (str): Lidar channel to resolve.

        Returns:
            Dict[str, dict]: Lidar sample-data record keyed by sample token.
        """
        records_by_sample: Dict[str, List[dict]] = {}
        for record in sample_data:
            if channel_by_calib.get(record["calibrated_sensor_token"]) == lidar_channel:
                records_by_sample.setdefault(record["sample_token"], []).append(record)

        selected: Dict[str, dict] = {}
        for s in sample:
            candidates = records_by_sample.get(s["token"], [])
            exact = [r for r in candidates if r["timestamp"] == s["timestamp"]]
            if len(exact) != 1:
                exact = [r for r in candidates if r.get("is_key_frame")]
            if len(exact) != 1:
                logger.warning(
                    f"Sample {s['token']} has {len(candidates)} {lidar_channel} sample_data "
                    f"record(s) and no unambiguous keyframe; excluding it from the frame index"
                )
                continue
            selected[s["token"]] = exact[0]
        return selected

    @staticmethod
    def _index_source_timestamps(
        scene_dir: Path,
        lidar_sd_by_sample: Dict[str, dict],
        entry_by_sample: Dict[str, Tuple[str, Optional[dict]]],
        by_timestamp_us: Dict[int, Tuple[str, Optional[dict]]],
    ) -> None:
        """Add each concat source's own capture stamp to the timestamp index.

        ``extract_pointclouds`` names a per-sensor CSV after that sensor's
        ``LIDAR_CONCAT_INFO`` stamp, which is offset from the fused sweep's
        timestamp by up to a large fraction of the frame period. Mapping those
        stamps back onto their concat sample keeps the match exact instead of
        relying on a tolerance wide enough to hit a neighbouring sweep.

        Args:
            scene_dir (Path): T4 scene directory.
            lidar_sd_by_sample (Dict[str, dict]): Selected lidar sample-data
                record keyed by sample token.
            entry_by_sample (Dict[str, Tuple[str, Optional[dict]]]): Index entry
                keyed by sample token.
            by_timestamp_us (Dict[int, Tuple[str, Optional[dict]]]): Timestamp
                index updated in place.

        Returns:
            None
        """
        for sample_token, record in lidar_sd_by_sample.items():
            info_filename = record.get("info_filename")
            entry = entry_by_sample.get(sample_token)
            if not info_filename or entry is None:
                continue
            info_path = scene_dir / info_filename
            if not info_path.exists():
                logger.warning(f"LIDAR_CONCAT_INFO is missing: {info_path}")
                continue
            metainfo = PointCloudMetainfo.from_file(str(info_path))
            for source in metainfo.sources:
                ts_ns = source.stamp.sec * 1_000_000_000 + source.stamp.nanosec
                if not ts_ns:
                    continue
                # setdefault: a real sample timestamp always wins a collision.
                by_timestamp_us.setdefault(round(ts_ns / 1000), entry)

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
            # categories whose indices are sparse, and a lidarseg index must
            # never be shared. A later segmentation import moves this category
            # again if its ontology needs the index.
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

    @staticmethod
    def _assign_segmentation_categories(
        category_table: TableHandler, ontology: Dict[int, str]
    ) -> Dict[int, int]:
        """Reserve the low category indices for segmentation and return the mapping.

        Lidarseg ``.bin`` files store a category ``index`` per point, so those
        indices must be stable and unique. ``background`` therefore always owns
        index 0 and each ontology class keeps its ontology id as its index,
        which holds whether this scene already carries bbox categories or is
        annotated segmentation-first. Categories outside this ontology (bbox
        classes, or classes from an earlier ontology) are renumbered above the
        segmentation block; only ``category.json`` records their index, so
        moving them cannot invalidate existing annotations.

        Args:
            category_table (TableHandler): Category table to reconcile in place.
            ontology (Dict[int, str]): Class names keyed by ontology ID.

        Returns:
            Dict[int, int]: Ontology ID to assigned T4 category index.

        Raises:
            ValueError: If an assigned index does not fit in a uint8 label.
        """
        # An ontology id of 0 would collide with background, so park it above
        # the ontology instead of shifting every other class.
        ceiling = max(ontology)
        index_by_class_id: Dict[int, int] = {}
        for class_id in sorted(ontology):
            if class_id == BACKGROUND_CATEGORY_INDEX:
                ceiling += 1
                index_by_class_id[class_id] = ceiling
            else:
                index_by_class_id[class_id] = class_id

        reserved: Dict[int, str] = {BACKGROUND_CATEGORY_INDEX: BACKGROUND_CATEGORY_NAME}
        for class_id, index in index_by_class_id.items():
            reserved[index] = ontology[class_id]

        if max(reserved) > np.iinfo(np.uint8).max:
            raise ValueError(
                f"Segmentation category index {max(reserved)} does not fit in uint8 "
                f"lidarseg labels"
            )

        reserved_names = set(reserved.values())
        existing = list(category_table.to_records())
        next_free = (
            max(
                [max(reserved)] + [r.index for r in existing if r.index is not None],
            )
            + 1
        )
        for record in existing:
            if record.name in reserved_names:
                continue
            if record.index is None or record.index in reserved:
                category_table.update_record_from_token(record.token, index=next_free)
                next_free += 1

        for index, name in sorted(reserved.items()):
            token = category_table.get_token_from_field("name", name)
            description = (
                BACKGROUND_CATEGORY_DESCRIPTION if name == BACKGROUND_CATEGORY_NAME else ""
            )
            if token is None:
                category_table.insert_into_table(name=name, description=description, index=index)
            else:
                category_table.update_record_from_token(token, index=index)

        return index_by_class_id

    @staticmethod
    def _verify_label_categories(category_table: TableHandler, emitted_labels: Set[int]) -> None:
        """Check that every written lidarseg label resolves to one category.

        Args:
            category_table (TableHandler): Reconciled category table.
            emitted_labels (Set[int]): Label values written to ``.bin`` files.

        Returns:
            None

        Raises:
            ValueError: If a label has no category, or its index is shared.
        """
        names_by_index: Dict[int, List[str]] = {}
        for record in category_table.to_records():
            names_by_index.setdefault(record.index, []).append(record.name)

        duplicated = {i: n for i, n in names_by_index.items() if len(n) > 1}
        if duplicated:
            raise ValueError(f"category.json assigns one index to several categories: {duplicated}")

        orphaned = sorted(label for label in emitted_labels if label not in names_by_index)
        if orphaned:
            raise ValueError(
                f"lidarseg labels {orphaned} have no category.json entry; the segmentation "
                f"ontology and the category indices disagree"
            )

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

        # Ontology id -> class name.
        ontology = _segmentation_ontology(openlabel)
        if not ontology:
            logger.warning(f"No segmentation ontology found in annotation; skipping {scene_dir}")
            return

        lidar_sd_by_sample = sample_index.sample_data_by_sample

        category_table = self._load_category_table(scene_dir)
        index_by_class_id = self._assign_segmentation_categories(category_table, ontology)
        # RLE label value -> T4 category index (countable objects are encoded as
        # per-instance classification_ids rather than ontology ids).
        value_map = _segmentation_value_map(openlabel, ontology, index_by_class_id)

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
        emitted_labels: Set[int] = set()
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

            info_filename = sample_data.get("info_filename")
            info_path = scene_dir / info_filename if info_filename else None
            num_points = _lidar_point_count(
                scene_dir / sample_data["filename"],
                info_path=info_path,
            )
            if num_points is None:
                logger.warning(
                    f"Could not read {sample_data['filename']} for frame {frame_key}; "
                    f"skipping its segmentation"
                )
                skipped += 1
                continue

            only_stream_tag = next(iter(rles)) if len(rles) == 1 else None
            if len(rles) == 1 and only_stream_tag is None:
                # A single untagged blob covers the whole (fused) cloud. A
                # single *tagged* blob still names one source lidar and must
                # go through its LIDAR_CONCAT_INFO slice, even when it is the
                # only annotated stream this frame.
                labels = self._single_stream_labels(
                    rles[only_stream_tag],
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
            emitted_labels.update(np.unique(labels).tolist())
            placed += 1

        self._verify_label_categories(category_table, emitted_labels)

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
                the RLE is malformed or declares more points than the cloud has.
        """
        try:
            decoded = _decode_rle_labels(rle, max_points=num_points, frame_key=frame_key)
        except ValueError as exc:
            # Either malformed content, or more labels than points, which means
            # the annotated cloud is not this extraction at all (a genuine data
            # mismatch); aligning either case is unsafe.
            logger.warning(f"Frame {frame_key} ({filename}): {exc}; skipping this frame")
            return None
        labels = _remap_labels(decoded, value_map, frame_key)
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
            try:
                decoded = _decode_rle_labels(
                    rle, max_points=length, frame_key=f"{frame_key}/{channel}"
                )
            except ValueError as exc:
                # Either malformed content, or more labels than the slice has
                # points, which means the annotated cloud differs from this T4
                # extraction; aligning either case is unsafe.
                logger.warning(f"{exc}; skipping this frame")
                return None
            stream_labels = _remap_labels(decoded, value_map, f"{frame_key}/{channel}")
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

    def _load_table(self, scene_dir: Path, name: str) -> list:
        """Load a T4 table, using the devkit for core dataset tables.

        The converter keeps dictionaries internally because OpenLABEL and
        concat metadata are handled as JSON, but the core T4 sensor tables are
        decoded by ``Tier4`` first so their schema and field interpretation
        stay centralized in t4-devkit.

        Args:
            scene_dir (Path): T4 scene directory.
            name (str): Annotation-table filename.

        Returns:
            list: Parsed records, or an empty list when the file is absent.
        """
        core_tables = {
            "sample.json",
            "sample_data.json",
            "sensor.json",
            "calibrated_sensor.json",
            "ego_pose.json",
        }
        cache_key = (scene_dir.resolve(), name)
        if name in core_tables:
            if cache_key not in self._t4_table_cache:
                t4_dataset = Tier4(data_root=str(scene_dir), verbose=False)
                table_name = Path(name).stem
                self._t4_table_cache[cache_key] = [
                    serialize_dataclass(record) for record in t4_dataset.get_table(table_name)
                ]
            return self._t4_table_cache[cache_key]

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
    """Resolve an OpenLABEL frame to its T4 (sample_token, ego_pose).

    Both the ego pose and ``sample_data_by_sample`` are derived from the same
    resolved lidar record, so poses and point clouds cannot disagree.
    """

    by_timestamp_us: Dict[int, Tuple[str, Optional[dict]]]
    sample_data_by_sample: Dict[str, dict]

    # Max |Δ| (µs) between an OpenLABEL capture time and an indexed timestamp
    # still treated as the same frame. A fused-upload uri matches its
    # ``sample_data`` key exactly, but per-source stamps are converted from
    # ``{sec, nanosec}`` and T4 timestamps come from a lossy float64 path
    # (``int((sec + nanosec * 1e-9) * 1e6)``), so the µs value can differ by ~1;
    # 1 ms is far below the
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
        # The lidar uri timestamp is the only ground truth: a frame whose
        # capture time has no nearby sample is genuinely unmatched (e.g.
        # annotation and point clouds from different recordings).
        ts_ns = self._uri_timestamp_ns(frame, lidar_channel)
        if ts_ns is None:
            return None
        candidate = self._nearest(round(ts_ns / 1000))
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


def _decode_rle_labels(val: str, max_points: int, frame_key: str) -> np.ndarray:
    """Expand a Kognic RLE string into per-point label values.

    Requires ``val`` to be fully covered by ``#<count>V<class>`` tokens (no
    interleaved or trailing garbage) and its declared point total to fit
    within ``max_points`` before expanding, so a single malformed or
    corrupted/adversarial count cannot force an oversized allocation.

    Args:
        val (str): Repeated ``#<count>V<class>`` tokens.
        max_points (int): Upper bound on the decoded point count, normally the
            target cloud's (or slice's) actual point count.
        frame_key (str): Frame identifier used in diagnostics.

    Returns:
        np.ndarray: Decoded integer class values.

    Raises:
        ValueError: If ``val`` is not fully covered by the RLE grammar, or its
            declared point total exceeds ``max_points``.
    """
    matches = list(_RLE_TOKEN.finditer(val))
    if not matches or "".join(match.group(0) for match in matches) != val:
        raise ValueError(
            f"Frame {frame_key}: RLE value is not fully covered by '#<count>V<class>' tokens"
        )
    counts = np.fromiter((int(m.group(1)) for m in matches), dtype=np.int64, count=len(matches))
    total_points = int(counts.sum())
    if total_points > max_points:
        raise ValueError(
            f"Frame {frame_key}: RLE declares {total_points} point(s), exceeding the "
            f"{max_points}-point cloud; refusing to expand"
        )
    classes = np.fromiter((int(m.group(2)) for m in matches), dtype=np.int64, count=len(matches))
    return np.repeat(classes, counts)


def _segmentation_value_map(
    openlabel: dict, ontology: Dict[int, str], index_by_class_id: Dict[int, int]
) -> Dict[int, int]:
    """Map an RLE label value to its assigned T4 category index.

    Kognic semseg RLEs mix two value spaces: stuff classes are encoded
    directly as ontology ids, while countable objects are encoded as the
    per-instance ``classification_id`` declared on the top-level object,
    whose ``type`` names the ontology class.

    Args:
        openlabel (dict): Parsed OpenLABEL document body.
        ontology (Dict[int, str]): Class names keyed by ontology ID.
        index_by_class_id (Dict[int, int]): Ontology ID to T4 category index.

    Returns:
        Dict[int, int]: Raw RLE values mapped to T4 category indices.
    """
    name_to_id = {name: class_id for class_id, name in ontology.items()}
    value_map = {class_id: index_by_class_id[class_id] for class_id in ontology}
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
        ontology_id = name_to_id.get(obj.get("type"))
        if ontology_id is None:
            logger.warning(
                f"Object {obj.get('name')} has type '{obj.get('type')}' not present in the "
                f"segmentation ontology; its points will be mapped to background"
            )
            continue
        value_map[int(class_id)] = index_by_class_id[ontology_id]
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
    unique_labels = np.unique(labels)
    unmapped = sorted(set(unique_labels.tolist()) - set(value_map) - {0})
    if unmapped:
        logger.warning(
            f"Frame {frame_key}: {len(unmapped)} RLE label value(s) have no ontology/object "
            f"mapping (e.g. {unmapped[:5]}); mapping them to background (0)"
        )
    # Remap by the labels actually present rather than a lookup table sized to
    # the largest raw value: a raw class ID is attacker/corruption controlled
    # and unbounded, while ``unique_labels`` is bounded by the point count.
    output = np.zeros(labels.shape, dtype=np.uint8)
    for raw_value in unique_labels:
        if raw_value == 0:
            continue
        index = value_map.get(int(raw_value))
        if index is not None:
            output[labels == raw_value] = index
    return output


def _lidar_point_count(
    bin_path: Path,
    info_path: Optional[Path] = None,
) -> Optional[int]:
    """Count points in a fused-lidar binary file.

    Args:
        bin_path (Path): Path to a ``.pcd.bin`` file.
        info_path (Optional[Path]): Corresponding ``LIDAR_CONCAT_INFO`` file.
            When present, it is supplied to the t4-devkit loader.

    Returns:
        Optional[int]: Point count, or ``None`` when the file is missing.
    """
    if not bin_path.exists():
        return None
    if info_path is not None and not info_path.exists():
        raise FileNotFoundError(f"Required LIDAR_CONCAT_INFO is missing: {info_path}")
    # Loading validates source coverage against the actual number of points.
    return LidarPointCloud.from_file(
        str(bin_path),
        metainfo_filepath=str(info_path) if info_path is not None else None,
    ).num_points()
