"""Convert T4 annotation tables to Kognic OpenLABEL pre-annotations."""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import uuid

from kognic.openlabel.models import models as openlabel
import numpy as np

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import LIDAR_CONCAT_CHANNEL, PREFERRED_LIDAR_SENSORS
from perception_dataset.kognic.openlabel import attribute_to_text, t4_box_to_cuboid_val
from perception_dataset.kognic.upload_dataset import _sensor_sort_key, _sort_key
from perception_dataset.kognic.utils import iter_scene_pairs
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.pointcloud import lidar_point_count
from perception_dataset.utils.t4_tables import (
    channel_by_calibrated_sensor,
    records_for_channel,
)

logger = configure_logger(modname=__name__)

# Kognic's stream literal for the merged lidar point cloud: segmentation RLEs
# address the merge of all per-sensor clouds, not an individual sensor stream
# (cf. https://docs.kognic.com/api-guide/pre-annotations and Kognic's own
# semseg OpenLabel exports).
MERGED_LIDAR_STREAM = "lidar"


class T4ToOpenLabelConverter(AbstractConverter[None]):
    """Convert T4 annotations to Kognic OpenLABEL pre-annotations.

    For every annotated T4 sequence under ``input_base``, the annotation kind
    is auto-detected per scene from the annotation tables present, and the
    matching pre-annotation files are written into the Kognic staging
    directory under ``output_base``, as previously produced by
    ``T4ToKognicConverter``::

        <output_base>/<scene>/
            calibration.json
            ego_poses.json
            cuboid_pre_annotation.json     <- from sample_annotation.json
            semseg_pre_annotation.json     <- from lidarseg.json
            cameras/...  lidar/...

    Conventions (https://docs.kognic.com/api-guide/pre-annotations):

    - Cuboids are expressed in the per-frame reference (ego/base_link)
      coordinate system, as required for multi-lidar scenes. This composes
      with the T0-normalised ego poses uploaded by ``KognicDatasetUploader``.
    - Cuboid ``val`` is ``[x, y, z, qx, qy, qz, qw, sx, sy, sz]`` with yaw 0
      facing +y, so T4 box rotations are post-multiplied by Rz(-90 deg) and
      T4 sizes (width, length, height) map unchanged.
    - Pre-annotation frames are matched to scene frames by timestamp, so
      ``frame_properties.timestamp`` mirrors the uploader's
      ``relative_timestamp`` (milliseconds since the first anchor frame).
    - A ``keyframes.json`` (staging frame indices of the T4 keyframes, from
      ``sample_data.is_key_frame``) is written next to the pre-annotation.
      The uploader marks exactly those frames ``annotate=True`` so the
      annotatable frames always line up with the pre-annotation frames:
      Kognic only surfaces pre-annotations on annotatable frames.

    Point-cloud segmentation (``annotation/lidarseg.json``, when present) is
    exported as ``semseg_pre_annotation.json``, the exact inverse of
    ``OpenLabelToT4Converter._convert_segmentation``: each frame carries one
    ``object_data.binary`` entry (``name="labels"``, ``encoding="rle"``, plus
    the ``stream`` text attribute Kognic requires to tie the labels to the
    LiDAR stream) whose value run-length encodes the per-point class ids
    (``#<count>V<class_id>``, ``0`` = unlabelled/background) in LIDAR_CONCAT
    point order, and the
    ``ontologies`` block maps each class id to its name from the T4
    ``category.json`` ``index`` fields. To attach it, point the uploader's
    project target at it: ``pre_annotation: semseg_pre_annotation.json``.
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        lidar_stream: str = "",
        category_map: Optional[Dict[str, str]] = None,
        include_attributes: bool = False,
        exclude_attributes: Optional[List[str]] = None,
        frame_match_tolerance_ms: float = 50.0,
    ):
        super().__init__(input_base, output_base)
        self._lidar_stream = lidar_stream
        self._category_map = category_map or {}
        self._include_attributes = include_attributes
        # attributes that exist in the annotation but does not want to be included in openlabel
        self._exclude_attributes = set(exclude_attributes or [])
        self._frame_match_tolerance_ms = frame_match_tolerance_ms

    def convert(self) -> None:
        start = time.time()

        for seq_path, staging_dir in iter_scene_pairs(
            Path(self._input_base), Path(self._output_base)
        ):
            if not (staging_dir / "lidar").is_dir():
                logger.warning(
                    f"No Kognic staging directory with lidar data at {staging_dir}; "
                    f"run convert_t4_to_kognic first. Skipping {seq_path}"
                )
                continue
            logger.info(f"[BEGIN] {seq_path} -> {staging_dir}")
            self._convert_one_scene(seq_path, staging_dir)
            logger.info(f"[DONE]  {seq_path}")

        logger.info(f"Elapsed: {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Scene conversion
    # ------------------------------------------------------------------

    def _convert_one_scene(self, seq_path: Path, staging_dir: Path) -> None:
        tables = {
            name: self._load_annotation(seq_path, f"{name}.json")
            for name in (
                "sensor",
                "calibrated_sensor",
                "sample",
                "sample_data",
                "sample_annotation",
                "instance",
                "category",
                "attribute",
                "ego_pose",
            )
        }

        lidarseg_records = self._load_annotation_optional(seq_path, "lidarseg.json")

        if not tables["sample_annotation"] and not lidarseg_records:
            logger.warning(f"No annotations in {seq_path}; skipping")
            return

        concat_records = self._collect_concat_records(tables)
        if not concat_records:
            logger.warning(f"No {LIDAR_CONCAT_CHANNEL} sample_data in {seq_path}; skipping")
            return

        anchor_ts_ns, relative_ms, anchor_stream = self._load_staging_frames(staging_dir)
        concat_to_frame = self._map_concat_to_frames(concat_records, anchor_ts_ns)
        stream_name = self._lidar_stream or anchor_stream

        self._write_keyframes(staging_dir, concat_records, concat_to_frame, len(anchor_ts_ns))

        if tables["sample_annotation"]:
            self._write_cuboid_pre_annotation(
                seq_path, staging_dir, tables, concat_records, concat_to_frame,
                relative_ms, stream_name,
            )
        if lidarseg_records:
            self._write_semseg_pre_annotation(
                seq_path, staging_dir, tables, lidarseg_records, concat_records,
                concat_to_frame, relative_ms, stream_name,
            )

    def _write_cuboid_pre_annotation(
        self,
        seq_path: Path,
        staging_dir: Path,
        tables: dict,
        concat_records: List[dict],
        concat_to_frame: Dict[int, int],
        relative_ms: List[int],
        stream_name: str,
    ) -> None:
        """Export ``sample_annotation.json`` boxes as ``cuboid_pre_annotation.json``."""
        concat_idx_by_sample = {
            record["sample_token"]: idx
            for idx, record in enumerate(concat_records)
            if record.get("is_key_frame")
        }

        categories = {c["token"]: c["name"] for c in tables["category"]}
        instances = {i["token"]: i for i in tables["instance"]}
        attributes = {a["token"]: a["name"] for a in tables["attribute"]}
        ego_pose_by_token = {ep["token"]: ep for ep in tables["ego_pose"]}

        annotations_by_sample: Dict[str, List[dict]] = {}
        for annotation in tables["sample_annotation"]:
            annotations_by_sample.setdefault(annotation["sample_token"], []).append(annotation)

        objects: Dict[str, openlabel.Object] = {}
        frames: Dict[str, openlabel.Frame] = {}
        skipped = 0

        for sample in sorted(tables["sample"], key=lambda s: s["timestamp"]):
            sample_annotations = annotations_by_sample.get(sample["token"])
            if not sample_annotations:
                continue

            concat_idx = concat_idx_by_sample.get(sample["token"])
            frame_idx = concat_to_frame.get(concat_idx) if concat_idx is not None else None
            if frame_idx is None:
                logger.warning(
                    f"Sample {sample['token']} could not be matched to a staging frame; "
                    f"dropping {len(sample_annotations)} annotation(s)"
                )
                skipped += len(sample_annotations)
                continue

            ego_pose = ego_pose_by_token[concat_records[concat_idx]["ego_pose_token"]]
            frame_objects: Dict[str, openlabel.Objects] = {}

            for annotation in sample_annotations:
                instance = instances[annotation["instance_token"]]
                category_name = categories.get(instance["category_token"], "unknown")
                object_uuid = _token_to_uuid(annotation["instance_token"])

                objects.setdefault(
                    object_uuid,
                    openlabel.Object(
                        name=_object_name(instance, object_uuid),
                        type=self._category_map.get(category_name, category_name),
                    ),
                )

                # The cuboid geometry only carries the ``stream`` marker that ties
                # it to the LiDAR sensor frame. Class properties (vehicle_state,
                # occlusion_state, ...) must live on the object, not on the
                # geometry: Kognic rejects source-specific properties on 3D
                # geometry ("3D geometry may not use source specific properties").
                class_properties = []
                if self._include_attributes:
                    class_properties = [
                        attribute_to_text(attributes[token])
                        for token in annotation.get("attribute_tokens", [])
                        if token in attributes
                        and attributes[token].rpartition(".")[0] not in self._exclude_attributes
                    ]

                frame_objects[object_uuid] = openlabel.Objects(
                    object_data=openlabel.ObjectData(
                        cuboid=[
                            openlabel.Cuboid(
                                name=f"cuboid-{object_uuid[:8]}",
                                val=t4_box_to_cuboid_val(annotation, ego_pose),
                                attributes=openlabel.Attributes(
                                    text=[openlabel.Text(name="stream", val=stream_name)]
                                ),
                            )
                        ],
                        text=class_properties or None,
                    )
                )

            frames[str(frame_idx)] = openlabel.Frame(
                frame_properties=openlabel.FrameProperties(
                    timestamp=relative_ms[frame_idx],
                    streams={stream_name: {}},
                    external_id=str(frame_idx),
                ),
                objects=frame_objects,
            )

        if not frames:
            logger.warning(f"No annotation could be placed on a staging frame for {seq_path}")
            return

        frame_indices = sorted(int(idx) for idx in frames)
        annotation = openlabel.OpenLabelAnnotation(
            openlabel=openlabel.Openlabel(
                metadata=openlabel.Metadata(
                    schema_version=openlabel.SchemaVersion.field_1_0_0,
                    name=staging_dir.name,
                ),
                objects=objects,
                frames=frames,
                frame_intervals=[
                    openlabel.FrameInterval(
                        frame_start=frame_indices[0], frame_end=frame_indices[-1]
                    )
                ],
                streams=self._build_streams(staging_dir),
            )
        )

        out_path = staging_dir / "cuboid_pre_annotation.json"
        with open(out_path, "w") as f:
            json.dump(annotation.model_dump(mode="json", exclude_none=True), f, indent=2)

        total = sum(len(frame.objects) for frame in frames.values())
        logger.info(
            f"{out_path}: {len(objects)} objects, {total} cuboids over "
            f"{len(frames)} frames (skipped {skipped})"
        )

    # ------------------------------------------------------------------
    # Point-cloud segmentation (T4 lidarseg -> semseg pre-annotation)
    # ------------------------------------------------------------------

    def _write_semseg_pre_annotation(
        self,
        seq_path: Path,
        staging_dir: Path,
        tables: dict,
        lidarseg_records: List[dict],
        concat_records: List[dict],
        concat_to_frame: Dict[int, int],
        relative_ms: List[int],
        stream_name: str,
    ) -> None:
        """Export ``lidarseg.json`` labels as ``semseg_pre_annotation.json``.

        The exact inverse of ``OpenLabelToT4Converter._convert_segmentation``:
        each frame's per-point uint8 class ids are run-length encoded as
        ``#<count>V<class_id>`` on ``binary`` object data entries, each bound
        to its lidar stream via a ``stream`` text attribute as Kognic's
        pre-annotation spec requires (``0`` = unlabelled), and the positive
        ``category.json`` ``index`` fields become the ontology
        ``classifications`` (index 0/background is implicit on the way back).

        Kognic expects one RLE per frame addressing its *merged* point cloud
        via the special stream literal ``"lidar"`` (as its own semseg exports
        show). The merge follows point-cloud registration order at scene
        creation, i.e. ``KognicDatasetUploader._collect_sensor_files``'s
        alphabetical sensor order — so when LIDAR_CONCAT_INFO is available the
        labels are re-sliced out of LIDAR_CONCAT order with the same
        ``idx_begin``/``length`` layout that ``extract_pointclouds`` used to
        stage the per-sensor CSVs, and concatenated alphabetically by stream.
        Without it the scene was staged as a single fused stream and the label
        array is used as-is.
        """
        classifications = self._segmentation_classifications(tables["category"])
        if not classifications:
            logger.warning(
                f"{seq_path}: lidarseg.json present but no category.json entry has a "
                "positive index; skipping the semseg pre-annotation"
            )
            return

        concat_idx_by_sd_token = {
            record["token"]: idx for idx, record in enumerate(concat_records)
        }
        channel_by_sensor_token = {
            sensor["token"]: sensor["channel"] for sensor in tables["sensor"]
        }
        staged_streams = {
            path.name for path in (staging_dir / "lidar").iterdir() if path.is_dir()
        }
        # One segmentation object shared by all frames; deterministic per scene.
        object_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{staging_dir.name}/lidarseg"))

        frames: Dict[str, openlabel.Frame] = {}
        skipped = 0
        for record in lidarseg_records:
            concat_idx = concat_idx_by_sd_token.get(record["sample_data_token"])
            frame_idx = concat_to_frame.get(concat_idx) if concat_idx is not None else None
            if frame_idx is None:
                logger.warning(
                    f"lidarseg record {record['token']} could not be matched to a "
                    f"staging frame; dropping its segmentation"
                )
                skipped += 1
                continue

            labels = np.fromfile(seq_path / record["filename"], dtype=np.uint8)
            num_points = lidar_point_count(seq_path / concat_records[concat_idx]["filename"])
            if num_points is not None and labels.size != num_points:
                logger.warning(
                    f"{record['filename']}: {labels.size} labels vs {num_points} points in "
                    f"{concat_records[concat_idx]['filename']}; dropping this frame"
                )
                skipped += 1
                continue
            unknown = set(np.unique(labels).tolist()) - {0} - set(classifications)
            if unknown:
                logger.warning(
                    f"{record['filename']}: label value(s) {sorted(unknown)} have no "
                    f"category.json index; dropping this frame"
                )
                skipped += 1
                continue

            stream_labels = self._split_labels_by_stream(
                seq_path,
                concat_records[concat_idx],
                labels,
                channel_by_sensor_token,
                staged_streams,
                stream_name,
            )
            if not stream_labels:
                skipped += 1
                continue
            # Kognic merges the per-sensor clouds in registration order
            # (alphabetical, cf. KognicDatasetUploader._collect_sensor_files).
            merged_labels = np.concatenate(
                [
                    labels_slice
                    for _, labels_slice in sorted(stream_labels, key=lambda x: x[0])
                ]
            )

            frames[str(frame_idx)] = openlabel.Frame(
                frame_properties=openlabel.FrameProperties(
                    timestamp=relative_ms[frame_idx],
                    streams={stream_name: {}},
                    external_id=str(frame_idx),
                ),
                objects={
                    object_uuid: openlabel.Objects(
                        object_data=openlabel.ObjectData(
                            binary=[
                                openlabel.Binary(
                                    name="labels",
                                    encoding="rle",
                                    data_type="",
                                    val=_encode_rle_labels(merged_labels),
                                    attributes=openlabel.Attributes(
                                        text=[
                                            openlabel.Text(
                                                name="stream",
                                                val=MERGED_LIDAR_STREAM,
                                            )
                                        ]
                                    ),
                                )
                            ]
                        )
                    )
                },
            )

        if not frames:
            logger.warning(
                f"No lidarseg frame could be placed on a staging frame for {seq_path}"
            )
            return

        frame_indices = sorted(int(idx) for idx in frames)
        annotation = openlabel.OpenLabelAnnotation(
            openlabel=openlabel.Openlabel(
                metadata=openlabel.Metadata(
                    schema_version=openlabel.SchemaVersion.field_1_0_0,
                    name=staging_dir.name,
                    annotation_type="semseg",
                ),
                ontologies={
                    "0": openlabel.OntologyItem(
                        uri="",
                        classifications={
                            str(index): name for index, name in sorted(classifications.items())
                        },
                    )
                },
                objects={
                    object_uuid: openlabel.Object(
                        name="lidarseg", type="3DPointCloudSegmentation"
                    )
                },
                frames=frames,
                frame_intervals=[
                    openlabel.FrameInterval(
                        frame_start=frame_indices[0], frame_end=frame_indices[-1]
                    )
                ],
                streams=self._build_streams(staging_dir),
            )
        )

        out_path = staging_dir / "semseg_pre_annotation.json"
        with open(out_path, "w") as f:
            json.dump(annotation.model_dump(mode="json", exclude_none=True), f, indent=2)

        logger.info(
            f"{out_path}: {len(classifications)} classes over {len(frames)} frames "
            f"(skipped {skipped})"
        )

    def _split_labels_by_stream(
        self,
        seq_path: Path,
        concat_record: dict,
        labels: np.ndarray,
        channel_by_sensor_token: Dict[str, str],
        staged_streams: set,
        stream_name: str,
    ) -> List[Tuple[str, np.ndarray]]:
        """Slice concat-order *labels* per staged lidar stream via LIDAR_CONCAT_INFO.

        Returns ``[(stream, labels_slice), ...]`` in ``idx_begin`` order, or the
        whole array on *stream_name* when the scene has no LIDAR_CONCAT_INFO
        (fused single-stream export). An empty list means the frame must be
        dropped because the info layout does not cover the labels.
        """
        info_filename = concat_record.get("info_filename")
        if not info_filename:
            return [(stream_name, labels)]

        with open(seq_path / info_filename) as f:
            sources = json.load(f)["sources"]

        total = sum(int(source["length"]) for source in sources)
        if total != labels.size:
            logger.warning(
                f"{info_filename}: sources cover {total} points but there are "
                f"{labels.size} labels; dropping this frame"
            )
            return []

        stream_labels: List[Tuple[str, np.ndarray]] = []
        for source in sorted(sources, key=lambda source: int(source["idx_begin"])):
            length = int(source["length"])
            if length == 0:
                continue
            channel = channel_by_sensor_token.get(source["sensor_token"])
            if channel not in staged_streams:
                logger.warning(
                    f"{info_filename}: source lidar {channel or source['sensor_token']} "
                    f"is not a staged stream; dropping its {length} labels"
                )
                continue
            idx_begin = int(source["idx_begin"])
            stream_labels.append((channel, labels[idx_begin : idx_begin + length]))
        return stream_labels

    def _segmentation_classifications(self, categories: List[dict]) -> Dict[int, str]:
        """Ontology class id -> class name, from ``category.json`` ``index`` fields.

        Categories without a positive ``index`` are not paint classes: index 0
        is the implicit unlabelled/background class, and box-only categories
        carry no index at all.
        """
        classifications: Dict[int, str] = {}
        for category in categories:
            index = category.get("index")
            if not index or index < 0:
                continue
            name = category["name"]
            classifications[int(index)] = self._category_map.get(name, name)
        return classifications

    @staticmethod
    def _write_keyframes(
        staging_dir: Path,
        concat_records: List[dict],
        concat_to_frame: Dict[int, int],
        frame_count: int,
    ) -> None:
        """Write the T4 keyframe positions for the uploader to ``keyframes.json``.

        The staging frame indices of the ``sample_data`` records with
        ``is_key_frame`` set. The uploader marks exactly those frames
        ``annotate=True`` (instead of walking a fixed ``target_hz`` grid), so
        the annotatable frames always coincide with the pre-annotation frames
        even when the source keyframe cadence skips a sweep. ``frame_count``
        lets the uploader detect a stale file after the staging data changed.

        ``T4ToKognicConverter`` writes the same file for every scene (including
        non-annotated ones); this refresh keeps re-running only the
        pre-annotation step on an older staging dir sufficient.
        """
        keyframe_indices = sorted(
            concat_to_frame[idx]
            for idx, record in enumerate(concat_records)
            if record.get("is_key_frame") and idx in concat_to_frame
        )
        out_path = staging_dir / "keyframes.json"
        with open(out_path, "w") as f:
            json.dump({"frame_count": frame_count, "keyframe_indices": keyframe_indices}, f)
        logger.info(f"{out_path}: {len(keyframe_indices)} keyframes over {frame_count} frames")

    @staticmethod
    def _load_annotation(seq_path: Path, name: str) -> list:
        with open(seq_path / "annotation" / name) as f:
            return json.load(f)

    @staticmethod
    def _load_annotation_optional(seq_path: Path, name: str) -> list:
        """Like ``_load_annotation``, but an absent table reads as empty."""
        if not (seq_path / "annotation" / name).exists():
            return []
        return T4ToOpenLabelConverter._load_annotation(seq_path, name)

    @staticmethod
    def _collect_concat_records(tables: dict) -> List[dict]:
        channel_by_calib = channel_by_calibrated_sensor(
            tables["sensor"], tables["calibrated_sensor"]
        )
        return records_for_channel(tables["sample_data"], channel_by_calib, LIDAR_CONCAT_CHANNEL)

    @staticmethod
    def _load_staging_frames(staging_dir: Path) -> Tuple[List[int], List[int], str]:
        """Enumerate scene frames exactly like ``KognicDatasetUploader.iterate_frames``."""
        lidar_root = staging_dir / "lidar"
        sensor_names = sorted(
            (p.name for p in lidar_root.iterdir() if p.is_dir()),
            key=lambda name: _sensor_sort_key(name, PREFERRED_LIDAR_SENSORS),
        )
        anchor = sensor_names[0]
        files = sorted((lidar_root / anchor).glob("*.csv"), key=_sort_key)
        if not files:
            raise FileNotFoundError(f"No lidar CSVs found in {lidar_root / anchor}")

        timestamps_ns = [int(path.stem) for path in files]
        reference = timestamps_ns[0]
        relative_ms = [int((ts - reference) / 1e6) for ts in timestamps_ns]
        return timestamps_ns, relative_ms, anchor

    def _map_concat_to_frames(
        self, concat_records: List[dict], anchor_ts_ns: List[int]
    ) -> Dict[int, int]:
        """Map LIDAR_CONCAT record index -> staging frame index.

        The staging frames were generated one per LIDAR_CONCAT record in
        timestamp order, so when the counts match the mapping is positional.
        Otherwise fall back to nearest-timestamp matching within tolerance
        (anchor per-source stamps are offset from the concat timestamp by a
        fraction of the sweep period).
        """
        if len(concat_records) == len(anchor_ts_ns):
            return {idx: idx for idx in range(len(concat_records))}

        logger.warning(
            f"Staging frame count ({len(anchor_ts_ns)}) != LIDAR_CONCAT record count "
            f"({len(concat_records)}); falling back to nearest-timestamp matching"
        )
        anchor = np.asarray(anchor_ts_ns, dtype=np.int64)
        mapping: Dict[int, int] = {}
        for idx, record in enumerate(concat_records):
            ts_ns = int(record["timestamp"]) * 1000
            frame_idx = int(np.argmin(np.abs(anchor - ts_ns)))
            diff_ms = abs(int(anchor[frame_idx]) - ts_ns) / 1e6
            if diff_ms <= self._frame_match_tolerance_ms:
                mapping[idx] = frame_idx
        return mapping

    @staticmethod
    def _build_streams(staging_dir: Path) -> Dict[str, openlabel.Stream]:
        streams: Dict[str, openlabel.Stream] = {}
        for path in sorted((staging_dir / "lidar").iterdir()):
            if path.is_dir():
                streams[path.name] = openlabel.Stream(type=openlabel.StreamTypes.lidar)
        cameras_root = staging_dir / "cameras"
        if cameras_root.is_dir():
            for path in sorted(cameras_root.iterdir()):
                if path.is_dir():
                    streams[path.name] = openlabel.Stream(type=openlabel.StreamTypes.camera)
        return streams


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode_rle_labels(labels: np.ndarray) -> str:
    """Run-length encode per-point labels as ``#<count>V<class_id>``.

    The inverse of ``openlabel_to_t4_converter._decode_rle_labels``. Every run
    is encoded explicitly, including unlabelled (0) runs, so the decoded label
    count always equals the point count.
    """
    if labels.size == 0:
        return ""
    boundaries = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [labels.size]))
    return "".join(
        f"#{int(end - start)}V{int(labels[start])}" for start, end in zip(starts, ends)
    )


def _token_to_uuid(token: str) -> str:
    try:
        return str(uuid.UUID(hex=token))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, token))


def _object_name(instance: dict, object_uuid: str) -> str:
    instance_name = instance.get("instance_name", "")
    if instance_name:
        return instance_name.split("::")[-1]
    return object_uuid
