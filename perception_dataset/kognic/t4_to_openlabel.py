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
from perception_dataset.utils.t4_tables import (
    channel_by_calibrated_sensor,
    records_for_channel,
)
from perception_dataset.kognic.t4_to_kognic_converter import T4ToKognicConverter
from perception_dataset.kognic.upload_dataset import _sensor_sort_key, _sort_key
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class T4ToOpenLabelConverter(AbstractConverter[None]):
    """Convert T4 3D box annotations to a Kognic OpenLABEL pre-annotation.

    For every annotated T4 sequence under ``input_base``, reads
    ``annotation/sample_annotation.json`` (and its companion tables) and
    writes a ``pre_annotation.json`` into the matching Kognic staging
    directory under ``output_base``, as previously produced by
    ``T4ToKognicConverter``::

        <output_base>/<scene>/
            calibration.json
            ego_poses.json
            pre_annotation.json     <- added by this converter
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
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        lidar_stream: str = "",
        category_map: Optional[Dict[str, str]] = None,
        include_attributes: bool = False,
        frame_match_tolerance_ms: float = 50.0,
    ):
        super().__init__(input_base, output_base)
        self._lidar_stream = lidar_stream
        self._category_map = category_map or {}
        self._include_attributes = include_attributes
        self._frame_match_tolerance_ms = frame_match_tolerance_ms

    def convert(self) -> None:
        start = time.time()

        for seq_path, staging_dir in self._iter_scene_pairs():
            if not (staging_dir / "lidar").is_dir():
                logger.warning(
                    f"No Kognic staging directory with lidar data at {staging_dir}; "
                    f"run convert_t4_to_kognic first. Skipping {seq_path}"
                )
                continue
            logger.info(f"[BEGIN] {seq_path} -> {staging_dir / 'pre_annotation.json'}")
            self._convert_one_scene(seq_path, staging_dir)
            logger.info(f"[DONE]  {seq_path}")

        logger.info(f"Elapsed: {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Sequence discovery (mirrors T4ToKognicConverter so names line up)
    # ------------------------------------------------------------------

    def _iter_scene_pairs(self) -> List[Tuple[Path, Path]]:
        input_base = Path(self._input_base)
        output_base = Path(self._output_base)

        if T4ToKognicConverter._is_sequence_root(input_base):
            return [(input_base, output_base / input_base.name)]

        pairs: List[Tuple[Path, Path]] = []
        for item in sorted(p for p in input_base.iterdir() if p.is_dir()):
            if item.name == "extracted_data":
                continue

            seq_roots = self._find_sequence_roots(item)
            if not seq_roots:
                logger.warning(f"No T4 sequence root found under {item}; skipping")
                continue

            if len(seq_roots) == 1:
                pairs.append((seq_roots[0], output_base / item.name))
            else:
                for seq_root in seq_roots:
                    pairs.append((seq_root, output_base / item.name / seq_root.name))

        return pairs

    @staticmethod
    def _find_sequence_roots(root: Path) -> List[Path]:
        if T4ToKognicConverter._is_sequence_root(root):
            return [root]

        return sorted(
            path
            for path in root.rglob("*")
            if T4ToKognicConverter._is_sequence_root(path) and "extracted_data" not in path.parts
        )

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

        if not tables["sample_annotation"]:
            logger.warning(f"No annotations in {seq_path}; skipping")
            return

        concat_records = self._collect_concat_records(tables)
        if not concat_records:
            logger.warning(f"No {LIDAR_CONCAT_CHANNEL} sample_data in {seq_path}; skipping")
            return

        anchor_ts_ns, relative_ms, anchor_stream = self._load_staging_frames(staging_dir)
        concat_to_frame = self._map_concat_to_frames(concat_records, anchor_ts_ns)
        stream_name = self._lidar_stream or anchor_stream

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

        out_path = staging_dir / "pre_annotation.json"
        with open(out_path, "w") as f:
            json.dump(annotation.model_dump(mode="json", exclude_none=True), f, indent=2)

        total = sum(len(frame.objects) for frame in frames.values())
        logger.info(
            f"{out_path}: {len(objects)} objects, {total} cuboids over "
            f"{len(frames)} frames (skipped {skipped})"
        )

    @staticmethod
    def _load_annotation(seq_path: Path, name: str) -> list:
        with open(seq_path / "annotation" / name) as f:
            return json.load(f)

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
