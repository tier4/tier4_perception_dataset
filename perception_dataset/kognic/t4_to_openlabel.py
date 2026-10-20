"""Convert T4 annotation tables to Kognic OpenLABEL pre-annotations."""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import uuid

from kognic.openlabel.models import models as openlabel
import numpy as np
from t4_devkit import Tier4
from t4_devkit.common.serialize import serialize_dataclass
from t4_devkit.schema.tables import Instance, SampleAnnotation, SampleData

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import LIDAR_CONCAT_CHANNEL
from perception_dataset.kognic.openlabel import attribute_to_text, t4_box_to_cuboid_val
from perception_dataset.kognic.openlabel.openlabel_attributes import (
    choose_static_value,
    resolve_multi_value_attribute,
)
from perception_dataset.kognic.sequence_artifact import (
    SEQUENCE_ARTIFACT_FILENAME,
    load_sequence_artifact,
)
from perception_dataset.kognic.utils import iter_scene_pairs
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


class T4ToOpenLabelConverter(AbstractConverter[None]):
    """Convert T4 3D box annotations to a Kognic OpenLABEL pre-annotation.

    For every annotated T4 sequence under ``input_base``, reads
    ``annotation/sample_annotation.json`` (and its companion tables) and
    writes a ``cuboid_pre_annotation.json`` into the matching Kognic staging
    directory under ``output_base``, as previously produced by
    ``T4ToKognicConverter``::

        <output_base>/<scene>/
            calibration.json
            ego_poses.json
            cuboid_pre_annotation.json     <- added by this converter
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
    - The converter-generated sequence artifact already marks source T4
      keyframes ``annotate=True``, so annotatable frames line up with the
      pre-annotation frames: Kognic only surfaces pre-annotations on
      annotatable frames.
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
        attribute_value_priority: Optional[Dict[str, List[str]]] = None,
        static_attributes: Optional[List[str]] = None,
        category_attributes: Optional[Dict[str, Dict[str, str]]] = None,
        task_definition_classes: Optional[List[str]] = None,
        attribute_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize the pre-annotation converter.

        Args:
            input_base (str): Input T4 dataset directory.
            output_base (str): Existing Kognic staging directory.
            lidar_stream (str): Explicit OpenLABEL lidar stream name.
            category_map (Optional[Dict[str, str]]): T4-to-Kognic category map.
            include_attributes (bool): Whether to include T4 box attributes.
            exclude_attributes (Optional[List[str]]): Attribute groups to omit, either for every
                class (``"extremities_state"``) or for one class only
                (``"pedestrian.extremities_state"``; the class may be the T4 category or the
                mapped Kognic class).
            frame_match_tolerance_ms (float): Maximum timestamp matching error.
            attribute_value_priority (Optional[Dict[str, List[str]]]): Per attribute group, the
                values in decreasing priority used to reduce a multi-valued T4 attribute
                (``extremities_state.protruding_object;none``) to the single value Kognic accepts.
            static_attributes (Optional[List[str]]): Attribute groups the Kognic task definition
                declares as static (one value per object, not per frame), with the same
                ``"group"`` / ``"<class>.group"`` scoping as ``exclude_attributes``. Their value is
                chosen across frames by ``attribute_value_priority`` (else the most frequent one)
                and emitted on the object instead of on every frame.
            category_attributes (Optional[Dict[str, Dict[str, str]]]): Extra properties per T4
                category, ``{"police_car": {"<property>": "police_car"}}``: keeps information a
                ``category_map`` entry folds away (e.g. the emergency-vehicle subtype) as a Kognic
                property of the object. Property names must exist in the task definition.
            task_definition_classes (Optional[List[str]]): Classes of the Kognic task definition.
                When given, a scene whose (mapped) categories include a class outside the list
                fails before anything is written: Kognic would reject the whole pre-annotation
                anyway, and the error names the category to add to ``category_map``.
            attribute_map (Optional[Dict[str, str]]): Attribute groups T4 and Kognic spell
                differently, T4 -> Kognic (``{"two_wheel_vehicle_state": "rider_state"}``);
                ``None`` uses the module default.
        """
        super().__init__(input_base, output_base)
        self._lidar_stream = lidar_stream
        self._category_map = category_map or {}
        self._include_attributes = include_attributes
        # attributes that exist in the annotation but does not want to be included in openlabel
        self._exclude_attributes = set(exclude_attributes or [])
        self._frame_match_tolerance_ms = frame_match_tolerance_ms
        self._attribute_value_priority = {
            group: list(values) for group, values in (attribute_value_priority or {}).items()
        }
        self._static_attributes = set(static_attributes or [])
        self._category_attributes = {
            category: dict(props) for category, props in (category_attributes or {}).items()
        }
        self._task_definition_classes = set(task_definition_classes or [])
        self._attribute_map = None if attribute_map is None else dict(attribute_map)

    @staticmethod
    def attribute_in_scoped_groups(
        attribute_name: str, category_name: str, mapped_category: str, scoped_groups: "set[str]"
    ) -> bool:
        """Whether a T4 attribute (``group.value``) falls under one of ``scoped_groups``.

        Entries are attribute groups (``extremities_state``, applied to every class) or class-scoped
        groups (``pedestrian.extremities_state``), where the class is the T4 category or the Kognic
        class it is mapped to. Used for ``exclude_attributes`` and ``static_attributes`` alike.
        """
        group = attribute_name.rpartition(".")[0]
        if not group:
            return False
        if group in scoped_groups:
            return True
        return f"{category_name}.{group}" in scoped_groups or f"{mapped_category}.{group}" in scoped_groups

    def is_attribute_excluded(self, attribute_name: str, category_name: str, mapped_category: str) -> bool:
        """Whether the attribute is left out of the pre-annotation (``exclude_attributes``)."""
        return self.attribute_in_scoped_groups(attribute_name, category_name, mapped_category, self._exclude_attributes)

    def is_attribute_static(self, attribute_name: str, category_name: str, mapped_category: str) -> bool:
        """Whether the attribute is a per-object (static) Kognic property (``static_attributes``)."""
        return self.attribute_in_scoped_groups(attribute_name, category_name, mapped_category, self._static_attributes)

    def _check_task_definition_classes(self, category_names, dataset_dir) -> None:
        """Fail fast when a (mapped) T4 category is not a class of the Kognic task definition."""
        if not self._task_definition_classes:
            return
        unknown = sorted(
            {
                f"{name} -> {self._category_map.get(name, name)}"
                for name in category_names
                if self._category_map.get(name, name) not in self._task_definition_classes
            }
        )
        if unknown:
            raise ValueError(
                f"{dataset_dir}: categories not in the Kognic task definition (add them to category_map): {unknown}"
            )

    def convert(self) -> None:
        """Convert annotations for every discovered T4 sequence.

        Returns:
            None
        """
        start = time.time()

        for seq_path, staging_dir in iter_scene_pairs(
            Path(self._input_base), Path(self._output_base)
        ):
            if not (staging_dir / SEQUENCE_ARTIFACT_FILENAME).is_file():
                logger.warning(
                    f"No {SEQUENCE_ARTIFACT_FILENAME} at {staging_dir}; "
                    f"run convert_t4_to_kognic first. Skipping {seq_path}"
                )
                continue
            logger.info(f"[BEGIN] {seq_path} -> {staging_dir / 'cuboid_pre_annotation.json'}")
            self._convert_one_scene(seq_path, staging_dir)
            logger.info(f"[DONE]  {seq_path}")

        logger.info(f"Elapsed: {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Scene conversion
    # ------------------------------------------------------------------

    def _convert_one_scene(self, seq_path: Path, staging_dir: Path) -> None:
        """Create an OpenLABEL pre-annotation for one scene.

        Args:
            seq_path (Path): Source T4 sequence root.
            staging_dir (Path): Destination Kognic staging directory.

        Returns:
            None
        """
        t4 = Tier4(data_root=str(seq_path), verbose=False)
        sample_annotations = t4.get_table("sample_annotation")

        if not sample_annotations:
            logger.warning(f"No annotations in {seq_path}; skipping")
            return

        concat_records = self._collect_concat_records(t4)
        if not concat_records:
            logger.warning(f"No {LIDAR_CONCAT_CHANNEL} sample_data in {seq_path}; skipping")
            return

        anchor_ts_ns, relative_ms, anchor_stream = self._load_staging_frames(staging_dir)
        concat_to_frame = self._map_concat_to_frames(concat_records, anchor_ts_ns)
        stream_name = self._lidar_stream or anchor_stream

        concat_idx_by_sample = {
            record.sample_token: idx
            for idx, record in enumerate(concat_records)
            if record.is_key_frame
        }

        categories = {category.token: category.name for category in t4.get_table("category")}
        self._check_task_definition_classes(categories.values(), seq_path)
        instances = {instance.token: instance for instance in t4.get_table("instance")}
        # some sources write the group with a trailing blank ("extremities_state .")
        attributes = {attribute.token: attribute.name.strip() for attribute in t4.get_table("attribute")}
        ego_pose_by_token = {pose.token: pose for pose in t4.get_table("ego_pose")}

        annotations_by_sample: Dict[str, List[SampleAnnotation]] = {}
        for annotation in sample_annotations:
            annotations_by_sample.setdefault(annotation.sample_token, []).append(annotation)

        objects: Dict[str, openlabel.Object] = {}
        frames: Dict[str, openlabel.Frame] = {}
        skipped = 0
        multi_value_resolved: Dict[str, str] = {}
        # object uuid -> attribute group -> values over frames (properties the task definition holds per object)
        static_values: Dict[str, Dict[str, List[str]]] = {}

        for sample in sorted(t4.get_table("sample"), key=lambda record: record.timestamp):
            annotations = annotations_by_sample.get(sample.token)
            if not annotations:
                continue

            concat_idx = concat_idx_by_sample.get(sample.token)
            frame_idx = concat_to_frame.get(concat_idx) if concat_idx is not None else None
            if frame_idx is None:
                logger.warning(
                    f"Sample {sample.token} could not be matched to a staging frame; "
                    f"dropping {len(annotations)} annotation(s)"
                )
                skipped += len(annotations)
                continue

            ego_pose = ego_pose_by_token[concat_records[concat_idx].ego_pose_token]
            frame_objects: Dict[str, openlabel.Objects] = {}

            for annotation in annotations:
                instance = instances[annotation.instance_token]
                category_name = categories.get(instance.category_token, "unknown")
                object_uuid = _token_to_uuid(annotation.instance_token)

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
                    mapped_category = self._category_map.get(category_name, category_name)
                    for token in annotation.attribute_tokens:
                        if token not in attributes or self.is_attribute_excluded(
                            attributes[token], category_name, mapped_category
                        ):
                            continue
                        attribute_name = resolve_multi_value_attribute(
                            attributes[token], self._attribute_value_priority
                        )
                        if attribute_name != attributes[token]:
                            multi_value_resolved[attributes[token]] = attribute_name
                        if self.is_attribute_static(attribute_name, category_name, mapped_category):
                            group, _, value = attribute_name.rpartition(".")
                            static_values.setdefault(object_uuid, {}).setdefault(group, []).append(value)
                            continue
                        class_properties.append(attribute_to_text(attribute_name, self._attribute_map))
                for name, value in self._category_attributes.get(category_name, {}).items():
                    class_properties.append(openlabel.Text(name=name, val=str(value)))

                frame_objects[object_uuid] = openlabel.Objects(
                    object_data=openlabel.ObjectData(
                        cuboid=[
                            openlabel.Cuboid(
                                name=f"cuboid-{object_uuid[:8]}",
                                val=t4_box_to_cuboid_val(
                                    serialize_dataclass(annotation),
                                    serialize_dataclass(ego_pose),
                                ),
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

        for object_uuid, groups in static_values.items():
            texts = [
                attribute_to_text(
                    f"{group}.{choose_static_value(values, self._attribute_value_priority.get(group, []))}",
                    self._attribute_map,
                )
                for group, values in sorted(groups.items())
            ]
            current = objects[object_uuid]
            objects[object_uuid] = openlabel.Object(
                name=current.name, type=current.type, object_data=openlabel.ObjectData(text=texts)
            )

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

        for original, resolved in sorted(multi_value_resolved.items()):
            logger.info(f"{out_path.name}: multi-valued attribute {original!r} -> {resolved!r}")
        if static_values:
            groups = sorted({group for per_object in static_values.values() for group in per_object})
            logger.info(
                f"{out_path.name}: {len(static_values)} object(s) carry static (per-object) properties {groups}"
            )
        total = sum(len(frame.objects) for frame in frames.values())
        logger.info(
            f"{out_path}: {len(objects)} objects, {total} cuboids over "
            f"{len(frames)} frames (skipped {skipped})"
        )

    @staticmethod
    def _collect_concat_records(t4: Tier4) -> List[SampleData]:
        """Collect fused-lidar sample-data records in timestamp order.

        Args:
            t4 (Tier4): Loaded T4 dataset.

        Returns:
            List[SampleData]: Ordered ``LIDAR_CONCAT`` records.
        """
        return sorted(
            (
                record
                for record in t4.get_table("sample_data")
                if record.channel == LIDAR_CONCAT_CHANNEL
            ),
            key=lambda record: record.timestamp,
        )

    @staticmethod
    def _load_staging_frames(staging_dir: Path) -> Tuple[List[int], List[int], str]:
        """Load staging frame timing from the validated sequence artifact.

        Args:
            staging_dir (Path): Kognic staging directory.

        Returns:
            Tuple[List[int], List[int], str]: Absolute timestamps, relative
                timestamps in milliseconds, and anchor lidar channel.

        Raises:
            FileNotFoundError: If the artifact contains no lidar resources.
        """
        sequence = load_sequence_artifact(staging_dir)
        if not sequence.frames or not sequence.frames[0].point_clouds:
            raise FileNotFoundError(
                f"No lidar resources found in {staging_dir}"
            )
        anchor = sequence.frames[0].point_clouds[0].sensor_name
        timestamps_ns = []
        relative_ms = []
        for frame in sequence.frames:
            if frame.unix_timestamp is None:
                raise ValueError(f"Frame {frame.frame_id} is missing unix_timestamp")
            timestamps_ns.append(int(frame.unix_timestamp))
            relative_ms.append(int(frame.relative_timestamp))
        return timestamps_ns, relative_ms, anchor

    def _map_concat_to_frames(
        self, concat_records: List[SampleData], anchor_ts_ns: List[int]
    ) -> Dict[int, int]:
        """Map LIDAR_CONCAT record index -> staging frame index.

        The staging frames were generated one per LIDAR_CONCAT record in
        timestamp order, so when the counts match the mapping is positional.
        Otherwise fall back to nearest-timestamp matching within tolerance
        (anchor per-source stamps are offset from the concat timestamp by a
        fraction of the sweep period).

        Args:
            concat_records (List[SampleData]): Ordered fused-lidar sample data.
            anchor_ts_ns (List[int]): Staging anchor timestamps in nanoseconds.

        Returns:
            Dict[int, int]: Concat-record indices mapped to staging frames.
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
            ts_ns = int(record.timestamp) * 1000
            frame_idx = int(np.argmin(np.abs(anchor - ts_ns)))
            diff_ms = abs(int(anchor[frame_idx]) - ts_ns) / 1e6
            if diff_ms <= self._frame_match_tolerance_ms:
                # MEMO: If frame mapping stops being one-to-one, detect duplicate
                # assignments and include them in the skipped-frame tally.
                mapping[idx] = frame_idx
        return mapping

    @staticmethod
    def _build_streams(staging_dir: Path) -> Dict[str, openlabel.Stream]:
        """Build OpenLABEL stream declarations from staged sensors.

        Args:
            staging_dir (Path): Kognic staging directory.

        Returns:
            Dict[str, openlabel.Stream]: Stream declarations keyed by channel.
        """
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
    """Convert a T4 token to a stable UUID string.

    Args:
        token (str): T4 token, which may or may not be hexadecimal.

    Returns:
        str: Canonical UUID derived from the token.
    """
    try:
        return str(uuid.UUID(hex=token))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, token))


def _object_name(instance: Instance, object_uuid: str) -> str:
    """Choose an OpenLABEL object name for a T4 instance.

    Args:
        instance (Instance): T4 instance record.
        object_uuid (str): Fallback object UUID.

    Returns:
        str: Instance-name suffix when available, otherwise ``object_uuid``.
    """
    instance_name = instance.instance_name
    if instance_name:
        return instance_name.split("::")[-1]
    return object_uuid
