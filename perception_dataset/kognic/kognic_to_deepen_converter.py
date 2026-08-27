"""Convert Kognic OpenLABEL annotations directly to Deepen annotations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.kognic.openlabel import ROTATION_T4_TO_KOGNIC
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

_RLE_TOKEN = re.compile(r"#(\d+)V(\d+)")
_ATTRIBUTE_TYPES = ("text", "num", "boolean")
_RESERVED_ATTRIBUTES = {
    "classification_id",
    "interpolated",
    "kognic geometry confidence",
    "kognic locked geometries",
    "kognic stationary object",
    "stream",
}


@dataclass(frozen=True)
class KognicToDeepenConverterOutputItem:
    input_path: str
    bbox_annotation_path: Optional[str]
    lidarseg_annotation_path: Optional[str]


@dataclass(frozen=True)
class KognicToDeepenConverterOutput:
    items: List[KognicToDeepenConverterOutputItem]


class KognicToDeepenConverter(AbstractConverter[KognicToDeepenConverterOutput]):
    """Convert cuboids, image boxes, and point semantic labels from OpenLABEL."""

    def __init__(
        self,
        input_base: str,
        output_base: str,
        dataset_id: str = "",
        iso_rotated_cuboids: bool = False,
        category_map: Optional[Dict[str, str]] = None,
        sensor_map: Optional[Dict[str, str]] = None,
        lidar_streams: Optional[List[str]] = None,
        include_attributes: bool = True,
    ):
        super().__init__(input_base, output_base)
        self._dataset_id = dataset_id
        self._iso_rotated_cuboids = iso_rotated_cuboids
        self._category_map = category_map or {}
        self._sensor_map = sensor_map or {}
        self._lidar_streams = lidar_streams or []
        self._include_attributes = include_attributes

    def convert(self) -> KognicToDeepenConverterOutput:
        """Convert every JSON file in ``input_base``, or the file itself."""
        input_path = Path(self._input_base)
        output_base = Path(self._output_base)
        paths = (
            [input_path]
            if input_path.is_file()
            else sorted(
                path
                for path in input_path.rglob("*.json")
                if path.is_file() and not path.resolve().is_relative_to(output_base.resolve())
            )
        )
        if not paths:
            raise ValueError(f"No JSON annotations found under {input_path}")

        output_base.mkdir(parents=True, exist_ok=True)
        items = [self._convert_file(path, output_base) for path in paths]
        return KognicToDeepenConverterOutput(items=items)

    def _convert_file(
        self, input_path: Path, output_base: Path
    ) -> KognicToDeepenConverterOutputItem:
        with open(input_path) as f:
            document = json.load(f)
        openlabel = document.get("openlabel", document)
        if not isinstance(openlabel.get("frames"), dict):
            raise ValueError(f"{input_path} does not contain OpenLABEL frames")

        stem = input_path.stem
        labels = self._convert_boxes(openlabel)
        bbox_path: Optional[Path] = None
        if labels:
            bbox_path = output_base / f"{stem}.json"
            with open(bbox_path, "w") as f:
                json.dump({"labels": labels}, f, indent=4)

        lidarseg_path = self._convert_segmentation(openlabel, output_base, stem)
        if not labels and lidarseg_path is None:
            raise ValueError(f"{input_path} contains no supported Kognic annotations")

        logger.info(
            f"{input_path}: {len(labels)} box label(s)"
            + (", semantic segmentation converted" if lidarseg_path else "")
        )
        return KognicToDeepenConverterOutputItem(
            input_path=str(input_path),
            bbox_annotation_path=str(bbox_path) if bbox_path else None,
            lidarseg_annotation_path=str(lidarseg_path) if lidarseg_path else None,
        )

    def _convert_boxes(self, openlabel: Dict[str, Any]) -> List[Dict[str, Any]]:
        objects = openlabel.get("objects", {})
        instance_ids = {uid: index for index, uid in enumerate(objects, start=1)}
        labels: List[Dict[str, Any]] = []

        for frame_key, frame in _sorted_frames(openlabel):
            file_id = _deepen_file_id(frame_key, frame)
            for object_uid, frame_object in frame.get("objects", {}).items():
                object_info = objects.get(object_uid, {})
                category = self._category_map.get(
                    object_info.get("type", "unknown"), object_info.get("type", "unknown")
                )
                object_data = frame_object.get("object_data", {})
                attributes = (
                    _collect_attributes(
                        object_info.get("object_data", {}),
                        object_data,
                    )
                    if self._include_attributes
                    else {}
                )
                label_id = (
                    f"{category}:{instance_ids.setdefault(object_uid, len(instance_ids) + 1)}"
                )

                for cuboid in object_data.get("cuboid", []):
                    geometry_attributes = (
                        attributes | _collect_attributes(cuboid.get("attributes", {}))
                        if self._include_attributes
                        else {}
                    )
                    labels.append(
                        self._base_label(
                            file_id,
                            category,
                            label_id,
                            "lidar",
                            geometry_attributes,
                            "3d_bbox",
                        )
                        | {"three_d_bbox": self._convert_cuboid(cuboid)}
                    )

                for bbox in object_data.get("bbox", []):
                    stream = _geometry_stream(bbox)
                    if not stream:
                        raise ValueError(
                            f"Frame {frame_key} object {object_uid}: bbox has no stream attribute"
                        )
                    cx, cy, width, height = _require_values(bbox, "bbox", 4)
                    if width < 0 or height < 0:
                        raise ValueError("bbox width and height must be non-negative")
                    box = [
                        cx - width / 2.0,
                        cy - height / 2.0,
                        width,
                        height,
                    ]
                    geometry_attributes = (
                        attributes | _collect_attributes(bbox.get("attributes", {}))
                        if self._include_attributes
                        else {}
                    )
                    labels.append(
                        self._base_label(
                            file_id,
                            category,
                            label_id,
                            stream,
                            geometry_attributes,
                            "box",
                        )
                        | {"box": box}
                    )
        return labels

    def _base_label(
        self,
        file_id: str,
        category: str,
        label_id: str,
        stream: str,
        attributes: Dict[str, Any],
        label_type: str,
    ) -> Dict[str, Any]:
        return {
            "attributes": attributes,
            "attributes_source": {},
            "create_time_millis": "null",
            "update_time_millis": "null",
            "dataset_id": self._dataset_id,
            "labeller_email": "default@tier4.jp",
            "user_id": "default@tier4.jp",
            "version": "null",
            "label_set_id": "default",
            "stage_id": "Labelling",
            "file_id": file_id,
            "label_category_id": category,
            "label_id": label_id,
            "sensor_id": self._sensor_map.get(stream, stream),
            "label_type": label_type,
        }

    def _convert_cuboid(self, cuboid: Dict[str, Any]) -> Dict[str, Any]:
        x, y, z, qx, qy, qz, qw, sx, sy, sz = _require_values(cuboid, "cuboid", 10)
        if min(sx, sy, sz) < 0:
            raise ValueError("cuboid dimensions must be non-negative")
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        if not self._iso_rotated_cuboids:
            rotation = rotation * ROTATION_T4_TO_KOGNIC.inv()
        qx, qy, qz, qw = rotation.as_quat()
        return {
            "cx": x,
            "cy": y,
            "cz": z,
            "h": sz,
            "l": sy,
            "w": sx,
            "quaternion": {"x": qx, "y": qy, "z": qz, "w": qw},
        }

    def _convert_segmentation(
        self, openlabel: Dict[str, Any], output_base: Path, stem: str
    ) -> Optional[Path]:
        frame_rles = [
            (frame_key, frame, _frame_segmentation_rles(frame))
            for frame_key, frame in _sorted_frames(openlabel)
        ]
        frame_rles = [entry for entry in frame_rles if entry[2]]
        if not frame_rles:
            return None

        ontology = _segmentation_ontology(openlabel)
        positive_classes = [(index, name) for index, name in sorted(ontology.items()) if index > 0]
        if not positive_classes:
            raise ValueError("Semantic segmentation has no positive ontology classes")
        if len(positive_classes) > np.iinfo(np.uint8).max:
            raise ValueError("Deepen paint-3D supports at most 255 semantic classes")
        paint_categories = [self._category_map.get(name, name) for _, name in positive_classes]
        ontology_to_deepen = {class_id: i for i, (class_id, _) in enumerate(positive_classes, 1)}
        raw_to_ontology = _segmentation_value_map(openlabel, ontology)

        lidarseg_dir = output_base / "lidarseg"
        lidarseg_dir.mkdir(parents=True, exist_ok=True)
        annotations = []
        for frame_key, frame, rles in frame_rles:
            labels = [
                _remap_segmentation_labels(
                    _decode_rle(rle, frame_key), raw_to_ontology, ontology_to_deepen
                )
                for _, rle in self._ordered_rles(rles, frame_key)
            ]
            merged = np.concatenate(labels).astype(np.uint8, copy=False)
            file_id = _deepen_file_id(frame_key, frame)
            output_name = f"{stem}_{file_id}.bin"
            merged.tofile(lidarseg_dir / output_name)
            annotations.append(
                {
                    "dataset_id": self._dataset_id,
                    "file_id": file_id,
                    "label_type": "3d_point",
                    "label_id": "none:1",
                    "label_category_id": "none",
                    "total_lidar_points": int(merged.size),
                    "sensor_id": "lidar",
                    "stage_id": "QA",
                    "paint_categories": paint_categories,
                    "lidarseg_anno_file": f"lidarseg/{output_name}",
                }
            )

        output_path = output_base / f"{stem}_lidarseg.json"
        with open(output_path, "w") as f:
            json.dump(annotations, f, indent=4)
        return output_path

    def _ordered_rles(
        self, rles: List[Tuple[Optional[str], str]], frame_key: str
    ) -> List[Tuple[Optional[str], str]]:
        streams = [stream for stream, _ in rles]
        duplicates = sorted({stream for stream in streams if streams.count(stream) > 1})
        if duplicates:
            raise ValueError(f"Frame {frame_key}: duplicate lidar streams {duplicates}")
        if not self._lidar_streams:
            return rles
        by_stream = dict(rles)
        missing = [stream for stream in self._lidar_streams if stream not in by_stream]
        unexpected = [stream for stream in by_stream if stream not in self._lidar_streams]
        if missing or unexpected:
            raise ValueError(
                f"Frame {frame_key}: lidar streams differ from configured lidar_streams; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return [(stream, by_stream[stream]) for stream in self._lidar_streams]


def _sorted_frames(openlabel: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    def key(item: Tuple[str, Any]) -> Tuple[int, Any]:
        try:
            return (0, int(item[0]))
        except ValueError:
            return (1, item[0])

    return sorted(openlabel.get("frames", {}).items(), key=key)


def _deepen_file_id(frame_key: str, frame: Dict[str, Any]) -> str:
    external_id = frame.get("frame_properties", {}).get("external_id", frame_key)
    value = str(external_id)
    return value if value.endswith(".pcd") else f"{value}.pcd"


def _geometry_stream(geometry: Dict[str, Any]) -> Optional[str]:
    return next(
        (
            str(attribute["val"])
            for attribute in geometry.get("attributes", {}).get("text", [])
            if attribute.get("name") == "stream"
        ),
        None,
    )


def _collect_attributes(*object_data_items: Dict[str, Any]) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    for object_data in object_data_items:
        for attribute_type in _ATTRIBUTE_TYPES:
            for attribute in object_data.get(attribute_type, []):
                name = attribute.get("name")
                if name and name not in _RESERVED_ATTRIBUTES:
                    attributes[name] = attribute.get("val")
    return attributes


def _require_values(geometry: Dict[str, Any], kind: str, count: int) -> List[float]:
    values = geometry.get("val")
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{kind} geometry must contain exactly {count} values")
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{kind} geometry values must be numeric")
    return [float(value) for value in values]


def _segmentation_ontology(openlabel: Dict[str, Any]) -> Dict[int, str]:
    ontology: Dict[int, str] = {}
    for entry in openlabel.get("ontologies", {}).values():
        for class_id, value in entry.get("classifications", {}).items():
            try:
                name = value.get("name") if isinstance(value, dict) else value
                ontology[int(class_id)] = str(name)
            except (TypeError, ValueError):
                continue
    return ontology


def _frame_segmentation_rles(frame: Dict[str, Any]) -> List[Tuple[Optional[str], str]]:
    output: List[Tuple[Optional[str], str]] = []
    for frame_object in frame.get("objects", {}).values():
        for binary in frame_object.get("object_data", {}).get("binary", []):
            if binary.get("name") == "labels" and binary.get("encoding") == "rle":
                value = binary.get("val")
                if not isinstance(value, str):
                    raise ValueError("Semantic segmentation RLE must be a string")
                output.append((_geometry_stream(binary), value))
    return output


def _decode_rle(value: str, frame_key: str) -> np.ndarray:
    matches = list(_RLE_TOKEN.finditer(value))
    if "".join(match.group(0) for match in matches) != value:
        raise ValueError(f"Frame {frame_key}: invalid Kognic RLE value")
    if not matches:
        raise ValueError(f"Frame {frame_key}: Kognic RLE value is empty")
    counts = np.fromiter((int(match.group(1)) for match in matches), dtype=np.int64)
    classes = np.fromiter((int(match.group(2)) for match in matches), dtype=np.int64)
    return np.repeat(classes, counts)


def _segmentation_value_map(openlabel: Dict[str, Any], ontology: Dict[int, str]) -> Dict[int, int]:
    name_to_id = {name: class_id for class_id, name in ontology.items()}
    value_map = {class_id: class_id for class_id in ontology}
    for obj in openlabel.get("objects", {}).values():
        classification_id = next(
            (
                attribute.get("val")
                for attribute in obj.get("object_data", {}).get("num", [])
                if attribute.get("name") == "classification_id"
            ),
            None,
        )
        class_id = name_to_id.get(obj.get("type"))
        if classification_id is not None and class_id is not None:
            value_map[int(classification_id)] = class_id
    return value_map


def _remap_segmentation_labels(
    labels: np.ndarray,
    raw_to_ontology: Dict[int, int],
    ontology_to_deepen: Dict[int, int],
) -> np.ndarray:
    unknown = sorted(set(np.unique(labels).tolist()) - set(raw_to_ontology) - {0})
    if unknown:
        raise ValueError(f"Semantic labels have no ontology/object mapping: {unknown[:5]}")
    output = np.zeros(labels.shape, dtype=np.uint8)
    for raw_value in np.unique(labels):
        if raw_value == 0:
            continue
        ontology_id = raw_to_ontology.get(int(raw_value))
        if ontology_id in ontology_to_deepen:
            output[labels == raw_value] = ontology_to_deepen[ontology_id]
    return output
