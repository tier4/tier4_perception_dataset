from datetime import datetime, timezone
import json
import logging
import math
import os.path as osp
from pathlib import Path
import shutil
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from perception_dataset.fastlabel_to_t4.fastlabel_2d_to_t4_converter import (
    FastLabel2dToT4Converter,
    _convert_polygon_to_bbox,
)
from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


SUPPORTED_SEMANTIC_LABELS: Set[str] = {
    "traffic_light",
    "traffic_light_back",
    "crosswalk_light_back",
    "crosswalk_red",
    "crosswalk_green",
    "crosswalk_unknown",
}
DIRECT_SEMANTIC_LABELS: Set[str] = SUPPORTED_SEMANTIC_LABELS - {"traffic_light"}

FRONT_BULB_LABELS: Set[str] = {"red", "green", "yellow"}
BACK_BULB_LABELS: Set[str] = {"backside", "head"}
SUPPORTED_BULB_LABELS: Set[str] = FRONT_BULB_LABELS | BACK_BULB_LABELS
CROSSWALK_BULB_TYPES: Set[str] = {"pedestrian", "cross"}
DEFAULT_VISIBILITY_LEVEL = "none"
VEHICLE_ARROW_DIRECTION_ORDER: Tuple[str, ...] = ("left", "straight", "right", "up_left", "up_right")
LEGACY_RED_CLASS_BY_GREEN_ARROWS: Dict[frozenset[str], str] = {
    frozenset(): "red",
    frozenset({"left"}): "left,red",
    frozenset({"left", "straight"}): "left,red,straight",
    frozenset({"right"}): "red,right",
    frozenset({"straight"}): "red,straight",
    frozenset({"right", "straight"}): "red,right,straight",
    frozenset({"up_left"}): "red,up_left",
    frozenset({"up_right"}): "red,up_right",
}


class FastLabel2dSemanticToT4TlrConverter(FastLabel2dToT4Converter):
    """Convert FastLabel bulb bbox + semantic traffic_light polygons into T4 2D annotations.

    Conversion rules:
    - Supports both legacy FastLabel frame JSON and tokenized t4 annotation tables
      (`object_ann/category/sample_data`).
    - Semantic traffic-light boxes can come from FastLabel segmentations (`traffic_light`) or
      t4 `object_ann` category `traffic_light`.
    - Bulb bboxes (`red`/`green`/`yellow`) are exported as `<color>_bulb`.
    - Parent semantic class is inferred from bulbs inside each `traffic_light` bbox:
      - legacy base classes are preserved (`red,right`, `left,red`, ...)
      - extra colored-arrow variants are generated (`green,right_red`, `red,right_yellow`, ...)
      - crosswalk classes (`crosswalk_red`, `crosswalk_green`, `crosswalk_unknown`) are preserved

    Corner cases:
    - no bulb in semantic bbox -> warning + `unknown`
    - unsupported/missing orientation for arrow -> warning + `unknown`
    - impossible combinations (e.g. multiple circle colors) -> warning + `unknown`
    """

    def __init__(
        self,
        input_base: str,
        output_base: str,
        input_anno_base: str,
        input_semantic_anno_base: str,
        overwrite_mode: bool,
        description: Dict[str, Dict[str, str]],
        input_bag_base: Optional[str],
        topic_list: Optional[Dict[str, List[str]]],
        arrow_angle_tolerance_deg: float = 10.0,
        output_dataset_version: Optional[str] = None,
    ):
        super().__init__(
            input_base=input_base,
            output_base=output_base,
            input_anno_base=input_anno_base,
            overwrite_mode=overwrite_mode,
            description=description,
            input_bag_base=input_bag_base,
            topic_list=topic_list,
            tlr_mode=True,
        )
        self._input_semantic_anno_base = Path(input_semantic_anno_base)
        self._failure_log_path = (
            self._output_base / "convert_fastlabel_2d_semantic_to_t4_tlr_failures.jsonl"
        )
        self._warning_log_path = (
            self._output_base / "convert_fastlabel_2d_semantic_to_t4_tlr_warnings.log"
        )
        self._arrow_angle_tolerance_deg = max(0.0, float(arrow_angle_tolerance_deg))
        self._output_dataset_version = (
            str(output_dataset_version) if output_dataset_version is not None else None
        )
        self._configure_warning_file_handler()

    def _configure_warning_file_handler(self) -> None:
        self._warning_log_path.parent.mkdir(parents=True, exist_ok=True)
        warning_log_path_str = str(self._warning_log_path.resolve())
        for handler in logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                continue
            if handler.level > logging.WARNING:
                continue
            if getattr(handler, "baseFilename", "") == warning_log_path_str:
                return

        warning_handler = logging.FileHandler(warning_log_path_str)
        warning_handler.setLevel(logging.WARNING)
        warning_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [file] %(filename)s "
                "[func] %(funcName)s [line] %(lineno)d [%(message)s]"
            )
        )
        logger.addHandler(warning_handler)

    def convert(self) -> None:
        t4_datasets = sorted([d.name for d in self._input_base.iterdir() if d.is_dir()])
        logger.info(f"Found {len(t4_datasets)} datasets to process")
        logger.info(f"Warning log file: {self._warning_log_path}")
        skipped_datasets: List[str] = []

        bulb_files_by_dataset = self._group_annotation_files_by_dataset_with_base(
            self._input_anno_base, t4_datasets
        )
        semantic_files_by_dataset = self._group_annotation_files_by_dataset_with_base(
            self._input_semantic_anno_base, t4_datasets
        )

        for t4dataset_name in t4_datasets:
            bulb_files = bulb_files_by_dataset.get(t4dataset_name, [])
            semantic_files = semantic_files_by_dataset.get(t4dataset_name, [])
            if not bulb_files and not semantic_files:
                logger.warning(f"No annotation files found for {t4dataset_name}")
                continue
            logger.info(f"Processing dataset: {t4dataset_name}")

            input_dir = self._input_base / t4dataset_name
            input_annotation_dir = input_dir / "annotation"
            if not osp.exists(input_annotation_dir):
                logger.warning(f"input_dir {input_dir} not exists.")
                continue

            output_dir = self._resolve_output_scene_dir(t4dataset_name)
            if self._input_bag_base is not None:
                input_bag_dir = Path(self._input_bag_base) / t4dataset_name

            is_dir_exist = False
            if osp.exists(output_dir):
                logger.warning(f"{output_dir} already exists.")
                is_dir_exist = True

            if self._overwrite_mode or not is_dir_exist:
                shutil.rmtree(output_dir, ignore_errors=True)
                self._copy_data(input_dir, output_dir)
                if self._input_bag_base is not None and not osp.exists(
                    osp.join(output_dir, "input_bag")
                ):
                    self._find_start_end_time(input_dir)
                    self._make_rosbag(str(input_bag_dir), str(output_dir))
                self._sanitize_autolabel_metadata_in_scene(output_dir, t4dataset_name)
            else:
                raise ValueError("If you want to overwrite files, use --overwrite option.")

            bulb_annotations = (
                self._load_annotation_jsons_for_dataset(bulb_files) if bulb_files else []
            )
            semantic_annotations = (
                self._load_annotation_jsons_for_dataset(semantic_files) if semantic_files else []
            )

            object_ann_bulbs, object_ann_semantic = self._extract_object_ann_frame_annotations(
                annotation_rows=[*bulb_annotations, *semantic_annotations],
                dataset_name=t4dataset_name,
            )
            if object_ann_bulbs:
                logger.info(
                    f"{t4dataset_name}: extracted {len(object_ann_bulbs)} object_ann bulb frames"
                )
                # Prefer object_ann bulbs when present to avoid duplicate bulbs coming from
                # legacy FastLabel + tokenized object_ann sources in the same dataset.
                bulb_annotations = object_ann_bulbs
            if object_ann_semantic:
                logger.info(
                    f"{t4dataset_name}: extracted {len(object_ann_semantic)} object_ann semantic frames"
                )
                # Keep direct semantic classes (crosswalk/back) from legacy sources but drop
                # generic traffic_light boxes to avoid duplicated parent boxes.
                semantic_annotations, removed_legacy_traffic_light = (
                    self._remove_generic_traffic_light_semantics(semantic_annotations)
                )
                if removed_legacy_traffic_light > 0:
                    logger.info(
                        f"{t4dataset_name}: removed {removed_legacy_traffic_light} legacy "
                        "generic traffic_light semantic boxes due to object_ann precedence"
                    )
                semantic_annotations.extend(object_ann_semantic)

            if not semantic_annotations:
                logger.warning(
                    f"{t4dataset_name}: no semantic traffic-light source found; "
                    "only bulb-level labels may be exported"
                )
            if not bulb_annotations:
                logger.warning(
                    f"{t4dataset_name}: no bulb source found; traffic_light boxes will map to unknown"
                )

            fl_annotations = self._format_semantic_and_bulb_annotations(
                bulb_annotations=bulb_annotations,
                semantic_annotations=semantic_annotations,
                dataset_name=t4dataset_name,
            )

            annotation_files_generator = AnnotationFilesGenerator(description=self._description)
            try:
                annotation_files_generator.convert_one_scene(
                    input_dir=output_dir,
                    output_dir=output_dir,
                    scene_anno_dict=fl_annotations,
                    dataset_name=t4dataset_name,
                )
            except Exception as e:
                traceback_text = traceback.format_exc()
                logger.exception(
                    f"{t4dataset_name}: failed to generate annotation tables. "
                    "Skipping this dataset."
                )
                self._append_failure_log(
                    dataset_name=t4dataset_name,
                    error=e,
                    stage="annotation_files_generator.convert_one_scene",
                    traceback_text=traceback_text,
                )
                shutil.rmtree(output_dir, ignore_errors=True)
                skipped_datasets.append(t4dataset_name)
                continue

        if skipped_datasets:
            logger.warning(
                f"Skipped {len(skipped_datasets)} datasets due to conversion errors. "
                f"See {self._failure_log_path}"
            )

    def _append_failure_log(
        self,
        dataset_name: str,
        error: Exception,
        stage: str,
        traceback_text: str,
    ) -> None:
        self._failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset_name,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback_text,
        }
        with open(self._failure_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _resolve_output_scene_dir(self, dataset_name: str) -> Path:
        if self._output_dataset_version is None:
            return self._output_base / dataset_name
        return self._output_base / dataset_name / self._output_dataset_version

    def _sanitize_autolabel_metadata_in_scene(self, scene_dir: Path, dataset_name: str) -> None:
        """Normalize legacy `autolabel_metadata` representation for t4-devkit compatibility."""
        annotation_dir = scene_dir / "annotation"
        if not annotation_dir.exists():
            return

        allowed_fields_by_file: Dict[str, Set[str]] = {
            "object_ann.json": {
                "token",
                "sample_data_token",
                "instance_token",
                "category_token",
                "attribute_tokens",
                "bbox",
                "mask",
                "orientation",
                "number",
                "automatic_annotation",
                "autolabel_metadata",
            },
            "surface_ann.json": {
                "token",
                "sample_data_token",
                "category_token",
                "mask",
                "automatic_annotation",
                "autolabel_metadata",
            },
            "sample_data.json": {
                "token",
                "sample_token",
                "ego_pose_token",
                "calibrated_sensor_token",
                "filename",
                "fileformat",
                "width",
                "height",
                "timestamp",
                "is_key_frame",
                "next",
                "prev",
                "is_valid",
                "info_filename",
                "autolabel_metadata",
            },
        }
        target_files = ("object_ann.json", "surface_ann.json", "sample_data.json")
        for filename in target_files:
            json_path = annotation_dir / filename
            if not json_path.exists():
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                logger.warning(f"{dataset_name}: failed to read {json_path} for metadata normalization")
                continue

            if not isinstance(records, list):
                continue

            changed = False
            removed_key_count = 0
            allowed_fields = allowed_fields_by_file.get(filename, set())
            for row in records:
                if not isinstance(row, dict):
                    continue

                if allowed_fields:
                    extra_keys = set(row.keys()) - allowed_fields
                    for key in extra_keys:
                        row.pop(key, None)
                    if extra_keys:
                        removed_key_count += len(extra_keys)
                        changed = True

                if "autolabel_metadata" in row:
                    if row.get("autolabel_metadata") is not None:
                        row["autolabel_metadata"] = None
                        changed = True

                if filename in {"object_ann.json", "surface_ann.json"}:
                    if row.get("automatic_annotation", False):
                        row["automatic_annotation"] = False
                        changed = True

            if not changed:
                continue

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=True)
            logger.info(
                f"{dataset_name}: normalized {json_path.name} for t4-devkit compatibility "
                f"(removed_extra_keys={removed_key_count})"
            )

    @staticmethod
    def _normalize_autolabel_metadata(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, list):
            return FastLabel2dSemanticToT4TlrConverter._normalize_autolabel_models(value)
        if isinstance(value, dict):
            # Common legacy shape: {"models": [ ... ]}
            return FastLabel2dSemanticToT4TlrConverter._normalize_autolabel_models(
                value.get("models")
            )
        return None

    @staticmethod
    def _normalize_autolabel_models(models: Any) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(models, list):
            return None

        normalized: List[Dict[str, Any]] = []
        for model in models:
            if not isinstance(model, dict):
                continue

            name = model.get("name")
            if name is None:
                continue

            raw_score = model.get("score")
            if raw_score is None:
                score = 1.0
            else:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = 1.0

            normalized.append({"name": str(name), "score": score})

        if not normalized:
            return None
        return normalized

    @staticmethod
    def _group_annotation_files_by_dataset_with_base(
        annotation_base: Path, t4_datasets: List[str]
    ) -> Dict[str, List[Path]]:
        anno_files_by_dataset: Dict[str, List[Path]] = {}
        all_anno_files = list(annotation_base.rglob("*.json"))

        for dataset_name in tqdm(t4_datasets):
            matched_files = [
                f for f in all_anno_files if dataset_name in f.name or dataset_name in f.parts
            ]
            if matched_files:
                anno_files_by_dataset[dataset_name] = matched_files
        return anno_files_by_dataset

    def _extract_object_ann_frame_annotations(
        self,
        annotation_rows: List[Dict[str, Any]],
        dataset_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build FastLabel-like frame records from tokenized t4 `annotation/*.json` tables."""
        category_token_to_name: Dict[str, str] = {}
        sample_data_token_to_name: Dict[str, str] = {}
        object_ann_rows_by_token: Dict[str, Dict[str, Any]] = {}

        for row in annotation_rows:
            if not isinstance(row, dict):
                continue

            token = row.get("token")
            if isinstance(token, str):
                if isinstance(row.get("name"), str):
                    category_token_to_name[token] = str(row["name"]).strip().lower()
                if isinstance(row.get("filename"), str):
                    sample_data_token_to_name[token] = str(row["filename"])

            if (
                "category_token" in row
                and "sample_data_token" in row
                and isinstance(row.get("bbox"), list)
            ):
                object_token = str(row.get("token", ""))
                if object_token:
                    object_ann_rows_by_token[object_token] = row

        if not object_ann_rows_by_token:
            return [], []

        bulbs_by_image_name: Dict[str, List[Dict[str, Any]]] = {}
        semantic_by_image_name: Dict[str, List[Dict[str, Any]]] = {}
        skipped_missing_sample_data = 0
        skipped_missing_category = 0

        for object_ann in object_ann_rows_by_token.values():
            sample_data_token = str(object_ann.get("sample_data_token", ""))
            image_name = sample_data_token_to_name.get(sample_data_token)
            if image_name is None:
                skipped_missing_sample_data += 1
                continue

            category_token = str(object_ann.get("category_token", ""))
            category_name = category_token_to_name.get(category_token)
            if category_name is None:
                skipped_missing_category += 1
                continue

            bbox = object_ann.get("bbox")
            if not self._is_valid_bbox(bbox):
                continue
            normalized_bbox = self._normalize_bbox(bbox)
            ann_id = str(object_ann.get("token", ""))

            if category_name == "traffic_light":
                semantic_by_image_name.setdefault(image_name, []).append(
                    {
                        "id": ann_id,
                        "type": "bbox",
                        "title": "traffic_light",
                        "points": normalized_bbox,
                    }
                )
                continue

            bulb_info = self._object_ann_category_to_bulb_info(category_name)
            if bulb_info is None:
                continue
            color, bulb_type = bulb_info
            attributes = [{"key": "type", "title": bulb_type}]
            if object_ann.get("orientation") is not None:
                attributes.append(
                    {"key": "orientation", "title": str(object_ann.get("orientation"))}
                )
            bulbs_by_image_name.setdefault(image_name, []).append(
                {
                    "id": ann_id,
                    "type": "bbox",
                    "title": color,
                    "points": normalized_bbox,
                    "attributes": attributes,
                }
            )

        if skipped_missing_sample_data > 0 or skipped_missing_category > 0:
            logger.warning(
                f"{dataset_name}: skipped {skipped_missing_sample_data} object_ann rows with missing sample_data and "
                f"{skipped_missing_category} rows with missing category lookup"
            )

        bulb_frames = [
            {"name": image_name, "annotations": anns}
            for image_name, anns in sorted(bulbs_by_image_name.items())
        ]
        semantic_frames = [
            {"name": image_name, "annotations": anns}
            for image_name, anns in sorted(semantic_by_image_name.items())
        ]
        return bulb_frames, semantic_frames

    @staticmethod
    def _remove_generic_traffic_light_semantics(
        semantic_annotations: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        filtered_records: List[Dict[str, Any]] = []
        removed_count = 0

        for record in semantic_annotations:
            annotations = record.get("annotations")
            if not isinstance(annotations, list):
                filtered_records.append(record)
                continue

            filtered_annotations: List[Dict[str, Any]] = []
            for ann in annotations:
                title = str(ann.get("title", "")).strip().lower()
                if title == "traffic_light":
                    removed_count += 1
                    continue
                filtered_annotations.append(ann)

            if not filtered_annotations and annotations:
                continue

            if len(filtered_annotations) == len(annotations):
                filtered_records.append(record)
                continue

            updated_record = dict(record)
            updated_record["annotations"] = filtered_annotations
            filtered_records.append(updated_record)

        return filtered_records, removed_count

    @staticmethod
    def _object_ann_category_to_bulb_info(category_name: str) -> Optional[Tuple[str, str]]:
        normalized_name = category_name.strip().lower()
        if normalized_name in FRONT_BULB_LABELS:
            return normalized_name, "circle"
        if normalized_name in BACK_BULB_LABELS:
            return normalized_name, "circle"

        parts = normalized_name.split("_", maxsplit=1)
        if len(parts) != 2:
            return None
        color, bulb_type = parts
        if color not in FRONT_BULB_LABELS:
            return None
        if bulb_type not in {"circle", "arrow", "pedestrian", "cross"}:
            return None
        return color, bulb_type

    def _format_semantic_and_bulb_annotations(
        self,
        bulb_annotations: List[Dict[str, Any]],
        semantic_annotations: List[Dict[str, Any]],
        dataset_name: str,
    ) -> Dict[int, List[Dict[str, Any]]]:
        by_image_bulbs: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        by_image_semantic: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}

        for ann in bulb_annotations:
            image_key = self._extract_image_key(ann)
            if image_key is None:
                continue
            by_image_bulbs.setdefault(image_key, []).extend(
                self._extract_bulb_labels_for_image(ann, dataset_name)
            )

        for ann in semantic_annotations:
            image_key = self._extract_image_key(ann)
            if image_key is None:
                continue
            by_image_semantic.setdefault(image_key, []).extend(
                self._extract_semantic_tlr_boxes_for_image(ann, dataset_name)
            )

        fl_annotations: Dict[int, List[Dict[str, Any]]] = {}
        all_image_keys = set(by_image_bulbs.keys()) | set(by_image_semantic.keys())
        for file_id, camera in sorted(all_image_keys):
            bulbs = by_image_bulbs.get((file_id, camera), [])
            semantic_boxes = by_image_semantic.get((file_id, camera), [])

            labels_for_frame: List[Dict[str, Any]] = []
            labels_for_frame.extend(
                self._convert_bulbs_to_t4_labels(
                    bulbs=bulbs, file_id=file_id, camera=camera, dataset_name=dataset_name
                )
            )
            labels_for_frame.extend(
                self._convert_semantic_boxes_to_t4_labels(
                    semantic_boxes=semantic_boxes,
                    bulbs=bulbs,
                    file_id=file_id,
                    camera=camera,
                    dataset_name=dataset_name,
                )
            )

            if labels_for_frame:
                fl_annotations.setdefault(file_id, []).extend(labels_for_frame)

        return fl_annotations

    @staticmethod
    def _extract_image_key(annotation_file_record: Dict[str, Any]) -> Optional[Tuple[int, str]]:
        name = annotation_file_record.get("name", "")
        parts = name.split("/")
        if len(parts) < 2:
            return None
        camera = parts[-2]
        frame_name = parts[-1]
        try:
            file_id = int(frame_name.split(".")[0])
        except Exception:
            return None
        return file_id, camera

    def _extract_bulb_labels_for_image(
        self, annotation_file_record: Dict[str, Any], dataset_name: str
    ) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        for ann in annotation_file_record.get("annotations", []):
            if ann.get("type") != "bbox":
                continue
            bbox = ann.get("points")
            if not self._is_valid_bbox(bbox):
                continue

            title = str(ann.get("title", "")).strip().lower()
            if title not in SUPPORTED_BULB_LABELS:
                logger.warning(
                    f"{dataset_name}: ignore non-bulb label '{title}' "
                    f"in {annotation_file_record.get('name', '')}"
                )
                continue
            labels.append(
                {
                    "id": str(ann.get("id", "")),
                    "bbox": self._normalize_bbox(bbox),
                    "title": title,
                    "type": self._get_attribute(ann.get("attributes", []), "type"),
                    "orientation": self._get_attribute(ann.get("attributes", []), "orientation"),
                }
            )
        return labels

    def _extract_semantic_tlr_boxes_for_image(
        self, annotation_file_record: Dict[str, Any], dataset_name: str
    ) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        for ann in annotation_file_record.get("annotations", []):
            semantic_title = str(ann.get("title", "")).strip().lower()
            if semantic_title not in SUPPORTED_SEMANTIC_LABELS:
                continue
            ann_type = ann.get("type")
            bbox: Optional[List[float]] = None
            if ann_type == "bbox":
                points = ann.get("points")
                if not self._is_valid_bbox(points):
                    logger.warning(
                        f"{dataset_name}: skip invalid {semantic_title} bbox "
                        f"in {annotation_file_record.get('name', '')}"
                    )
                    continue
                bbox = self._normalize_bbox(points)
            elif ann_type == "segmentation":
                points = ann.get("points")
                bbox = self._semantic_points_to_bbox(points)
                if bbox is None:
                    logger.warning(
                        f"{dataset_name}: skip invalid {semantic_title} segmentation "
                        f"in {annotation_file_record.get('name', '')}"
                    )
                    continue
            else:
                continue
            labels.append({"id": str(ann.get("id", "")), "bbox": bbox, "title": semantic_title})
        return labels

    def _convert_bulbs_to_t4_labels(
        self,
        bulbs: List[Dict[str, Any]],
        file_id: int,
        camera: str,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        t4_labels: List[Dict[str, Any]] = []
        sensor_id = self._camera2idx.get(camera)
        if sensor_id is None:
            logger.warning(f"{dataset_name}: unknown camera '{camera}' at frame {file_id}")
            return t4_labels

        for bulb in bulbs:
            if bulb["title"] not in FRONT_BULB_LABELS:
                continue
            category_name = self._bulb_category_name(
                bulb=bulb, dataset_name=dataset_name, camera=camera, file_id=file_id
            )
            t4_labels.append(
                {
                    "category_name": category_name,
                    # Avoid instance/category mismatches if the same FastLabel id is reused with
                    # different bulb types/orientations.
                    "instance_id": f"{bulb['id']}:{category_name}",
                    "attribute_names": ["occlusion_state.none"],
                    "visibility_name": DEFAULT_VISIBILITY_LEVEL,
                    "two_d_box": bulb["bbox"],
                    "sensor_id": sensor_id,
                }
            )
        return t4_labels

    def _convert_semantic_boxes_to_t4_labels(
        self,
        semantic_boxes: List[Dict[str, Any]],
        bulbs: List[Dict[str, Any]],
        file_id: int,
        camera: str,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        t4_labels: List[Dict[str, Any]] = []
        sensor_id = self._camera2idx.get(camera)
        if sensor_id is None:
            logger.warning(f"{dataset_name}: unknown camera '{camera}' at frame {file_id}")
            return t4_labels

        assignments = self._assign_bulbs_to_semantic_boxes(semantic_boxes, bulbs)
        for idx, semantic in enumerate(semantic_boxes):
            matched_bulbs = [bulbs[bulb_idx] for bulb_idx in assignments.get(idx, [])]
            semantic_title = semantic.get("title", "traffic_light")

            if semantic_title in DIRECT_SEMANTIC_LABELS:
                category_name = semantic_title
            elif semantic_title != "traffic_light":
                logger.warning(
                    f"{dataset_name}: unsupported semantic traffic-light title '{semantic_title}' "
                    f"(camera={camera}, frame={file_id}); mapped to unknown"
                )
                category_name = "unknown"
            elif not matched_bulbs:
                logger.warning(
                    f"{dataset_name}: no bulbs found for traffic_light bbox "
                    f"(camera={camera}, frame={file_id})"
                )
                category_name = "unknown"
            else:
                category_name = self._map_semantic_class(
                    matched_bulbs=matched_bulbs,
                    dataset_name=dataset_name,
                    camera=camera,
                    file_id=file_id,
                )

            t4_labels.append(
                {
                    "category_name": category_name,
                    "instance_id": semantic["id"],
                    "attribute_names": ["occlusion_state.none"],
                    "visibility_name": DEFAULT_VISIBILITY_LEVEL,
                    "two_d_box": semantic["bbox"],
                    "sensor_id": sensor_id,
                }
            )

        return t4_labels

    def _assign_bulbs_to_semantic_boxes(
        self, semantic_boxes: List[Dict[str, Any]], bulbs: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        assigned: Dict[int, List[int]] = {}
        for bulb_idx, bulb in enumerate(bulbs):
            center_x, center_y = self._bbox_center(bulb["bbox"])
            matched_box_indices: List[int] = []
            for box_idx, semantic in enumerate(semantic_boxes):
                if self._point_in_bbox(center_x, center_y, semantic["bbox"]):
                    matched_box_indices.append(box_idx)
            if not matched_box_indices:
                continue
            if len(matched_box_indices) > 1:
                matched_box_indices = sorted(
                    matched_box_indices,
                    key=lambda idx: self._bbox_area(semantic_boxes[idx]["bbox"]),
                )
                logger.warning(
                    "bulb center matched multiple semantic traffic_light boxes; "
                    "assigned to smallest box"
                )
            assigned.setdefault(matched_box_indices[0], []).append(bulb_idx)
        return assigned

    def _map_semantic_class(
        self,
        matched_bulbs: List[Dict[str, Any]],
        dataset_name: str,
        camera: str,
        file_id: int,
    ) -> str:
        front_bulbs = [bulb for bulb in matched_bulbs if bulb["title"] in FRONT_BULB_LABELS]
        back_bulbs = [bulb for bulb in matched_bulbs if bulb["title"] in BACK_BULB_LABELS]

        crosswalk_bulbs = [
            bulb
            for bulb in front_bulbs
            if self._is_crosswalk_bulb_type((bulb.get("type") or "").strip().lower())
        ]
        vehicular_bulbs = [
            bulb
            for bulb in front_bulbs
            if not self._is_crosswalk_bulb_type((bulb.get("type") or "").strip().lower())
        ]

        if crosswalk_bulbs and vehicular_bulbs:
            logger.warning(
                f"{dataset_name}: mixed pedestrian/crosswalk and vehicular bulbs "
                f"{self._summarize_bulbs(matched_bulbs)} "
                f"(camera={camera}, frame={file_id}); mapped to unknown"
            )
            return "unknown"

        if crosswalk_bulbs:
            return self._map_crosswalk_class(crosswalk_bulbs, dataset_name, camera, file_id)

        if vehicular_bulbs:
            return self._map_vehicle_tlr_class(vehicular_bulbs, dataset_name, camera, file_id)

        if back_bulbs:
            has_crosswalk_back_hint = any(
                self._is_crosswalk_bulb_type((bulb.get("type") or "").strip().lower())
                for bulb in back_bulbs
            )
            return "crosswalk_light_back" if has_crosswalk_back_hint else "traffic_light_back"

        logger.warning(
            f"{dataset_name}: strange bulb combination {self._summarize_bulbs(matched_bulbs)} "
            f"(camera={camera}, frame={file_id}); mapped to unknown"
        )
        return "unknown"

    def _map_crosswalk_class(
        self,
        crosswalk_bulbs: List[Dict[str, Any]],
        dataset_name: str,
        camera: str,
        file_id: int,
    ) -> str:
        red_on = any(bulb["title"] == "red" for bulb in crosswalk_bulbs)
        green_on = any(bulb["title"] == "green" for bulb in crosswalk_bulbs)
        yellow_on = any(bulb["title"] == "yellow" for bulb in crosswalk_bulbs)

        if red_on and not green_on and not yellow_on:
            return "crosswalk_red"
        if green_on and not red_on and not yellow_on:
            return "crosswalk_green"

        logger.warning(
            f"{dataset_name}: unsupported crosswalk bulb combination "
            f"{self._summarize_bulbs(crosswalk_bulbs)} "
            f"(camera={camera}, frame={file_id}); mapped to crosswalk_unknown"
        )
        return "crosswalk_unknown"

    def _map_vehicle_tlr_class(
        self,
        vehicular_bulbs: List[Dict[str, Any]],
        dataset_name: str,
        camera: str,
        file_id: int,
    ) -> str:
        circle_colors: Set[str] = set()
        arrows_by_color: Dict[str, Set[str]] = {color: set() for color in FRONT_BULB_LABELS}

        has_unsupported_arrow = False
        for bulb in vehicular_bulbs:
            color = str(bulb.get("title", "")).strip().lower()
            bulb_type = (bulb.get("type") or "").strip().lower()
            if color not in FRONT_BULB_LABELS:
                continue

            if bulb_type == "arrow":
                direction = self._orientation_to_direction(bulb.get("orientation"))
                if direction is None:
                    has_unsupported_arrow = True
                else:
                    arrows_by_color[color].add(direction)
                continue

            # Treat unknown/non-arrow vehicle bulb types as circle bulbs.
            circle_colors.add(color)

        if has_unsupported_arrow:
            logger.warning(
                f"{dataset_name}: unsupported arrow orientation detected "
                f"{self._summarize_bulbs(vehicular_bulbs)} "
                f"(camera={camera}, frame={file_id}); mapped to unknown"
            )
            return "unknown"

        if len(circle_colors) > 1:
            logger.warning(
                f"{dataset_name}: multiple circle colors in one traffic_light "
                f"{self._summarize_bulbs(vehicular_bulbs)} "
                f"(camera={camera}, frame={file_id}); mapped to unknown"
            )
            return "unknown"

        base_circle = next(iter(circle_colors), None)
        has_any_arrow = any(arrows_by_color[color] for color in FRONT_BULB_LABELS)

        # Keep legacy classes for single-red-circle + green-arrow combinations.
        if base_circle == "red" and not arrows_by_color["red"] and not arrows_by_color["yellow"]:
            legacy_class = LEGACY_RED_CLASS_BY_GREEN_ARROWS.get(frozenset(arrows_by_color["green"]))
            if legacy_class is not None:
                return legacy_class

        if base_circle in {"green", "yellow"} and not has_any_arrow:
            return base_circle

        if base_circle is None and not has_any_arrow:
            logger.warning(
                f"{dataset_name}: no vehicular bulbs after filtering "
                f"{self._summarize_bulbs(vehicular_bulbs)} "
                f"(camera={camera}, frame={file_id}); mapped to unknown"
            )
            return "unknown"

        arrow_tokens = self._build_arrow_tokens(arrows_by_color)
        if base_circle is None:
            if not arrow_tokens:
                logger.warning(
                    f"{dataset_name}: failed to map traffic-light bulbs "
                    f"{self._summarize_bulbs(vehicular_bulbs)} "
                    f"(camera={camera}, frame={file_id}); mapped to unknown"
                )
                return "unknown"
            return ",".join(arrow_tokens)

        if not arrow_tokens:
            return base_circle

        return ",".join([base_circle, *arrow_tokens])

    @staticmethod
    def _build_arrow_tokens(arrows_by_color: Dict[str, Set[str]]) -> List[str]:
        tokens: List[str] = []
        for direction in VEHICLE_ARROW_DIRECTION_ORDER:
            if direction in arrows_by_color.get("green", set()):
                tokens.append(direction)
        for arrow_color in ("red", "yellow"):
            for direction in VEHICLE_ARROW_DIRECTION_ORDER:
                if direction in arrows_by_color.get(arrow_color, set()):
                    tokens.append(f"{direction}_{arrow_color}")
        return tokens

    def _bulb_category_name(
        self,
        bulb: Dict[str, Any],
        dataset_name: str,
        camera: str,
        file_id: int,
    ) -> str:
        """Return output category name for a single bulb bbox.

        - Non-arrow bulbs: `{color}_bulb`
        - Arrow bulbs: `{color}_{direction}_arrow_bulb` (direction inferred from orientation)
        - Unsupported arrow orientation: `{color}_arrow_unknown_bulb`
        """
        color = bulb["title"]
        bulb_type = (bulb.get("type") or "").strip().lower()
        if bulb_type != "arrow":
            return f"{color}_bulb"

        direction = self._orientation_to_direction(bulb.get("orientation"))
        if direction is None:
            logger.warning(
                f"{dataset_name}: unsupported arrow orientation '{bulb.get('orientation')}' "
                f"(camera={camera}, frame={file_id}); exported as {color}_arrow_unknown_bulb"
            )
            return f"{color}_arrow_unknown_bulb"

        return f"{color}_{direction}_arrow_bulb"

    @staticmethod
    def _summarize_bulbs(bulbs: List[Dict[str, Any]]) -> List[str]:
        values: List[str] = []
        for bulb in bulbs:
            orientation = bulb.get("orientation")
            values.append(
                f"{bulb.get('title','')}:{(bulb.get('type') or '').strip().lower()}:"
                f"{str(orientation).strip().lower() if orientation is not None else ''}"
            )
        return values

    @staticmethod
    def _semantic_points_to_bbox(points: Any) -> Optional[List[float]]:
        if not isinstance(points, list) or len(points) == 0:
            return None

        x_min = None
        y_min = None
        x_max = None
        y_max = None
        for polygon_group in points:
            if not isinstance(polygon_group, list) or len(polygon_group) == 0:
                continue
            outer_polygon = polygon_group[0]
            if not isinstance(outer_polygon, list) or len(outer_polygon) < 4:
                continue
            bbox = _convert_polygon_to_bbox(outer_polygon)
            x1, y1, x2, y2 = bbox
            x_min = x1 if x_min is None else min(x_min, x1)
            y_min = y1 if y_min is None else min(y_min, y1)
            x_max = x2 if x_max is None else max(x_max, x2)
            y_max = y2 if y_max is None else max(y_max, y2)
        if x_min is None:
            return None
        return [x_min, y_min, x_max, y_max]

    def _orientation_to_direction(self, orientation: Any) -> Optional[str]:
        if orientation is None:
            return "straight"

        if isinstance(orientation, (int, float)):
            return self._direction_from_numeric_orientation(float(orientation))

        value = str(orientation).strip().lower()
        if value in {"", "n/a", "na"}:
            return "straight"
        if value in {"left", "right", "straight", "up_left", "up_right"}:
            return value
        if value == "minus_90":
            return "left"
        if value == "minus_45":
            return "up_left"
        try:
            return self._direction_from_numeric_orientation(float(value))
        except ValueError:
            return None

    def _direction_from_numeric_orientation(self, raw_orientation: float) -> Optional[str]:
        angle_deg = self._normalize_orientation_to_degrees(raw_orientation)
        for direction, target_deg in (
            ("straight", 0.0),
            ("left", -90.0),
            ("right", 90.0),
            ("up_left", -45.0),
            ("up_right", 45.0),
        ):
            if self._is_close_angle(angle_deg, target_deg, self._arrow_angle_tolerance_deg):
                return direction
        return None

    @staticmethod
    def _normalize_orientation_to_degrees(raw_orientation: float) -> float:
        if abs(raw_orientation) <= (2.0 * math.pi + 1e-6):
            return math.degrees(raw_orientation)
        return raw_orientation

    @staticmethod
    def _is_close_angle(angle_deg: float, target_deg: float, tolerance_deg: float) -> bool:
        diff = ((angle_deg - target_deg + 180.0) % 360.0) - 180.0
        return abs(diff) <= tolerance_deg

    @staticmethod
    def _is_crosswalk_bulb_type(bulb_type: str) -> bool:
        return bulb_type in CROSSWALK_BULB_TYPES

    @staticmethod
    def _is_valid_bbox(bbox: Any) -> bool:
        return isinstance(bbox, list) and len(bbox) == 4

    @staticmethod
    def _normalize_bbox(bbox: List[float]) -> List[float]:
        x1, y1, x2, y2 = bbox
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _point_in_bbox(x: float, y: float, bbox: List[float]) -> bool:
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    @staticmethod
    def _bbox_area(bbox: List[float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @staticmethod
    def _get_attribute(attributes: List[Dict[str, Any]], key: str) -> Optional[str]:
        for att in attributes:
            if att.get("key") != key:
                continue
            title = att.get("title")
            value = att.get("value")
            if title is not None:
                return str(title)
            if value is not None:
                return str(value)
            return None
        return None
