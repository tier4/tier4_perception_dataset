from datetime import datetime, timezone
import json
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


class FastLabel2dSemanticToT4TlrConverter(FastLabel2dToT4Converter):
    """Convert FastLabel bulb bbox + semantic traffic_light polygons into T4 2D annotations.

    Conversion rules:
    - Semantic polygons with `title` in supported traffic-light classes are converted to bbox.
    - Bulb bboxes (`red`/`green`/`yellow`) are converted into `<color>_bulb` labels.
    - Parent semantic class is inferred from bulbs inside each generic `traffic_light` bbox:
      - `green`, `yellow`, `red`
      - `left,red`, `left,red,straight`, `red,right`, `red,right,straight`, `red,straight`
      - `red,up_left`, `red,up_right`
      - `crosswalk_red`, `crosswalk_green`, `crosswalk_unknown`
      - `traffic_light_back`, `crosswalk_light_back`
      - fallback `unknown`

    Corner cases:
    - no bulb in semantic bbox -> warning + `unknown`
    - unsupported bulb combination -> warning + `unknown`
    - unsupported arrow orientation -> warning + `unknown` (via unsupported combination)
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

    def convert(self) -> None:
        t4_datasets = sorted([d.name for d in self._input_base.iterdir() if d.is_dir()])
        logger.info(f"Found {len(t4_datasets)} datasets to process")
        skipped_datasets: List[str] = []

        bulb_files_by_dataset = self._group_annotation_files_by_dataset(t4_datasets)
        semantic_files_by_dataset = self._group_annotation_files_by_dataset_with_base(
            self._input_semantic_anno_base, t4_datasets
        )

        for t4dataset_name in t4_datasets:
            if t4dataset_name not in bulb_files_by_dataset:
                logger.warning(f"No bulb annotation files found for {t4dataset_name}")
                continue
            if t4dataset_name not in semantic_files_by_dataset:
                logger.warning(f"No semantic annotation files found for {t4dataset_name}")
                continue
            logger.info(f"Processing dataset: {t4dataset_name}")

            input_dir = self._input_base / t4dataset_name
            input_annotation_dir = input_dir / "annotation"
            if not osp.exists(input_annotation_dir):
                logger.warning(f"input_dir {input_dir} not exists.")
                continue

            output_dir = self._output_base / t4dataset_name
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
            else:
                raise ValueError("If you want to overwrite files, use --overwrite option.")

            bulb_annotations = self._load_annotation_jsons_for_dataset(
                bulb_files_by_dataset[t4dataset_name]
            )
            semantic_annotations = self._load_annotation_jsons_for_dataset(
                semantic_files_by_dataset[t4dataset_name]
            )
            fl_annotations = self._format_semantic_and_bulb_annotations(
                bulb_annotations=bulb_annotations,
                semantic_annotations=semantic_annotations,
                dataset_name=t4dataset_name,
            )

            annotation_files_generator = AnnotationFilesGenerator(description=self._description)
            try:
                annotation_files_generator.convert_one_scene(
                    input_dir=input_dir,
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

    @staticmethod
    def _group_annotation_files_by_dataset_with_base(
        annotation_base: Path, t4_datasets: List[str]
    ) -> Dict[str, List[Path]]:
        anno_files_by_dataset: Dict[str, List[Path]] = {}
        all_anno_files = list(annotation_base.rglob("*.json"))

        for dataset_name in tqdm(t4_datasets):
            matched_files = [f for f in all_anno_files if dataset_name in f.name]
            if matched_files:
                anno_files_by_dataset[dataset_name] = matched_files
        return anno_files_by_dataset

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
            if ann.get("type") != "segmentation":
                continue
            semantic_title = str(ann.get("title", "")).strip().lower()
            if semantic_title not in SUPPORTED_SEMANTIC_LABELS:
                continue
            points = ann.get("points")
            bbox = self._semantic_points_to_bbox(points)
            if bbox is None:
                logger.warning(
                    f"{dataset_name}: skip invalid {semantic_title} segmentation "
                    f"in {annotation_file_record.get('name', '')}"
                )
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
            t4_labels.append(
                {
                    "category_name": f"{bulb['title']}_bulb",
                    "instance_id": bulb["id"],
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
        red_on = False
        yellow_on = False
        green_on = False
        arrows: Set[str] = set()

        has_unsupported_arrow = False
        has_non_green_arrow = False
        for bulb in vehicular_bulbs:
            color = bulb["title"]
            bulb_type = (bulb.get("type") or "").strip().lower()

            if bulb_type == "arrow":
                if color != "green":
                    has_non_green_arrow = True
                    continue
                direction = self._orientation_to_direction(bulb.get("orientation") or "")
                if direction is None:
                    has_unsupported_arrow = True
                else:
                    arrows.add(direction)
                continue

            if color == "red":
                red_on = True
            elif color == "yellow":
                yellow_on = True
            elif color == "green":
                green_on = True

        if has_non_green_arrow:
            logger.warning(
                f"{dataset_name}: non-green arrow bulb detected "
                f"(camera={camera}, frame={file_id}); mapped to unknown"
            )
            return "unknown"

        if has_unsupported_arrow:
            logger.warning(
                f"{dataset_name}: unsupported arrow orientation detected "
                f"(camera={camera}, frame={file_id}); mapped to unknown"
            )
            return "unknown"

        if red_on and not yellow_on and not green_on:
            if arrows == set():
                return "red"
            if arrows == {"left"}:
                return "left,red"
            if arrows == {"left", "straight"}:
                return "left,red,straight"
            if arrows == {"right"}:
                return "red,right"
            if arrows == {"straight"}:
                return "red,straight"
            if arrows == {"right", "straight"}:
                return "red,right,straight"
            if arrows == {"up_left"}:
                return "red,up_left"
            if arrows == {"up_right"}:
                return "red,up_right"

        if not red_on and yellow_on and not green_on and not arrows:
            return "yellow"

        if not red_on and not yellow_on and green_on and not arrows:
            return "green"

        logger.warning(
            f"{dataset_name}: strange bulb combination {self._summarize_bulbs(vehicular_bulbs)} "
            f"(camera={camera}, frame={file_id}); mapped to unknown"
        )
        return "unknown"

    @staticmethod
    def _summarize_bulbs(bulbs: List[Dict[str, Any]]) -> List[str]:
        values: List[str] = []
        for bulb in bulbs:
            values.append(
                f"{bulb.get('title','')}:{(bulb.get('type') or '').strip().lower()}:"
                f"{(bulb.get('orientation') or '').strip().lower()}"
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

    @staticmethod
    def _orientation_to_direction(orientation: str) -> Optional[str]:
        value = orientation.strip().lower()
        if value in {"", "n/a", "na", "0"}:
            return "straight"
        if value in {"-90", "minus_90"}:
            return "left"
        if value in {"90"}:
            return "right"
        if value in {"-45", "minus_45"}:
            return "up_left"
        if value in {"45"}:
            return "up_right"
        return None

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
