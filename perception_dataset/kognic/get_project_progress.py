import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Dict, List, Optional

from kognic.io.client import KognicIOClient
import yaml

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

# The size to fetch scene UUIDs in chunks to keep each request body small.
_SCENE_UUIDS_CHUNK_SIZE = 100


@dataclass(frozen=True)
class KognicProgressConfig:
    organization_id: str
    workspace_id: str
    project_external_id: str
    report_path: Path
    batch: Optional[str] = None
    include_annotation_stats: bool = False


def _load_progress_config(config_dict: Dict) -> KognicProgressConfig:
    conversion = config_dict["conversion"]
    organization_id = conversion.get("organization_id") or conversion.get("client_organization_id")
    workspace_id = conversion.get("workspace_id") or conversion.get("write_workspace_id")

    if not organization_id:
        raise ValueError("conversion.organization_id is required")
    if not workspace_id:
        raise ValueError("conversion.workspace_id is required")
    if not conversion.get("project_external_id"):
        raise ValueError("conversion.project_external_id is required")
    if not conversion.get("batch"):
        logger.info("No batch specified in config; reporting on all batches in the project")

    return KognicProgressConfig(
        organization_id=organization_id,
        workspace_id=workspace_id,
        project_external_id=conversion["project_external_id"],
        report_path=Path(conversion.get("report_path", "reports.json")),
        batch=conversion.get("batch"),
        include_annotation_stats=conversion.get("include_annotation_stats", False),
    )


def _count_shapes(openlabel_content: Dict) -> Counter:
    """Count object_data entries per OpenLabel data category (bbox, cuboid, poly2d, binary, ...)."""
    counts: Counter = Counter()
    openlabel = openlabel_content.get("openlabel", openlabel_content)

    for frame in openlabel.get("frames", {}).values():
        for obj in frame.get("objects", {}).values():
            for data_type, entries in obj.get("object_data", {}).items():
                if isinstance(entries, list):
                    counts[data_type] += len(entries)
    return counts


_RLE_TOKEN = re.compile(r"#(\d+)V(\d+)")


def _count_segmentation_points(openlabel_content: Dict) -> Counter:
    """Count points per class id from RLE-encoded segmentation labels.

    Kognic delivers segmentation as `binary` object_data entries with
    encoding='rle' and val like '#138V2#2V4...': 138 points of class 2,
    then 2 points of class 4, etc. Returns class id -> total point count.
    """
    counts: Counter = Counter()
    openlabel = openlabel_content.get("openlabel", openlabel_content)
    class_names = _get_class_names(openlabel_content)

    for frame in openlabel.get("frames", {}).values():
        for obj in frame.get("objects", {}).values():
            for entry in obj.get("object_data", {}).get("binary", []):
                if entry.get("encoding") != "rle":
                    continue
                for count, class_id in _RLE_TOKEN.findall(entry.get("val", "")):
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    counts[class_name] += int(count)
    return counts


def _get_class_names(openlabel_content: Dict) -> Dict[str, str]:
    """Map class id -> class name from the OpenLabel ontology (e.g. '1' -> 'drivable_surface').

    Handles both layouts: ontologies keyed by ontology uid ({'0': {'classifications': ...}})
    and a flat ontology entry ({'classifications': ...}).
    """
    openlabel = openlabel_content.get("openlabel", openlabel_content)
    ontologies = openlabel.get("ontologies") or {}
    if "classifications" in ontologies:
        return ontologies["classifications"]
    return ontologies.get("0", {}).get("classifications", {})


class KognicProjectProgress:
    def __init__(self, config: KognicProgressConfig):
        self.config = config
        self._kognic_io_client: Optional[KognicIOClient] = None

    @property
    def kognic_io_client(self) -> KognicIOClient:
        if self._kognic_io_client is None:
            self._kognic_io_client = KognicIOClient(
                client_organization_id=self.config.organization_id,
                write_workspace_id=self.config.workspace_id,
            )
        return self._kognic_io_client

    def report(self) -> Dict:
        batches = self._collect_batches()
        annotation_types = self.kognic_io_client.project.get_annotation_types(
            self.config.project_external_id, batch=self.config.batch
        )
        inputs = self.kognic_io_client.input.query_inputs(
            project=self.config.project_external_id,
            batch=self.config.batch,
        )
        scenes_by_uuid = self._fetch_scenes([inp.scene_uuid for inp in inputs])
        frames_by_scene = self._fetch_frame_counts(list(scenes_by_uuid.keys()))
        delivered_by_scene, stats_by_scene = self._fetch_delivered_annotations(annotation_types)

        logger.info(
            f"Project '{self.config.project_external_id}' "
            f"(batch={self.config.batch or '(all)'}): "
            f"{len(inputs)} input(s), annotation_types={annotation_types}, "
            f"batches={[(b['batch'], b['status']) for b in batches]}"
        )

        input_entries = []
        for inp in inputs:
            entry = self._build_input_entry(
                inp, scenes_by_uuid, annotation_types, delivered_by_scene
            )
            if self.config.include_annotation_stats:
                entry["annotations"] = stats_by_scene.get(inp.scene_uuid, [])
            input_entries.append(entry)

        n_completed = sum(1 for e in input_entries if e["annotation_progress"] == "completed")

        report = {
            "project": self.config.project_external_id,
            "batch": self.config.batch,
            "annotation_types": annotation_types,
            "batches": batches,
            "total_inputs": len(inputs),
            "total_frames_to_annotate": sum(
                n for n in frames_by_scene.values() if n is not None
            ),
            "annotation_completed": n_completed,
            "inputs": input_entries,
        }

        self._save_report(report)
        return report

    def _collect_batches(self) -> List[Dict]:
        return [
            {
                "batch": b.batch,
                "title": b.title,
                "status": b.status.value,
                "created": b.created.isoformat(),
                "updated": b.updated.isoformat(),
            }
            for b in self.kognic_io_client.project.get_project_batches(
                self.config.project_external_id
            )
        ]

    def _fetch_scenes(self, scene_uuids: List[str]) -> Dict:
        scenes = {}
        for i in range(0, len(scene_uuids), _SCENE_UUIDS_CHUNK_SIZE):
            chunk = scene_uuids[i : i + _SCENE_UUIDS_CHUNK_SIZE]
            for scene in self.kognic_io_client.scene.get_scenes_by_uuids(chunk):
                scenes[scene.uuid] = scene
        return scenes

    def _fetch_frame_counts(self, scene_uuids: List[str]) -> Dict[str, Optional[int]]:
        """Fetch the number of frames to annotate (metadata annotate=True) per scene.

        Uses the raw scene endpoint because SceneSummary drops per-frame metadata,
        where the uploader stores the annotate flag. Scenes without per-frame
        metadata fall back to the total frame count (single-frame scenes count as 1).
        Returns scene_uuid -> frame count (None if the scene could not be fetched).
        """
        frames: Dict[str, Optional[int]] = {}
        for i, scene_uuid in enumerate(scene_uuids, start=1):
            try:
                scene_json = self.kognic_io_client.scene._client.get(f"v2/scenes/{scene_uuid}")
                frames[scene_uuid] = self._count_frames_to_annotate(scene_json)
            except Exception as e:
                logger.warning(f"Failed to fetch scene summary for {scene_uuid}: {e}")
                frames[scene_uuid] = None
            if i % 10 == 0:
                logger.info(f"  ...{i}/{len(scene_uuids)} scene summaries fetched")
        return frames

    @staticmethod
    def _count_frames_to_annotate(scene_json: Dict) -> int:
        frame_entries = scene_json.get("frames")
        if isinstance(frame_entries, list) and frame_entries:
            flagged = [frame.get("metadata", {}).get("annotate") for frame in frame_entries]
            if any(value is not None for value in flagged):
                return sum(1 for value in flagged if value)
            return len(frame_entries)
        relative_times = scene_json.get("frameRelativeTimes")
        return len(relative_times) if relative_times else 1

    def _fetch_delivered_annotations(self, annotation_types: List[str]) -> tuple:
        """Fetch delivered annotations once per annotation type.

        Returns (delivered_by_scene, stats_by_scene):
          delivered_by_scene: scene_uuid -> set of annotation types already delivered
          stats_by_scene: scene_uuid -> list of per-annotation stats (only populated
            when include_annotation_stats is on; contents are fetched in the same pass)
        """
        include_content = self.config.include_annotation_stats
        delivered: Dict[str, set] = {}
        stats: Dict[str, List[Dict]] = {}
        for annotation_type in annotation_types:
            logger.info(
                f"Fetching delivered annotations for annotation_type='{annotation_type}' "
                f"(include_content={include_content})..."
            )
            annotations = self.kognic_io_client.annotation.get_project_annotations(
                project=self.config.project_external_id,
                annotation_type=annotation_type,
                batch=self.config.batch,
                include_content=include_content,
            )
            n_fetched = 0
            for annotation in annotations:
                delivered.setdefault(annotation.scene_uuid, set()).add(annotation_type)
                if include_content and annotation.content:
                    stats.setdefault(annotation.scene_uuid, []).append(
                        self._build_annotation_stats(annotation_type, annotation)
                    )
                n_fetched += 1
                if n_fetched % 10 == 0:
                    logger.info(f"  ...{n_fetched} annotation(s) fetched so far")
            logger.info(f"  done: {n_fetched} annotation(s) for '{annotation_type}'")
        return delivered, stats

    @staticmethod
    def _build_annotation_stats(annotation_type: str, annotation) -> Dict:
        shape_counts = _count_shapes(annotation.content)
        stats = {
            "annotation_type": annotation_type,
            "created": annotation.created.isoformat(),
            "total_annotations": sum(shape_counts.values()),
            "shapes": dict(sorted(shape_counts.items())),
        }

        point_counts = _count_segmentation_points(annotation.content)
        if point_counts:
            stats["total_points"] = sum(point_counts.values())
            stats["points_per_class"] = dict(point_counts.most_common())
        # TODO: add more annotation-type-specific stats here as needed (e.g. object counts per class for bounding box annotations)

        return stats

    def _build_input_entry(
        self,
        inp,
        scenes_by_uuid: Dict,
        annotation_types: List[str],
        delivered_by_scene: Dict[str, set],
    ) -> Dict:
        delivered = delivered_by_scene.get(inp.scene_uuid, set())
        if delivered >= set(annotation_types):
            annotation_progress = "completed"
        elif delivered:
            annotation_progress = "partial"
        else:
            annotation_progress = "not delivered"

        entry = {
            "external_id": inp.scene_external_id,
            "scene_uuid": inp.scene_uuid,
            "input_uuid": inp.uuid,
            "view_link": inp.view_link,
            "annotation_progress": annotation_progress,
            "delivered_annotation_types": sorted(delivered),
        }

        scene = scenes_by_uuid.get(inp.scene_uuid)
        if scene is not None:
            entry["scene_type"] = scene.scene_type
            entry["created"] = scene.created.isoformat()
            if scene.error_message:
                entry["error_message"] = scene.error_message

        return entry

    def _save_report(self, report: Dict) -> None:
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(
            f"Report saved to {self.config.report_path} "
            f"({report['annotation_completed']}/{report['total_inputs']} inputs completed, "
            f"{report['total_frames_to_annotate']} frames to annotate)"
        )

def main():
    time_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/get_kognic_project_progress_sample.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    assert (
        config_dict["task"] == "get_kognic_project_progress"
    ), f"Expected task 'get_kognic_project_progress', got '{config_dict['task']}'"

    progress_config = _load_progress_config(config_dict)
    KognicProjectProgress(progress_config).report()
    time_end = time.time()
    logger.info(f"Total execution time: {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
