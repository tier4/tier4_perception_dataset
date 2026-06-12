# Copyright 2024 Tier IV, Inc.
# This is a sample script to report annotation progress for specific scenes in a Kognic project, given by dataset filename/path (the basename is the scene external_id).
# The Dataset directory name is the equivalent to the scene external_id, so the basename of each given path is matched against the scene external_id
# (or scene uuid). Delivered annotations are fetched per scene.

# example usage:
## uv run python perception_dataset/kognic/get_scene_status.py --organization-id 114 --workspace-id efa90d1e-99bc-4064-98bb-5bfc8758157d
# --project-external-id semantic_segmentation_poc_tier_iv --filenames DB_semaseg_kognic_x2_dev_cfa23601-97c4-4d47-a37e-edfc7e080d8c_2025-07-04_14-39-52_14-40-52
# or add --include-annotation-stats to also fetch annotation contents per input (shape counts + category counts; slower)

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Tuple

from requests import HTTPError

from perception_dataset.kognic.get_project_progress import (
    KognicProgressConfig,
    KognicProjectProgress,
)
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


@dataclass(frozen=True)
class KognicSceneStatusConfig(KognicProgressConfig):
    scenes: Tuple[str, ...] = ()


class KognicSceneStatus(KognicProjectProgress):
    """Report annotation progress for specific scenes instead of a whole project/batch.

    Scenes are given as dataset filenames/paths: the uploader uses the dataset
    directory name as the scene external_id, so the basename of each path is
    matched against the scene external_id (or scene uuid). Delivered
    annotations are fetched per scene, so only the requested scenes are
    touched regardless of project size.
    """

    config: KognicSceneStatusConfig

    def report(self) -> Dict:
        annotation_types = self.kognic_io_client.project.get_annotation_types(
            self.config.project_external_id, batch=self.config.batch
        )
        all_inputs = self.kognic_io_client.input.query_inputs(
            project=self.config.project_external_id,
            batch=self.config.batch,
        )

        requested = set(self.config.scenes)
        inputs = [
            inp
            for inp in all_inputs
            if inp.scene_external_id in requested or inp.scene_uuid in requested
        ]
        found = {inp.scene_external_id for inp in inputs} | {inp.scene_uuid for inp in inputs}
        not_found = sorted(requested - found)
        if not_found:
            logger.warning(
                f"{len(not_found)} requested scene(s) not found in project "
                f"'{self.config.project_external_id}' (batch={self.config.batch or '(all)'}): "
                f"{not_found}"
            )

        logger.info(
            f"Project '{self.config.project_external_id}' "
            f"(batch={self.config.batch or '(all)'}): "
            f"{len(inputs)}/{len(requested)} requested scene(s) found, "
            f"annotation_types={annotation_types}"
        )

        scene_uuids = [inp.scene_uuid for inp in inputs]
        scenes_by_uuid = self._fetch_scenes(scene_uuids)
        frames_by_scene = self._fetch_frame_counts(scene_uuids)
        delivered_by_scene, stats_by_scene = self._fetch_scene_annotations(
            scene_uuids, annotation_types
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
            "scenes": sorted(requested),
            "scenes_not_found": not_found,
            "annotation_types": annotation_types,
            "total_inputs": len(inputs),
            "total_frames_to_annotate": sum(n for n in frames_by_scene.values() if n is not None),
            "annotation_completed": n_completed,
            "inputs": input_entries,
        }

        self._save_report(report)
        return report

    def _fetch_scene_annotations(
        self, scene_uuids: List[str], annotation_types: List[str]
    ) -> tuple:
        """Fetch delivered annotations per scene and annotation type.

        Returns (delivered_by_scene, stats_by_scene) with the same shapes as
        KognicProjectProgress._fetch_delivered_annotations, but queries each
        scene individually (404 means the annotation is not delivered yet).
        """
        delivered: Dict[str, set] = {}
        stats: Dict[str, List[Dict]] = {}
        for scene_uuid in scene_uuids:
            for annotation_type in annotation_types:
                try:
                    annotation = self.kognic_io_client.annotation.get_annotation(
                        scene_uuid=scene_uuid, annotation_type=annotation_type
                    )
                except HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        continue
                    raise
                delivered.setdefault(scene_uuid, set()).add(annotation_type)
                if self.config.include_annotation_stats and annotation.content:
                    stats.setdefault(scene_uuid, []).append(
                        self._build_annotation_stats(annotation_type, annotation)
                    )
            logger.info(
                f"  scene {scene_uuid}: "
                f"delivered={sorted(delivered.get(scene_uuid, set())) or 'none'}"
            )
        return delivered, stats


def main():
    time_start = time.time()
    parser = argparse.ArgumentParser(
        description="Report Kognic annotation status for specific scenes, "
        "given by dataset filename/path (the basename is the scene external_id)."
    )
    parser.add_argument("--organization-id", type=str, required=True)
    parser.add_argument("--workspace-id", type=str, required=True)
    parser.add_argument("--project-external-id", type=str, required=True)
    parser.add_argument(
        "--filenames",
        type=str,
        nargs="+",
        required=True,
        help="Dataset filenames/paths to report on; the basename is the scene "
        "external_id (the uploader uses the dataset directory name as external_id). "
        "Bare scene names or scene uuids also work.",
    )
    parser.add_argument("--batch", type=str, default=None)
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports_scene.json",
        help="Where the status report JSON is written",
    )
    parser.add_argument(
        "--include-annotation-stats",
        action="store_true",
        help="Also fetch annotation contents per input (shape counts + category counts; slower)",
    )
    args = parser.parse_args()

    config = KognicSceneStatusConfig(
        organization_id=args.organization_id,
        workspace_id=args.workspace_id,
        project_external_id=args.project_external_id,
        report_path=Path(args.report_path),
        batch=args.batch,
        include_annotation_stats=args.include_annotation_stats,
        # The uploader uses the dataset directory name as the scene external_id,
        # so the basename of each given path is the scene id.
        scenes=tuple(Path(f).name for f in args.filenames),
    )
    KognicSceneStatus(config).report()
    time_end = time.time()
    logger.info(f"Total execution time: {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
