import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional

from kognic.io.client import KognicIOClient
import yaml
import time

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


@dataclass(frozen=True)
class KognicDownloadConfig:
    output_base: Path
    organization_id: str
    workspace_id: str
    project_external_id: str
    annotation_type: Optional[str] = None
    batch: Optional[str] = None
    scene_external_id: Optional[str] = None
    iso_rotated_cuboids: bool = False


def _load_download_config(config_dict: Dict) -> KognicDownloadConfig:
    conversion = config_dict["conversion"]
    organization_id = conversion.get("organization_id") or conversion.get("client_organization_id")
    workspace_id = conversion.get("workspace_id") or conversion.get("write_workspace_id")
    scene_external_id = conversion.get("scene_external_id")

    if not organization_id:
        raise ValueError("conversion.organization_id is required")
    if not workspace_id:
        raise ValueError("conversion.workspace_id is required")
    if not conversion.get("project_external_id"):
        raise ValueError("conversion.project_external_id is required")
    # annotation_type is only needed for project-wide downloads; a per-scene
    # download (scene_external_id set) fetches every annotation type for the scene.
    if not scene_external_id and not conversion.get("annotation_type"):
        raise ValueError(
            "conversion.annotation_type is required (unless conversion.scene_external_id is set)"
        )

    return KognicDownloadConfig(
        output_base=Path(conversion["output_base"]),
        organization_id=organization_id,
        workspace_id=workspace_id,
        project_external_id=conversion["project_external_id"],
        annotation_type=conversion.get("annotation_type"),
        batch=conversion.get("batch"),
        scene_external_id=scene_external_id,
        iso_rotated_cuboids=conversion.get("iso_rotated_cuboids", False),
    )


class KognicAnnotationDownloader:
    def __init__(self, config: KognicDownloadConfig):
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

    def _resolve_scene_uuid(self, scene_external_id: str) -> str:
        inputs = self.kognic_io_client.input.query_inputs(
            project=self.config.project_external_id,
            external_ids=[scene_external_id],
        )
        scene_uuids = {i.scene_uuid for i in inputs if i.scene_uuid}
        if not scene_uuids:
            raise ValueError(
                f"No scene found with external_id={scene_external_id} "
                f"in project {self.config.project_external_id}"
            )
        if len(scene_uuids) > 1:
            raise ValueError(
                f"Multiple scenes ({len(scene_uuids)}) match external_id={scene_external_id} "
                f"in project {self.config.project_external_id}: {sorted(scene_uuids)}"
            )
        return scene_uuids.pop()

    def download_scene(self) -> None:
        scene_external_id = self.config.scene_external_id
        scene_uuid = self._resolve_scene_uuid(scene_external_id)
        logger.info(
            f"Fetching annotations for scene_external_id={scene_external_id} "
            f"(scene_uuid={scene_uuid}) in project {self.config.project_external_id}"
        )

        annotations = self.kognic_io_client.annotation.get_annotations_for_scene(
            scene_uuid=scene_uuid,
            iso_rotated_cuboids=self.config.iso_rotated_cuboids,
        )

        if not annotations:
            logger.warning(f"No annotations found for scene_external_id={scene_external_id}")
            return

        logger.info(f"Found {len(annotations)} annotation(s)")

        out_dir = self.config.output_base / self.config.project_external_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # A scene may carry multiple annotations (e.g. several requests); suffix
        # with request_uid so they don't overwrite each other.
        multiple = len(annotations) > 1
        for annotation in annotations:
            stem = (
                f"{scene_external_id}_{annotation.request_uid}"
                if multiple
                else scene_external_id
            )
            out_path = out_dir / f"{stem}.json"
            with open(out_path, "w") as f:
                json.dump(annotation.content, f, indent=2)
            logger.info(f"  Saved {out_path}")

        logger.info(
            f"Done. {len(annotations)} annotation(s) written to {self.config.output_base}"
        )

    def download(self) -> None:
        if self.config.scene_external_id:
            logger.info(
                f"Scene external ID specified ({self.config.scene_external_id}), "
                f"downloading annotations for that scene only."
            )
            self.download_scene()
        else:
            logger.info(
                f"No scene external ID specified, downloading all annotations for project "
                f"{self.config.project_external_id} with annotation type "
                f"{self.config.annotation_type} and batch {self.config.batch or '(all)'}"
                )
            self.download_all()

    def download_all(self) -> None:
        scope = (
            f"project={self.config.project_external_id}, "
            f"batch={self.config.batch or '(all)'}, "
            f"annotation_type={self.config.annotation_type}"
        )
        logger.info(f"Fetching annotations: {scope}")

        annotations = list(
            self.kognic_io_client.annotation.get_project_annotations(
                project=self.config.project_external_id,
                annotation_type=self.config.annotation_type,
                batch=self.config.batch,
                include_content=True,
                iso_rotated_cuboids=self.config.iso_rotated_cuboids,
            )
        )

        if not annotations:
            logger.warning(f"No annotations found for {scope}")
            return

        logger.info(f"Found {len(annotations)} annotation(s)")

        (self.config.output_base / self.config.project_external_id).mkdir(
            parents=True, exist_ok=True
        )

        for annotation in annotations:
            out_path = (
                self.config.output_base
                / self.config.project_external_id
                / f"{annotation.scene_uuid}.json"
            )
            with open(out_path, "w") as f:
                json.dump(annotation.content, f, indent=2)
            logger.info(f"  Saved {out_path}")

        logger.info(f"Done. {len(annotations)} annotation(s) written to {self.config.output_base}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/download_kognic_annotation_per_dataset_sample.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    assert (
        config_dict["task"] == "download_kognic_annotation"
    ), f"Expected task 'download_kognic_annotation', got '{config_dict['task']}'"

    logger.info(f"Loaded config from {args.config}")

    download_config = _load_download_config(config_dict)
    downloader = KognicAnnotationDownloader(download_config)

    logger.info("Starting download...")
    time_start = time.time()
    downloader.download()
    time_end = time.time()
    logger.info(f"Finished download in {time_end - time_start:.1f} seconds")



if __name__ == "__main__":
    main()
