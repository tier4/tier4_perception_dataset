import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional

from kognic.io.client import KognicIOClient
import yaml

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


@dataclass(frozen=True)
class KognicDownloadConfig:
    output_base: Path
    organization_id: str
    workspace_id: str
    project_external_id: str
    annotation_type: str
    batch: Optional[str] = None
    iso_rotated_cuboids: bool = False


def _load_download_config(config_dict: Dict) -> KognicDownloadConfig:
    conversion = config_dict["conversion"]
    organization_id = conversion.get("organization_id") or conversion.get("client_organization_id")
    workspace_id = conversion.get("workspace_id") or conversion.get("write_workspace_id")

    if not organization_id:
        raise ValueError("conversion.organization_id is required")
    if not workspace_id:
        raise ValueError("conversion.workspace_id is required")
    if not conversion.get("project_external_id"):
        raise ValueError("conversion.project_external_id is required")
    if not conversion.get("annotation_type"):
        raise ValueError("conversion.annotation_type is required")

    return KognicDownloadConfig(
        output_base=Path(conversion["output_base"]),
        organization_id=organization_id,
        workspace_id=workspace_id,
        project_external_id=conversion["project_external_id"],
        annotation_type=conversion["annotation_type"],
        batch=conversion.get("batch"),
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

        (self.config.output_base / self.config.project_external_id).mkdir(parents=True, exist_ok=True)


        for annotation in annotations:
            out_path = self.config.output_base / self.config.project_external_id / f"{annotation.scene_uuid}.json"
            with open(out_path, "w") as f:
                json.dump(annotation.content, f, indent=2)
            logger.info(f"  Saved {out_path}")

        logger.info(f"Done. {len(annotations)} annotation(s) written to {self.config.output_base}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/download_kognic_annotation_sample.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    assert config_dict["task"] == "download_kognic_annotation", (
        f"Expected task 'download_kognic_annotation', got '{config_dict['task']}'"
    )

    download_config = _load_download_config(config_dict)
    downloader = KognicAnnotationDownloader(download_config)
    downloader.download_all()


if __name__ == "__main__":
    main()
