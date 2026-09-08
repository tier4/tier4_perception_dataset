import json
from pathlib import Path
from types import SimpleNamespace

from perception_dataset.kognic.download_annotation import (
    KognicAnnotationDownloader,
    KognicDownloadConfig,
)


def test_download_all_preserves_multiple_requests_for_one_scene(tmp_path: Path):
    """Test that multiple annotation requests for one scene get unique files.

    The first scene has separate cuboid and segmentation requests. Their request
    IDs must be added to the filenames so both payloads survive, while a scene
    with only one request keeps the simpler scene-based filename.

    Args:
        tmp_path (Path): Pytest directory where downloaded JSON files are written.
    """
    annotations = [
        SimpleNamespace(scene_uuid="scene-1", request_uid="cuboid", content={"kind": "box"}),
        SimpleNamespace(scene_uuid="scene-1", request_uid="semseg", content={"kind": "mask"}),
        SimpleNamespace(scene_uuid="scene-2", request_uid="only", content={"kind": "box"}),
    ]
    annotation_api = SimpleNamespace(
        get_project_annotations=lambda **_: iter(annotations)
    )
    downloader = KognicAnnotationDownloader(
        KognicDownloadConfig(
            output_base=tmp_path,
            organization_id="organization",
            workspace_id="workspace",
            project_external_id="project",
            annotation_type="all",
        )
    )
    downloader._kognic_io_client = SimpleNamespace(annotation=annotation_api)

    downloader.download_all()

    output = tmp_path / "project"
    assert {path.name for path in output.iterdir()} == {
        "scene-1_cuboid.json",
        "scene-1_semseg.json",
        "scene-2.json",
    }
    assert json.loads((output / "scene-1_cuboid.json").read_text()) == {"kind": "box"}
    assert json.loads((output / "scene-1_semseg.json").read_text()) == {"kind": "mask"}
