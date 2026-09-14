import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydantic import ValidationError
import pytest

from perception_dataset.kognic.sequence_artifact import SEQUENCE_ARTIFACT_FILENAME
from perception_dataset.kognic.t4_to_kognic_converter import (
    T4ToKognicConverter,
    _validate_sensor_file_counts,
    _validate_sensor_timestamps,
)
from perception_dataset.kognic.upload_dataset import (
    KognicDatasetUploader,
    KognicUploadConfig,
    ProjectTarget,
    SceneInputError,
    SceneUploadResult,
    _exception_report_fields,
    _result_report_rows,
    _wait_for_pre_annotation,
    _write_upload_report,
    main,
    read_upload_report_scene_uuids,
)


def test_sensor_count_validation_rejects_partial_staging(tmp_path: Path):
    """Test that an incomplete in-memory resource manifest is rejected.

    LiDAR contains two frames while the camera contains only one. Since the
    converter builds each frame by resource-list index, accepting this manifest
    would omit the camera from the second frame.

    Args:
        tmp_path (Path): Pytest directory used as the sequence path in the
            validation error.
    """
    files = {
        "LIDAR_FRONT": [Path("1.csv"), Path("2.csv")],
        "CAM_FRONT": [Path("1.jpg")],
    }

    with pytest.raises(ValueError, match="Sensor file counts disagree"):
        _validate_sensor_file_counts(tmp_path, files, "LIDAR_FRONT", expected_count=2)


def test_sensor_count_validation_rejects_duplicate_output_paths(tmp_path: Path):
    """Test that two frames cannot reference one overwritten sensor resource."""
    duplicate = Path("100.csv")
    files = {"LIDAR_FRONT": [duplicate, duplicate]}

    with pytest.raises(ValueError, match="duplicate paths"):
        _validate_sensor_file_counts(tmp_path, files, "LIDAR_FRONT", expected_count=2)


def test_sensor_timestamp_validation_rejects_shifted_stream(tmp_path: Path):
    """Test that a sensor file far from its anchor timestamp is rejected.

    The middle camera timestamp is shifted beyond the allowed part of one frame
    interval. This catches stale or mixed staging files that have equal counts
    but would still pair camera and LiDAR data from different frames.

    Args:
        tmp_path (Path): Pytest directory used as the sequence path in the
            validation error.
    """
    frame_timestamps_ns = [100, 200, 300]
    sensor_files = {
        "LIDAR_FRONT": [Path("100.csv"), Path("200.csv"), Path("300.csv")],
        "CAM_FRONT": [Path("100.jpg"), Path("291.jpg"), Path("300.jpg")],
    }

    with pytest.raises(ValueError, match="misaligned at frame 1"):
        _validate_sensor_timestamps(tmp_path, sensor_files, frame_timestamps_ns)


def test_upload_validates_sequence_artifact_before_uploading_calibration(tmp_path: Path):
    """Test that invalid sequence JSON creates no remote calibration resource.

    Args:
        tmp_path (Path): Pytest directory used as the uploader input base.
    """
    uploader = KognicDatasetUploader(KognicUploadConfig(input_base=tmp_path))
    uploader._get_or_upload_calibration = Mock()
    (tmp_path / SEQUENCE_ARTIFACT_FILENAME).write_text("{}")

    with pytest.raises(ValidationError):
        uploader.upload_one(tmp_path, "scene")

    uploader._get_or_upload_calibration.assert_not_called()


def test_upload_only_loads_sequence_built_during_conversion(tmp_path: Path):
    """Test that upload reuses the validated sequence artifact without rebuilding it."""
    lidar_path = tmp_path / "lidar" / "LIDAR_FRONT" / "100.csv"
    lidar_path.parent.mkdir(parents=True)
    lidar_path.write_text("ts_gps,x,y,z,intensity\n100,0,0,0,1\n")
    converter = T4ToKognicConverter(
        str(tmp_path),
        str(tmp_path),
        camera_sensors=[],
        include_imu_data=False,
    )
    converter._lidar_channels = ["LIDAR_FRONT"]
    converter._write_sequence_artifact(
        tmp_path,
        {},
        {"LIDAR_FRONT": [lidar_path]},
        [100],
        [0],
    )

    payload = json.loads((tmp_path / SEQUENCE_ARTIFACT_FILENAME).read_text())
    assert payload["frames"][0]["point_clouds"][0]["filename"] == ("lidar/LIDAR_FRONT/100.csv")

    uploader = KognicDatasetUploader(KognicUploadConfig(input_base=tmp_path))
    uploader._get_or_upload_calibration = Mock(return_value="calibration-id")
    uploader._upload_scene = Mock(return_value=("scene-id", [], [], []))

    result = uploader.upload_one(tmp_path, "scene")

    uploaded_scene = uploader._upload_scene.call_args.args[0]
    assert uploaded_scene.calibration_id == "calibration-id"
    assert uploaded_scene.frames[0].metadata.annotate
    assert result[0].scene_uuid == "scene-id"


def test_upload_main_skips_scene_with_missing_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that batch upload logs and skips a scene with a missing resource."""
    sequence_dir = tmp_path / "scene"
    sequence_dir.mkdir()
    lidar_path = sequence_dir / "lidar" / "LIDAR_FRONT" / "100.csv"
    lidar_path.parent.mkdir(parents=True)
    lidar_path.write_text("ts_gps,x,y,z,intensity\n100,0,0,0,1\n")

    converter = T4ToKognicConverter(
        str(sequence_dir),
        str(sequence_dir),
        camera_sensors=[],
        include_imu_data=False,
    )
    converter._lidar_channels = ["LIDAR_FRONT"]
    converter._write_sequence_artifact(
        sequence_dir,
        {},
        {"LIDAR_FRONT": [lidar_path]},
        [100],
        [0],
    )
    lidar_path.unlink()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task: upload_kognic_dataset",
                "conversion:",
                f"  input_base: {tmp_path.as_posix()}",
                "  projects: []",
                "  dryrun: true",
                "  generate_tsv_report: false",
            ]
        )
    )

    monkeypatch.setattr("sys.argv", ["upload_dataset", "--config", str(config_path)])

    with pytest.raises(SystemExit, match="scene upload\\(s\\) failed"):
        main()


def test_wait_for_pre_annotation_reaches_available(monkeypatch: pytest.MonkeyPatch):
    """Test that polling continues from processing to the available state.

    Upload creation only queues server-side processing. The helper must wait
    through the temporary ``processing`` response and return only after Kognic
    reports that the pre-annotation is ready for an input.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the polling sleep so the test
            completes immediately.
    """
    pre_annotation = Mock()
    pre_annotation.list.side_effect = [
        [{"status": "processing"}],
        [{"status": "Available", "id": "pre-1"}],
    ]
    monkeypatch.setattr("perception_dataset.kognic.upload_dataset.time.sleep", lambda _: None)

    result = _wait_for_pre_annotation(
        SimpleNamespace(pre_annotation=pre_annotation), "pre-1", timeout_s=10, poll_s=0
    )

    assert result["status"] == "Available"


@pytest.mark.parametrize("status", ["failed", "mystery-state"])
def test_wait_for_pre_annotation_waits_for_success(status: str):
    """Continue polling every non-success status until availability."""
    pre_annotation = Mock()
    pre_annotation.list.side_effect = [
        [{"status": status}],
        [{"status": "available", "id": "pre-1"}],
    ]
    result = _wait_for_pre_annotation(
        SimpleNamespace(pre_annotation=pre_annotation), "pre-1", poll_s=0
    )
    assert result["id"] == "pre-1"
    assert pre_annotation.list.call_count == 2


@pytest.mark.parametrize("pending_status", ["processing", "indexed", "failed", "mystery-state"])
def test_wait_for_pre_annotation_does_not_attach_after_timeout(pending_status: str):
    """Test that strict polling raises when processing exceeds the deadline.

    Kognic continues to report a known non-ready status after an immediate
    timeout. The uploader must raise instead of attaching an unfinished
    pre-annotation, which Kognic could later reject without creating the
    expected input.
    """
    client = SimpleNamespace(
        pre_annotation=SimpleNamespace(list=lambda **_: [{"status": pending_status}])
    )

    with pytest.raises(TimeoutError, match="not attaching it to an input"):
        _wait_for_pre_annotation(client, "pre-1", timeout_s=0, poll_s=0, raise_on_timeout=True)


def test_upload_pre_annotations_waits_for_availability_before_returning(tmp_path: Path):
    """Test that uploading checks the server status before returning an ID.

    The create call returns ``pre-1``, but that response alone does not mean the
    annotation is usable. The status call must occur before the identifier is
    returned to the later input-creation step.

    Args:
        tmp_path (Path): Pytest directory used as the uploader input base.
    """
    calls = []
    pre_annotation_api = SimpleNamespace(
        create=lambda **_: SimpleNamespace(id="pre-1"),
        list=lambda **_: calls.append("status") or [{"status": "available"}],
    )
    uploader = KognicDatasetUploader(
        KognicUploadConfig(
            input_base=tmp_path,
            pre_annotation_timeout_s=0,
            pre_annotation_poll_interval_s=0,
        )
    )
    uploader._kognic_io_client = SimpleNamespace(pre_annotation=pre_annotation_api)

    result = uploader._upload_pre_annotations("scene-1", "scene", {"pre.json": object()})

    assert result == {"pre.json": "pre-1"}
    assert calls == ["status"]


def test_upload_report_preserves_nested_http_error_and_filters_scene_ids(tmp_path: Path):
    """Test that reports keep nested HTTP details and return safe recovery IDs.

    The original HTTP 422 error is wrapped by a scene-level error. The report
    must preserve that leaf error while recovery returns each successful scene
    UUID once and excludes dry-run placeholders.

    Args:
        tmp_path (Path): Pytest directory where the TSV report is written.
    """
    http_error = RuntimeError("server rejected input")
    http_error.response = SimpleNamespace(status_code=422)
    wrapped = SceneInputError(
        "scene",
        "scene-1",
        "input creation",
        http_error,
        invalidated=False,
    )
    fields = _exception_report_fields(wrapped)
    assert fields == {
        "stage": "input creation",
        "error_code": "422",
        "error_type": "RuntimeError",
        "error_message": "server rejected input",
    }

    rows = _result_report_rows(
        "nested/scene",
        SceneUploadResult("scene", "scene-1", failed=True, error=wrapped),
        duration_seconds=1.25,
    )
    rows.extend(
        [
            {
                **rows[0],
                "status": "successful",
                "scene_uuid": "scene-2",
            },
            {
                **rows[0],
                "status": "successful",
                "scene_uuid": "scene-2",
            },
            {
                **rows[0],
                "status": "dryrun_successful",
                "scene_uuid": "dryrun",
            },
        ]
    )
    report_path = _write_upload_report(tmp_path, rows)

    assert read_upload_report_scene_uuids(report_path, {"successful"}) == ["scene-2"]
    with open(report_path, newline="", encoding="utf-8") as report_file:
        saved = list(csv.DictReader(report_file, dialect="excel-tab"))
    assert saved[0]["scene"] == "nested/scene"
    assert saved[0]["error_code"] == "422"


def test_upload_report_rejects_missing_required_columns(tmp_path: Path):
    """Test that recovery rejects a report without the scene UUID column.

    A malformed TSV contains statuses but cannot identify the corresponding
    Kognic scenes. Raising a clear error prevents cleanup or retry tools from
    silently operating on incomplete report data.

    Args:
        tmp_path (Path): Pytest directory where the malformed report is created.
    """
    report = tmp_path / "upload_report.tsv"
    report.write_text("status\ncomplete\n")

    with pytest.raises(ValueError, match="scene_uuid"):
        read_upload_report_scene_uuids(report, {"complete"})
