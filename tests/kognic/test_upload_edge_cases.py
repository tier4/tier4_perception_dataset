import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from perception_dataset.kognic.upload_dataset import (
    KognicDatasetUploader,
    KognicUploadConfig,
    ProjectTarget,
    SceneInputError,
    SceneUploadResult,
    _exception_report_fields,
    _result_report_rows,
    _validate_sensor_file_counts,
    _validate_sensor_timestamps,
    _wait_for_pre_annotation,
    _write_upload_report,
    read_upload_report_scene_uuids,
)


def _touch_sensor_files(root: Path, relative_dir: str, timestamps: list[int]) -> list[Path]:
    sensor_dir = root / relative_dir
    sensor_dir.mkdir(parents=True)
    files = []
    for timestamp in timestamps:
        path = sensor_dir / f"{timestamp}.csv"
        path.touch()
        files.append(path)
    return files


def test_sensor_count_validation_rejects_partial_staging(tmp_path: Path):
    """Test that staging is rejected when sensor file counts disagree.

    LiDAR contains two frames while the camera contains only one. Since the
    uploader pairs files by their sorted position, accepting this directory
    would either omit the camera or shift later sensor data onto the wrong frame.

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


def test_sensor_timestamp_validation_rejects_shifted_stream(tmp_path: Path):
    """Test that a sensor file far from its anchor timestamp is rejected.

    The middle camera timestamp is shifted beyond the allowed part of one frame
    interval. This catches stale or mixed staging files that have equal counts
    but would still pair camera and LiDAR data from different frames.

    Args:
        tmp_path (Path): Pytest directory used as the sequence path in the
            validation error.
    """
    anchor = [Path("100.csv"), Path("200.csv"), Path("300.csv")]
    sensor_files = {
        "LIDAR_FRONT": anchor,
        "CAM_FRONT": [Path("100.jpg"), Path("291.jpg"), Path("300.jpg")],
    }

    with pytest.raises(ValueError, match="misaligned at frame 1"):
        _validate_sensor_timestamps(tmp_path, sensor_files, anchor)


def test_iterate_frames_requires_every_sensor_in_every_frame(tmp_path: Path):
    """Test that frame iteration stops when one sensor has a missing frame.

    The staged sequence has two LiDAR files and one camera file. The public
    iteration path must run the count validation before yielding frames so an
    incomplete scene is never sent to Kognic.

    Args:
        tmp_path (Path): Pytest directory used to build the incomplete staging
            sequence.
    """
    _touch_sensor_files(tmp_path, "lidar/LIDAR_FRONT", [100, 200])
    camera_dir = tmp_path / "cameras/CAM_FRONT"
    camera_dir.mkdir(parents=True)
    (camera_dir / "100.jpg").touch()

    uploader = KognicDatasetUploader(KognicUploadConfig(input_base=tmp_path))

    with pytest.raises(ValueError, match="Sensor file counts disagree"):
        list(uploader.iterate_frames(tmp_path))


def test_keyframe_metadata_is_required(tmp_path: Path):
    """Test that upload staging without keyframe metadata is rejected.

    The uploader must not silently mark every frame for annotation when a
    staging directory was produced without ``keyframes.json``. Requiring the
    file keeps annotation flags aligned with the source T4 keyframes.

    Args:
        tmp_path (Path): Pytest directory representing a staging sequence
            without keyframe metadata.
    """
    with pytest.raises(FileNotFoundError, match="Required keyframe metadata is missing"):
        KognicDatasetUploader._load_keyframe_indices(tmp_path, frame_count=2)


def test_keyframe_metadata_must_match_staging_frame_count(tmp_path: Path):
    """Test that stale keyframe metadata cannot annotate changed staging data.

    The metadata claims it was generated from three frames while the caller
    has discovered two current staging frames. Accepting it could mark the
    wrong sensor data for annotation, so the uploader must fail instead.

    Args:
        tmp_path (Path): Pytest directory used to hold deliberately stale
            ``keyframes.json`` metadata.
    """
    (tmp_path / "keyframes.json").write_text(
        json.dumps({"frame_count": 3, "keyframe_indices": [0, 2]})
    )

    with pytest.raises(ValueError, match="generated for 3 frames.*has 2"):
        KognicDatasetUploader._load_keyframe_indices(tmp_path, frame_count=2)


def test_upload_validates_frames_before_uploading_calibration(tmp_path: Path):
    """Test that invalid staging creates no remote calibration resource.

    Frame construction raises the same missing-metadata error that a real
    staging sequence would produce. The calibration uploader must remain
    uncalled, proving local staging validation completes before any remote
    resource can be created.

    Args:
        tmp_path (Path): Pytest directory used as the uploader input base.
    """
    uploader = KognicDatasetUploader(KognicUploadConfig(input_base=tmp_path))
    uploader._load_ego_poses = Mock(return_value=None)
    uploader._build_frames = Mock(side_effect=FileNotFoundError("missing keyframes.json"))
    uploader._get_or_upload_calibration = Mock()

    with pytest.raises(FileNotFoundError, match="missing keyframes.json"):
        uploader.upload_one(tmp_path, "scene")

    uploader._get_or_upload_calibration.assert_not_called()


def test_wait_for_pre_annotation_reaches_indexed(monkeypatch: pytest.MonkeyPatch):
    """Test that polling continues from processing to the indexed state.

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
        [{"status": "Indexed", "id": "pre-1"}],
    ]
    monkeypatch.setattr("perception_dataset.kognic.upload_dataset.time.sleep", lambda _: None)

    result = _wait_for_pre_annotation(
        SimpleNamespace(pre_annotation=pre_annotation), "pre-1", timeout_s=10, poll_s=0
    )

    assert result["status"] == "Indexed"


@pytest.mark.parametrize("status", ["failed", "mystery-state"])
def test_wait_for_pre_annotation_rejects_terminal_failure(status: str):
    """Test that failed and unknown pre-annotation states raise an error.

    A known server failure and an undocumented state are both unsafe to attach
    to an input. Raising prevents the uploader from treating either response as
    a successful pre-annotation.

    Args:
        status (str): Terminal or unknown status supplied by the parametrized
            test.
    """
    client = SimpleNamespace(
        pre_annotation=SimpleNamespace(list=lambda **_: [{"status": status, "details": "bad"}])
    )

    with pytest.raises(RuntimeError, match="failed server-side|unrecognized status"):
        _wait_for_pre_annotation(client, "pre-1", timeout_s=0, poll_s=0)


def test_wait_for_pre_annotation_does_not_attach_after_timeout():
    """Test that strict polling raises when processing exceeds the deadline.

    Kognic continues to report ``processing`` after an immediate timeout. The
    uploader must raise instead of attaching an unfinished pre-annotation,
    which Kognic could later reject without creating the expected input.
    """
    client = SimpleNamespace(
        pre_annotation=SimpleNamespace(list=lambda **_: [{"status": "processing"}])
    )

    with pytest.raises(TimeoutError, match="not attaching it to an input"):
        _wait_for_pre_annotation(
            client, "pre-1", timeout_s=0, poll_s=0, raise_on_timeout=True
        )


def test_upload_pre_annotations_waits_for_index_before_returning(tmp_path: Path):
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
        list=lambda **_: calls.append("status") or [{"status": "indexed"}],
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
