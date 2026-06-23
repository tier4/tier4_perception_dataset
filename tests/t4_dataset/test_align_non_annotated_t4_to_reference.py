import json
from pathlib import Path
import shutil

import pytest
from t4_devkit import Tier4
import yaml

from perception_dataset import convert
from perception_dataset.t4_dataset.align_non_annotated_t4_to_reference import (
    AlignNonAnnotatedT4ToReferenceConverter,
)
from tests.constants import TEST_DATA_ROOT_DIR

TEST_DATASET_ROOT = TEST_DATA_ROOT_DIR / "t4_sample_0"
# Bases follow the same IO layout as the other T4 tasks: a parent directory
# containing one scene sub-directory each.
NON_ANNOTATED_BASE = TEST_DATASET_ROOT / "non_annotated_dataset"
ANNOTATED_BASE = TEST_DATASET_ROOT / "annotated_t4"
SCENE_NAME = "sample_bag"
ANNOTATED_T4 = ANNOTATED_BASE / SCENE_NAME / "t4_dataset"


def load_annotation_table(dataset_dir: Path, table_name: str) -> list[dict]:
    with (dataset_dir / "annotation" / f"{table_name}.json").open() as f:
        return json.load(f)


def assert_sample_lidar_keyframes_match(dataset_dir: Path) -> None:
    t4_dataset = Tier4(data_root=str(dataset_dir), verbose=False)

    for sample in t4_dataset.sample:
        lidar = t4_dataset.get("sample_data", sample.data["LIDAR_CONCAT"])
        assert sample.timestamp == lidar.timestamp
        assert lidar.is_key_frame


def test_align_non_annotated_t4_to_reference(tmp_path):
    output_base = tmp_path / "aligned"

    reports = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_BASE),
        reference_base=str(ANNOTATED_BASE),
        output_base=str(output_base),
        max_abs_diff_ms=1.0,
        copy_data=True,
    ).convert()

    output_dir = output_base / SCENE_NAME
    assert len(reports) == 1
    assert reports[0]["max_abs_timestamp_diff_us"] == 0
    assert reports[0]["interior_frame_drop_ratio"] == 0
    assert reports[0]["num_interior_dropped_keyframes"] == 0
    assert reports[0]["num_trimmed_leading_keyframes"] == 0
    assert reports[0]["num_trimmed_trailing_keyframes"] == 0
    assert len(reports[0]["alignment_results"]) == reports[0]["num_keyframes"]
    assert len(reports[0]["timestamp_diffs_us"]) == reports[0]["num_keyframes"]

    reference_samples = load_annotation_table(ANNOTATED_T4, "sample")
    output_samples = load_annotation_table(output_dir, "sample")
    output_sample_data = load_annotation_table(output_dir, "sample_data")
    output_sample_annotations = load_annotation_table(output_dir, "sample_annotation")

    assert len(output_samples) == len(reference_samples)
    reference_sample_annotations = load_annotation_table(ANNOTATED_T4, "sample_annotation")
    assert len(output_sample_annotations) == len(reference_sample_annotations)
    assert_sample_lidar_keyframes_match(output_dir)

    output_sample_tokens = {row["token"] for row in output_samples}
    assert {row["sample_token"] for row in output_sample_annotations} <= output_sample_tokens
    assert all(
        row["sample_token"] in output_sample_tokens
        for row in output_sample_data
        if row["is_key_frame"]
    )
    saved_report = json.loads((output_dir / "alignment_report.json").read_text())
    assert saved_report["alignment_results"] == reports[0]["alignment_results"]
    assert saved_report["unmatched_reference_results"] == []


def test_match_samples_by_timestamp_rejects_adjacent_10hz_frame():
    matches, unmatched = AlignNonAnnotatedT4ToReferenceConverter._match_samples_by_timestamp(
        candidate_samples=[
            {"timestamp": 1_100_000},
            {"timestamp": 2_000_000},
        ],
        reference_samples=[
            {"timestamp": 1_000_000},
            {"timestamp": 2_000_000},
        ],
        max_abs_diff_ms=1.0,
    )

    assert matches == [(1, 1, 0)]
    assert unmatched == [
        {
            "reference_index": 0,
            "reference_timestamp": 1_000_000,
            "candidate_index": 0,
            "candidate_timestamp": 1_100_000,
            "timestamp_diff_us": 100_000,
        }
    ]


def test_align_rejects_when_no_frames_matched(tmp_path, monkeypatch):
    monkeypatch.setattr(
        AlignNonAnnotatedT4ToReferenceConverter,
        "_match_samples_by_timestamp",
        staticmethod(lambda candidate_samples, reference_samples, *, max_abs_diff_ms: ([], [{}])),
    )
    converter = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_BASE),
        reference_base=str(ANNOTATED_BASE),
        output_base=str(tmp_path / "aligned"),
        max_abs_diff_ms=1.0,
    )
    with pytest.raises(RuntimeError, match="no matched keyframes"):
        converter.convert()


def test_align_rejects_when_interior_frame_drop_exceeds_ratio(tmp_path, monkeypatch):
    # Matches span reference indices 0..9 (span 10) with 8 interior gaps -> 80%.
    monkeypatch.setattr(
        AlignNonAnnotatedT4ToReferenceConverter,
        "_match_samples_by_timestamp",
        staticmethod(
            lambda candidate_samples, reference_samples, *, max_abs_diff_ms: (
                [(0, 0, 0), (9, 9, 0)],
                [{"reference_index": i} for i in range(1, 9)],
            )
        ),
    )
    converter = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_BASE),
        reference_base=str(ANNOTATED_BASE),
        output_base=str(tmp_path / "aligned"),
        max_abs_diff_ms=1.0,
        max_frame_drop_ratio=0.1,
    )
    with pytest.raises(RuntimeError, match="within the covered span"):
        converter.convert()


def test_classify_unmatched_splits_boundary_and_interior():
    matches = [(2, 0, 0), (5, 1, 0)]
    unmatched = [
        {"reference_index": 0},  # leading
        {"reference_index": 1},  # leading
        {"reference_index": 3},  # interior
        {"reference_index": 6},  # trailing
    ]
    leading, interior, trailing = AlignNonAnnotatedT4ToReferenceConverter._classify_unmatched(
        matches, unmatched
    )
    assert [row["reference_index"] for row in leading] == [0, 1]
    assert [row["reference_index"] for row in interior] == [3]
    assert [row["reference_index"] for row in trailing] == [6]


def test_align_non_annotated_t4_to_reference_preserves_lidar_info(tmp_path):
    input_base = tmp_path / "candidate_with_lidar_info"
    scene_dir = input_base / SCENE_NAME
    output_base = tmp_path / "aligned"
    shutil.copytree(NON_ANNOTATED_BASE / SCENE_NAME, scene_dir)

    sample_data_path = scene_dir / "annotation" / "sample_data.json"
    sample_data = load_annotation_table(scene_dir, "sample_data")
    for row in sample_data:
        if "/LIDAR_CONCAT/" not in row["filename"]:
            continue
        info_filename = row["filename"].replace("LIDAR_CONCAT", "LIDAR_CONCAT_INFO")
        info_filename = info_filename.replace(".pcd.bin", ".json")
        row["info_filename"] = info_filename
        info_path = scene_dir / info_filename
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with info_path.open("w") as f:
            json.dump({"source": row["filename"]}, f)
    with sample_data_path.open("w") as f:
        json.dump(sample_data, f, indent=2)

    AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(input_base),
        reference_base=str(ANNOTATED_BASE),
        output_base=str(output_base),
        max_abs_diff_ms=1.0,
        copy_data=True,
    ).convert()

    output_dir = output_base / SCENE_NAME
    output_sample_data = load_annotation_table(output_dir, "sample_data")
    lidar_info_rows = [row for row in output_sample_data if row.get("info_filename")]

    assert lidar_info_rows
    assert all((output_dir / row["info_filename"]).exists() for row in lidar_info_rows)
    assert_sample_lidar_keyframes_match(output_dir)


def test_align_non_annotated_t4_to_reference_can_skip_report(tmp_path):
    output_base = tmp_path / "aligned"

    reports = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_BASE),
        reference_base=str(ANNOTATED_BASE),
        output_base=str(output_base),
        max_abs_diff_ms=1.0,
        copy_data=True,
        write_alignment_report=False,
    ).convert()

    assert reports[0]["alignment_results"]
    assert not (output_base / SCENE_NAME / "alignment_report.json").exists()


def test_align_non_annotated_t4_to_reference_convert_task(tmp_path, monkeypatch):
    config_path = tmp_path / "align_non_annotated_t4_to_reference.yaml"
    output_base = tmp_path / "aligned_from_config"
    with config_path.open("w") as f:
        yaml.safe_dump(
            {
                "task": "align_non_annotated_t4_to_reference",
                "conversion": {
                    "input_base": str(NON_ANNOTATED_BASE),
                    "reference_base": str(ANNOTATED_BASE),
                    "output_base": str(output_base),
                    "max_abs_diff_ms": 1.0,
                    "copy_data": True,
                    "write_alignment_report": False,
                },
            },
            f,
        )

    monkeypatch.setattr("sys.argv", ["convert", "--config", str(config_path)])

    convert.main()

    output_dir = output_base / SCENE_NAME
    assert_sample_lidar_keyframes_match(output_dir)
    assert not (output_dir / "alignment_report.json").exists()
