import json
import shutil
from pathlib import Path

import yaml
from t4_devkit import Tier4

from perception_dataset import convert
from perception_dataset.t4_dataset.align_non_annotated_t4_to_reference import (
    AlignNonAnnotatedT4ToReferenceConverter,
    match_reference_samples_by_timestamp,
)
from tests.constants import TEST_DATA_ROOT_DIR


TEST_DATASET_ROOT = TEST_DATA_ROOT_DIR / "t4_sample_0"
NON_ANNOTATED_T4 = TEST_DATASET_ROOT / "non_annotated_dataset" / "sample_bag"
ANNOTATED_T4 = TEST_DATASET_ROOT / "annotated_t4" / "sample_bag" / "t4_dataset"


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
    output_dir = tmp_path / "aligned"

    reports = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_T4),
        reference_base=str(ANNOTATED_T4),
        output_base=str(output_dir),
        copy_data=True,
        overwrite_mode=True,
    ).convert()

    assert len(reports) == 1
    assert reports[0]["max_abs_timestamp_diff_us"] == 0
    assert len(reports[0]["alignment_results"]) == reports[0]["num_keyframes"]
    assert len(reports[0]["timestamp_diffs_us"]) == reports[0]["num_keyframes"]
    assert "first_20_timestamp_diffs_us" not in reports[0]
    assert "first_20_key_candidate_indices" not in reports[0]

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


def test_match_reference_samples_by_timestamp_rejects_adjacent_10hz_frame():
    matches, unmatched = match_reference_samples_by_timestamp(
        candidate_samples=[
            {"timestamp": 1_100_000},
            {"timestamp": 2_000_000},
        ],
        reference_samples=[
            {"timestamp": 1_000_000},
            {"timestamp": 2_000_000},
        ],
        candidate_start=0,
        reference_start=0,
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


def test_align_non_annotated_t4_to_reference_preserves_lidar_info(tmp_path):
    input_dir = tmp_path / "candidate_with_lidar_info"
    output_dir = tmp_path / "aligned"
    shutil.copytree(NON_ANNOTATED_T4, input_dir)

    sample_data_path = input_dir / "annotation" / "sample_data.json"
    sample_data = load_annotation_table(input_dir, "sample_data")
    for row in sample_data:
        if "/LIDAR_CONCAT/" not in row["filename"]:
            continue
        info_filename = row["filename"].replace("LIDAR_CONCAT", "LIDAR_CONCAT_INFO")
        info_filename = info_filename.replace(".pcd.bin", ".json")
        row["info_filename"] = info_filename
        info_path = input_dir / info_filename
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with info_path.open("w") as f:
            json.dump({"source": row["filename"]}, f)
    with sample_data_path.open("w") as f:
        json.dump(sample_data, f, indent=2)

    AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(input_dir),
        reference_base=str(ANNOTATED_T4),
        output_base=str(output_dir),
        copy_data=True,
        overwrite_mode=True,
    ).convert()

    output_sample_data = load_annotation_table(output_dir, "sample_data")
    lidar_info_rows = [row for row in output_sample_data if row.get("info_filename")]

    assert lidar_info_rows
    assert all((output_dir / row["info_filename"]).exists() for row in lidar_info_rows)
    assert_sample_lidar_keyframes_match(output_dir)


def test_align_non_annotated_t4_to_reference_can_skip_report(tmp_path):
    output_dir = tmp_path / "aligned"

    reports = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_T4),
        reference_base=str(ANNOTATED_T4),
        output_base=str(output_dir),
        copy_data=True,
        write_alignment_report=False,
        overwrite_mode=True,
    ).convert()

    assert reports[0]["alignment_results"]
    assert not (output_dir / "alignment_report.json").exists()


def test_align_non_annotated_t4_to_reference_parent_directories(tmp_path):
    output_base = tmp_path / "aligned_parent"

    reports = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(TEST_DATASET_ROOT / "non_annotated_dataset"),
        reference_base=str(TEST_DATASET_ROOT / "annotated_t4"),
        output_base=str(output_base),
        copy_data=True,
        overwrite_mode=True,
    ).convert()

    output_dir = output_base / "sample_bag"
    assert len(reports) == 1
    assert output_dir.exists()
    assert_sample_lidar_keyframes_match(output_dir)


def test_align_non_annotated_t4_to_reference_convert_task(tmp_path, monkeypatch):
    config_path = tmp_path / "align_non_annotated_t4_to_reference.yaml"
    output_dir = tmp_path / "aligned_from_config"
    with config_path.open("w") as f:
        yaml.safe_dump(
            {
                "task": "align_non_annotated_t4_to_reference",
                "conversion": {
                    "input_base": str(NON_ANNOTATED_T4),
                    "reference_base": str(ANNOTATED_T4),
                    "output_base": str(output_dir),
                    "copy_data": True,
                    "write_alignment_report": False,
                },
            },
            f,
        )

    monkeypatch.setattr(
        "sys.argv",
        ["convert", "--config", str(config_path), "--overwrite"],
    )

    convert.main()

    assert_sample_lidar_keyframes_match(output_dir)
    assert not (output_dir / "alignment_report.json").exists()
