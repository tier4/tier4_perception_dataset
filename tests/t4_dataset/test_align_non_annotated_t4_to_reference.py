import json
from pathlib import Path
import shutil

import numpy as np
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
    ).convert()

    output_dir = output_base / SCENE_NAME
    output_sample_data = load_annotation_table(output_dir, "sample_data")
    lidar_info_rows = [row for row in output_sample_data if row.get("info_filename")]

    assert lidar_info_rows
    assert all((output_dir / row["info_filename"]).exists() for row in lidar_info_rows)
    assert_sample_lidar_keyframes_match(output_dir)


def test_vote_point_labels_uses_surrounding_point_classes():
    # Two well-separated clusters with different classes and a different
    # number of candidate points than reference points.
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=(0.0, 0.0, 0.0), scale=0.05, size=(40, 3))
    cluster_b = rng.normal(loc=(10.0, 0.0, 0.0), scale=0.05, size=(25, 3))
    reference_points = np.vstack([cluster_a, cluster_b])
    reference_labels = np.array([3] * 40 + [7] * 25, dtype=np.uint8)

    candidate_points = np.vstack(
        [
            rng.normal(loc=(0.0, 0.0, 0.0), scale=0.05, size=(11, 3)),
            rng.normal(loc=(10.0, 0.0, 0.0), scale=0.05, size=(6, 3)),
            [[100.0, 100.0, 100.0]],  # no reference point nearby -> unpainted
        ]
    )

    labels, stats = AlignNonAnnotatedT4ToReferenceConverter._vote_point_labels(
        reference_points,
        reference_labels,
        candidate_points,
        num_neighbors=5,
        max_neighbor_dist_m=1.0,
    )

    assert labels.dtype == np.uint8
    assert labels[:11].tolist() == [3] * 11
    assert labels[11:17].tolist() == [7] * 6
    assert labels[17] == 0
    assert stats["num_points"] == 18
    assert stats["num_out_of_range_points"] == 1
    assert stats["max_nearest_neighbor_dist_m"] > 1.0


def test_vote_point_labels_prefers_coincident_point():
    # The candidate point sits exactly on a reference point of class 2 while
    # being surrounded by more numerous but farther class-9 points: the
    # inverse-distance weighting must let the coincident point win the vote.
    reference_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, -0.3, 0.0],
        ]
    )
    reference_labels = np.array([2, 9, 9, 9, 9], dtype=np.uint8)

    labels, _ = AlignNonAnnotatedT4ToReferenceConverter._vote_point_labels(
        reference_points,
        reference_labels,
        np.array([[0.0, 0.0, 0.0]]),
        num_neighbors=5,
        max_neighbor_dist_m=1.0,
    )

    assert labels.tolist() == [2]


def add_lidarseg_to_reference(reference_scene_dir: Path) -> dict[str, int]:
    """Attach synthetic lidarseg annotations to every LIDAR_CONCAT keyframe of
    a reference scene, labeling every point of frame ``i`` with class
    ``(i % 5) + 1``. Returns the expected class per lidar sample_data token.
    """
    sample_data = load_annotation_table(reference_scene_dir, "sample_data")
    lidarseg_dir = reference_scene_dir / "lidarseg" / "annotation"
    lidarseg_dir.mkdir(parents=True)

    lidarseg_rows = []
    expected_class_by_token: dict[str, int] = {}
    lidar_rows = sorted(
        (row for row in sample_data if "/LIDAR_CONCAT/" in row["filename"] and row["is_key_frame"]),
        key=lambda row: row["timestamp"],
    )
    for index, row in enumerate(lidar_rows):
        num_points = (reference_scene_dir / row["filename"]).stat().st_size // (4 * 5)
        label = (index % 5) + 1
        token = f"lidarsegtoken{index:03d}"
        filename = f"lidarseg/annotation/{token}.bin"
        np.full(num_points, label, dtype=np.uint8).tofile(reference_scene_dir / filename)
        lidarseg_rows.append(
            {"token": token, "sample_data_token": row["token"], "filename": filename}
        )
        expected_class_by_token[row["token"]] = label

    with (reference_scene_dir / "annotation" / "lidarseg.json").open("w") as f:
        json.dump(lidarseg_rows, f, indent=2)
    return expected_class_by_token


def test_align_transfers_lidarseg_when_reference_has_lidarseg(tmp_path):
    reference_base = tmp_path / "reference_with_lidarseg"
    shutil.copytree(ANNOTATED_BASE, reference_base)
    add_lidarseg_to_reference(reference_base / SCENE_NAME / "t4_dataset")
    output_base = tmp_path / "aligned"

    reports = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_BASE),
        reference_base=str(reference_base),
        output_base=str(output_base),
        max_abs_diff_ms=1.0,
    ).convert()

    output_dir = output_base / SCENE_NAME
    output_lidarseg = load_annotation_table(output_dir, "lidarseg")
    output_sample_data = load_annotation_table(output_dir, "sample_data")
    lidar_keyframes = {
        row["token"]: row
        for row in output_sample_data
        if "/LIDAR_CONCAT/" in row["filename"] and row["is_key_frame"]
    }

    # One lidarseg record per matched lidar keyframe, each pointing at a
    # candidate sample_data with a label bin matching its point count.
    assert len(output_lidarseg) == reports[0]["num_keyframes"]
    lidar_keyframes_by_timestamp = sorted(lidar_keyframes.values(), key=lambda r: r["timestamp"])
    for index, row in enumerate(output_lidarseg):
        sample_data_row = lidar_keyframes[row["sample_data_token"]]
        labels = np.fromfile(output_dir / row["filename"], dtype=np.uint8)
        num_points = (output_dir / sample_data_row["filename"]).stat().st_size // (4 * 5)
        assert labels.shape[0] == num_points
        # The candidate clouds equal the reference clouds in this fixture, so
        # every point must inherit the frame's constant reference class.
        expected_label = (
            lidar_keyframes_by_timestamp.index(sample_data_row) % 5
        ) + 1
        assert set(labels.tolist()) == {expected_label}

    summary = reports[0]["lidarseg"]
    assert summary["num_records"] == len(output_lidarseg)
    assert summary["num_out_of_range_points"] == 0
    assert len(summary["frames"]) == len(output_lidarseg)
    assert_sample_lidar_keyframes_match(output_dir)


def test_align_without_reference_lidarseg_writes_no_lidarseg(tmp_path):
    output_base = tmp_path / "aligned"

    reports = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_BASE),
        reference_base=str(ANNOTATED_BASE),
        output_base=str(output_base),
        max_abs_diff_ms=1.0,
    ).convert()

    output_dir = output_base / SCENE_NAME
    assert not (output_dir / "annotation" / "lidarseg.json").exists()
    assert not (output_dir / "lidarseg").exists()
    assert reports[0]["lidarseg"] is None


def test_align_non_annotated_t4_to_reference_can_skip_report(tmp_path):
    output_base = tmp_path / "aligned"

    reports = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(NON_ANNOTATED_BASE),
        reference_base=str(ANNOTATED_BASE),
        output_base=str(output_base),
        max_abs_diff_ms=1.0,
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
