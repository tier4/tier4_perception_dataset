"""Synthetic-dataset tests for the annotation/lidar consistency health check."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from perception_dataset.t4_dataset.annotation_health import compare_points_preserved, compare_versions, estimate_annotation_offset

SPEED = 2.0  # m/s along +x
NUM_KF = 8


def _write_dataset(root: Path, *, annotation_shift: int = 0, cloud_scale: int = 1, extra_field: bool = False, ground_patch_ahead: bool = False) -> None:
    """A stationary ego, one moving object (dense cluster) and one static cluster, 1 Hz keyframes.

    ``annotation_shift`` attaches the box that describes keyframe k to sample k + shift (a shift of 2
    reproduces the bug where annotations were copied by index onto frames cut 2 s later).
    """
    ann = root / "annotation"
    ann.mkdir(parents=True)
    (root / "data" / "LIDAR_CONCAT").mkdir(parents=True)
    rng = np.random.default_rng(0)
    nfields = 7 if extra_field else 5
    samples, sample_data, ego_poses, annotations = [], [], [], []
    tokens = [f"s{k}" for k in range(NUM_KF)]
    for k in range(NUM_KF):
        ts = 1_000_000 * (k + 1)
        samples.append({"token": tokens[k], "timestamp": ts, "scene_token": "scene", "prev": tokens[k - 1] if k else "", "next": tokens[k + 1] if k + 1 < NUM_KF else ""})
        ego_poses.append({"token": f"e{k}", "timestamp": ts, "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]})
        # moving object at x = 5 + SPEED * k, static object at (-4, 6)
        obj_z = (-1.5, -0.3) if ground_patch_ahead else (0.2, 1.6)
        obj = np.column_stack([rng.normal(5 + SPEED * k, 0.15, 400 * cloud_scale), rng.normal(3.0, 0.15, 400 * cloud_scale), rng.uniform(obj_z[0], obj_z[1], 400 * cloud_scale)])
        static = np.column_stack([rng.normal(-4.0, 0.3, 300), rng.normal(6.0, 0.3, 300), rng.uniform(0.0, 2.0, 300)])
        ground = np.column_stack([rng.uniform(-20, 20, 3000), rng.uniform(-20, 20, 3000), rng.normal(-1.7, 0.02, 3000)])
        parts = [obj, static, ground]
        if ground_patch_ahead:
            # dense road returns exactly under the moving object's box of keyframe k + 3 (ground at z = -1.7, boxes
            # with ground_patch_ahead are placed so that their bottom touches the ground)
            parts.append(np.column_stack([rng.normal(5 + SPEED * (k + 3), 0.3, 4000), rng.normal(3.0, 0.3, 4000), rng.normal(-1.7, 0.02, 4000)]))
        pts = np.vstack(parts).astype(np.float32)
        extra = np.zeros((len(pts), nfields - 3), np.float32)
        (root / "data" / "LIDAR_CONCAT" / f"{k:05d}.pcd.bin").write_bytes(np.hstack([pts, extra]).astype(np.float32).tobytes())
        record = {"token": f"sd{k}", "sample_token": tokens[k], "ego_pose_token": f"e{k}", "calibrated_sensor_token": "cs", "timestamp": ts, "fileformat": "pcd.bin", "is_key_frame": True, "filename": f"data/LIDAR_CONCAT/{k:05d}.pcd.bin", "prev": "", "next": ""}
        if extra_field:  # the new pipeline writes LIDAR_CONCAT_INFO with the point layout
            (root / "data" / "LIDAR_CONCAT_INFO").mkdir(exist_ok=True)
            (root / "data" / "LIDAR_CONCAT_INFO" / f"{k:05d}.json").write_text(json.dumps({"num_pts_feats": nfields, "sources": []}))
            record["info_filename"] = f"data/LIDAR_CONCAT_INFO/{k:05d}.json"
        sample_data.append(record)
    for k in range(NUM_KF):
        target = k + annotation_shift
        if not 0 <= target < NUM_KF:
            continue
        mov_z = -1.7 + 1.6 / 2 if ground_patch_ahead else 0.9
        annotations.append({"token": f"a{k}", "sample_token": tokens[target], "instance_token": "inst_mov", "attribute_tokens": [], "visibility_token": "v", "translation": [5 + SPEED * k, 3.0, mov_z], "size": [0.8, 0.8, 1.6], "rotation": [1.0, 0.0, 0.0, 0.0], "num_lidar_pts": 0, "num_radar_pts": 0, "prev": "", "next": ""})
        annotations.append({"token": f"b{k}", "sample_token": tokens[target], "instance_token": "inst_static", "attribute_tokens": [], "visibility_token": "v", "translation": [-4.0, 6.0, 1.0], "size": [1.2, 1.2, 2.2], "rotation": [1.0, 0.0, 0.0, 0.0], "num_lidar_pts": 0, "num_radar_pts": 0, "prev": "", "next": ""})
    tables = {
        "sample": samples, "sample_data": sample_data, "ego_pose": ego_poses, "sample_annotation": annotations,
        "sensor": [{"token": "sensor", "channel": "LIDAR_CONCAT", "modality": "lidar"}],
        "calibrated_sensor": [{"token": "cs", "sensor_token": "sensor", "translation": [0, 0, 0], "rotation": [1, 0, 0, 0], "camera_intrinsic": []}],
        "instance": [{"token": "inst_mov", "category_token": "cat_ped", "nbr_annotations": NUM_KF, "first_annotation_token": "a0", "last_annotation_token": f"a{NUM_KF-1}"},
                     {"token": "inst_static", "category_token": "cat_car", "nbr_annotations": NUM_KF, "first_annotation_token": "b0", "last_annotation_token": f"b{NUM_KF-1}"}],
        "category": [{"token": "cat_ped", "name": "pedestrian", "description": ""}, {"token": "cat_car", "name": "car", "description": ""}],
    }
    for name, rows in tables.items():
        (ann / f"{name}.json").write_text(json.dumps(rows))


def test_consistent_dataset_is_ok(tmp_path):
    _write_dataset(tmp_path / "ds")
    report = estimate_annotation_offset(str(tmp_path / "ds"))
    assert report.status == "ok", report.message
    assert report.dominant_offset == 0 and report.consistent_fraction == 1.0
    assert report.num_moving_boxes == NUM_KF - 2  # first/last keyframe have no velocity estimate


def test_shifted_annotations_are_detected(tmp_path):
    _write_dataset(tmp_path / "ds", annotation_shift=2, extra_field=True)  # 7-field clouds like the new pipeline
    report = estimate_annotation_offset(str(tmp_path / "ds"))
    assert report.status == "inconsistent", report.message
    assert report.dominant_offset == -2
    assert "-2 keyframe" in report.message


def test_inconclusive_without_moving_objects(tmp_path):
    _write_dataset(tmp_path / "ds")
    ann = json.loads((tmp_path / "ds" / "annotation" / "sample_annotation.json").read_text())
    (tmp_path / "ds" / "annotation" / "sample_annotation.json").write_text(json.dumps([a for a in ann if a["instance_token"] == "inst_static"]))
    report = estimate_annotation_offset(str(tmp_path / "ds"))
    assert report.status == "inconclusive"


def test_points_preserved_between_reference_and_candidate(tmp_path):
    _write_dataset(tmp_path / "ref")
    _write_dataset(tmp_path / "cand", cloud_scale=2, extra_field=True)  # denser regenerated cloud, same scene
    ok = compare_points_preserved(str(tmp_path / "ref"), str(tmp_path / "cand"))
    assert ok["status"] == "ok" and ok["boxes_failed"] == 0 and ok["boxes_checked"] == 2 * NUM_KF
    _write_dataset(tmp_path / "cand_shifted", annotation_shift=0)
    # shift the candidate's clouds by relabelling: emulate a candidate whose frames are 1 s late
    import shutil
    lidar = tmp_path / "cand_shifted" / "data" / "LIDAR_CONCAT"
    files = sorted(lidar.iterdir())
    for src, dst in zip(files[1:], files[:-1]):
        shutil.copy(src, dst)
    bad = compare_points_preserved(str(tmp_path / "ref"), str(tmp_path / "cand_shifted"))
    assert bad["status"] == "inconsistent" and bad["boxes_failed"] >= NUM_KF - 1


def test_compare_versions_detects_index_copied_annotations(tmp_path):
    _write_dataset(tmp_path / "v0")
    _write_dataset(tmp_path / "v1", annotation_shift=2)  # same boxes attached to samples 2 s later
    same = compare_versions(str(tmp_path / "v0"), str(tmp_path / "v0"))
    assert same["status"] == "ok" and same["dominant_offset_s"] == 0.0
    shifted = compare_versions(str(tmp_path / "v1"), str(tmp_path / "v0"))
    assert shifted["status"] == "inconsistent" and shifted["dominant_offset_s"] == -2.0
    assert shifted["matched_samples"] == NUM_KF - 2


def test_busy_scene_without_clear_shift_is_not_flagged(tmp_path):
    """Other moving clusters passing through a box's spot at other times must not look like a shift."""
    _write_dataset(tmp_path / "ds")
    import numpy as np
    lidar = tmp_path / "ds" / "data" / "LIDAR_CONCAT"
    rng = np.random.default_rng(1)
    for k in range(NUM_KF):
        pts = np.fromfile(lidar / f"{k:05d}.pcd.bin", dtype=np.float32).reshape(-1, 5)
        # add a second dense cluster where the moving object will be 3 keyframes later (and was 3 earlier)
        extra = []
        for kk in (k + 3, k - 3):
            if 0 <= kk < NUM_KF:
                extra.append(np.column_stack([rng.normal(5 + SPEED * kk, 0.15, 300), rng.normal(3.0, 0.15, 300), rng.uniform(0.2, 1.6, 300), np.zeros(300), np.zeros(300)]))
        if extra:
            pts = np.vstack([pts, *extra]).astype(np.float32)
        (lidar / f"{k:05d}.pcd.bin").write_bytes(pts.tobytes())
    report = estimate_annotation_offset(str(tmp_path / "ds"))
    assert report.status != "inconsistent", report.message


def test_dense_ground_under_a_box_does_not_vote_for_another_frame(tmp_path):
    """Batch 8 row 4: a lead car followed at ~3 s headway. Its box at keyframe k sits over the road that was 10 m
    ahead of the ego at k-3, where the ground rings are far denser than the car's own returns."""
    _write_dataset(tmp_path, ground_patch_ahead=True)
    report = estimate_annotation_offset(str(tmp_path))
    assert report.status == "ok", report.message
    assert report.dominant_offset == 0
    assert report.instance_offset_histogram.get(0, 0) >= 1
