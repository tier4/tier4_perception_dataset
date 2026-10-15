"""Health checks for the consistency between 3D box annotations and lidar frames of a T4 dataset.

The main check estimates whether the boxes attached to each keyframe describe that keyframe's point
cloud or a neighbouring one. It only looks at *moving* objects (static ones fit every frame): for each
moving box at keyframe ``k`` it counts lidar points inside the box using the clouds of keyframes
``k + d`` for ``d`` in ``[-max_offset, max_offset]``; a consistent dataset peaks at ``d = 0``.
A systematic peak elsewhere means the annotations are attached to the wrong samples (this happened
when a dataset version was re-cut with ``skip_timestamp`` and annotations were copied by sample index).

Usage::

    python -m perception_dataset.t4_dataset.annotation_health <dataset_dir> [--json]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

LIDAR_CHANNEL = "LIDAR_CONCAT"


@dataclass
class HealthReport:
    dataset_dir: str
    num_keyframes: int
    num_boxes: int
    num_moving_boxes: int
    offset_histogram: Dict[int, int] = field(default_factory=dict)
    points_by_offset: Dict[int, int] = field(default_factory=dict)
    instance_offset_histogram: Dict[int, int] = field(default_factory=dict)  # one vote per moving instance (its majority)
    num_moving_instances: int = 0
    dominant_offset: Optional[int] = None
    consistent_fraction: Optional[float] = None
    status: str = "inconclusive"  # ok | inconsistent | inconclusive
    message: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["offset_histogram"] = {str(k): v for k, v in self.offset_histogram.items()}
        d["points_by_offset"] = {str(k): v for k, v in self.points_by_offset.items()}
        d["instance_offset_histogram"] = {str(k): v for k, v in self.instance_offset_histogram.items()}
        return d


def _quat_wxyz(q) -> Rotation:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _load(root: Path, name: str):
    with open(root / "annotation" / f"{name}.json") as fh:
        return json.load(fh)


def _num_point_fields(root: Path, sample_data: dict, default: int = 5) -> int:
    info = sample_data.get("info_filename")
    if info and (root / info).exists():
        try:
            with open(root / info) as fh:
                n = json.load(fh).get("num_pts_feats")
            if isinstance(n, int) and n > 0:
                return n
        except (OSError, ValueError):
            pass
    size = os.path.getsize(root / sample_data["filename"])
    for n in (default, 4, 6, 7, 8):
        if size % (4 * n) == 0:
            return n
    return default


class _Dataset:
    """Minimal T4 reader: keyframe lidar clouds in the global frame and boxes per sample index."""

    def __init__(self, root: Path):
        self.root = root
        sensors = {s["token"]: s["channel"] for s in _load(root, "sensor")}
        calibrated = {c["token"]: c for c in _load(root, "calibrated_sensor")}
        self.ego_pose = {e["token"]: e for e in _load(root, "ego_pose")}
        self.samples = sorted(_load(root, "sample"), key=lambda s: s["timestamp"])
        self.sample_index = {s["token"]: i for i, s in enumerate(self.samples)}
        self.lidar_by_index: Dict[int, dict] = {}
        for sd in _load(root, "sample_data"):
            if not sd["is_key_frame"]:
                continue
            if sensors[calibrated[sd["calibrated_sensor_token"]]["sensor_token"]] != LIDAR_CHANNEL:
                continue
            idx = self.sample_index.get(sd["sample_token"])
            if idx is not None:
                self.lidar_by_index.setdefault(idx, sd)
        instances = {i["token"]: i for i in _load(root, "instance")}
        categories = {c["token"]: c["name"] for c in _load(root, "category")}
        self.boxes_by_instance: Dict[str, Dict[int, dict]] = {}
        for a in _load(root, "sample_annotation"):
            idx = self.sample_index.get(a["sample_token"])
            if idx is None:
                continue
            a = dict(a)
            a["category"] = categories.get(instances[a["instance_token"]]["category_token"], "unknown")
            self.boxes_by_instance.setdefault(a["instance_token"], {})[idx] = a
        self._clouds: Dict[int, np.ndarray] = {}

    @property
    def num_keyframes(self) -> int:
        return len(self.lidar_by_index)

    def cloud(self, idx: int) -> np.ndarray:
        if idx not in self._clouds:
            sd = self.lidar_by_index[idx]
            n = _num_point_fields(self.root, sd)
            pts = np.fromfile(self.root / sd["filename"], dtype=np.float32).reshape(-1, n)[:, :3].astype(np.float64)
            ego = self.ego_pose[sd["ego_pose_token"]]
            self._clouds[idx] = _quat_wxyz(ego["rotation"]).apply(pts) + np.asarray(ego["translation"])
        return self._clouds[idx]

    def ego_translation(self, idx: int) -> np.ndarray:
        return np.asarray(self.ego_pose[self.lidar_by_index[idx]["ego_pose_token"]]["translation"])


def points_in_box(points: np.ndarray, box: dict, margin: float = 0.15, ground_margin: float = 0.3) -> int:
    """Lidar points inside ``box`` (with ``margin``), ignoring the bottom ``ground_margin`` metres.

    Boxes touch the road, so the footprint of a box also contains ground returns; near the ego those rings are so
    dense that a *different* frame's ground (e.g. the road 10 m ahead of the ego three keyframes earlier, lying under
    a lead vehicle's current box) outnumbers the object's own points. Only the volume above the ground slab counts.
    """
    local = _quat_wxyz(box["rotation"]).inv().apply(points - np.asarray(box["translation"]))
    w, l, h = box["size"]
    inside = (
        (np.abs(local[:, 0]) <= w / 2 + margin)
        & (np.abs(local[:, 1]) <= l / 2 + margin)
        & (local[:, 2] <= h / 2 + margin)
        & (local[:, 2] >= -h / 2 + ground_margin)
    )
    return int(inside.sum())


def estimate_annotation_offset(
    dataset_dir: str,
    *,
    max_offset: int = 3,
    min_speed_mps: float = 1.0,
    max_range_m: float = 40.0,
    min_points: int = 15,
    min_moving_boxes: int = 3,
    min_consistent_fraction: float = 0.5,
    shift_ratio: float = 1.5,
) -> HealthReport:
    """Estimate, in keyframes, how far the annotations are shifted against the lidar frames.

    Returns a :class:`HealthReport`. ``inconsistent`` means a non-zero offset wins both the per-box votes and
    the lidar point totals by ``shift_ratio``; ``ok`` means offset 0 holds (>= ``min_consistent_fraction`` of the
    votes, or within 20 % of the best point total); ``inconclusive`` covers too few moving boxes or an ambiguous
    picture (busy traffic).
    """
    root = Path(dataset_dir)
    ds = _Dataset(root)
    n = len(ds.samples)
    keyframes = sorted(ds.lidar_by_index)
    num_boxes = sum(len(v) for v in ds.boxes_by_instance.values())
    offsets = list(range(-max_offset, max_offset + 1))
    hist: Counter = Counter()
    inst_hist: Counter = Counter()
    totals: Dict[int, int] = {d: 0 for d in offsets}
    used = 0

    for per_index in ds.boxes_by_instance.values():
        inst_votes: Counter = Counter()
        for k, box in per_index.items():
            if k not in ds.lidar_by_index:
                continue
            prev_box, next_box = per_index.get(k - 1), per_index.get(k + 1)
            if prev_box is None or next_box is None:
                continue
            velocity = (np.asarray(next_box["translation"]) - np.asarray(prev_box["translation"])) / 2.0
            if np.linalg.norm(velocity[:2]) < min_speed_mps:
                continue
            if np.linalg.norm(np.asarray(box["translation"]) - ds.ego_translation(k)) > max_range_m:
                continue
            counts = {d: points_in_box(ds.cloud(k + d), box) for d in offsets if (k + d) in ds.lidar_by_index}
            if not counts or max(counts.values()) < min_points:
                continue
            used += 1
            best = max(counts, key=counts.get)
            if best != 0 and counts[best] < 1.5 * max(1, counts.get(0, 0)):
                best = 0  # not a clear win over the box's own frame -> counts as consistent
            hist[best] += 1
            inst_votes[best] += 1
            for d, c in counts.items():
                totals[d] += c
        if inst_votes:
            inst_hist[max(inst_votes, key=inst_votes.get)] += 1

    report = HealthReport(
        dataset_dir=str(root),
        num_keyframes=len(keyframes),
        num_boxes=num_boxes,
        num_moving_boxes=used,
        offset_histogram=dict(sorted(hist.items())),
        points_by_offset=totals,
        instance_offset_histogram=dict(sorted(inst_hist.items())),
        num_moving_instances=sum(inst_hist.values()),
    )
    if used < min_moving_boxes:
        report.status = "inconclusive"
        report.message = f"only {used} moving box(es) with enough lidar points; need {min_moving_boxes} to judge"
        return report
    # Decision: a real attachment shift shows up as a non-zero offset that wins clearly on BOTH counts - votes and
    # total points (>= shift_ratio x the points at offset 0). Busy scenes produce diffuse runner-up votes (other
    # vehicles passing through a box's location at other times) with totals close to offset 0; those are not shifts.
    best_total = max(totals, key=totals.get)
    ratio = totals[best_total] / max(1, totals.get(0, 0))
    report.dominant_offset = int(max(hist, key=hist.get))
    report.consistent_fraction = hist[0] / used
    # A shift caused by copying annotations onto other samples moves *every* instance, so it must also win the
    # per-instance majority vote; a single long-lived object (e.g. a lead vehicle) cannot flag a dataset alone.
    if best_total != 0 and ratio >= shift_ratio and hist[best_total] >= hist[0] and inst_hist[best_total] > inst_hist[0]:
        report.status = "inconsistent"
        report.dominant_offset = int(best_total)
        report.message = (
            f"annotations best match the lidar frame {best_total:+d} keyframe(s) away: {hist[best_total]}/{used} moving boxes "
            f"({inst_hist[best_total]}/{report.num_moving_instances} instances) and {ratio:.1f}x the lidar points of offset 0 "
            f"(offset histogram {dict(hist)})"
        )
    elif report.consistent_fraction >= min_consistent_fraction or totals.get(0, 0) >= 0.8 * totals[best_total]:
        report.status = "ok"
        report.message = (
            f"{hist[0]}/{used} moving boxes match their own lidar frame; no offset has >= {shift_ratio:.1f}x its lidar points"
        )
    else:
        report.status = "inconclusive"
        report.message = (
            f"ambiguous: {hist[0]}/{used} moving boxes match their own frame and no offset dominates "
            f"(histogram {dict(hist)}, per-instance {dict(inst_hist)}, points {totals})"
        )
    return report


def compare_points_preserved(
    reference_dir: str,
    candidate_dir: str,
    *,
    min_reference_points: int = 5,
    min_ratio: float = 0.5,
) -> Dict:
    """Check that boxes keep their lidar support after re-attaching annotations to a new dataset.

    Both datasets must share sample timestamps (keyframes are matched by timestamp). For every box
    with at least ``min_reference_points`` points in the reference cloud, the candidate cloud must
    contain at least ``min_ratio`` times that many points inside the same (global-frame) box.
    """
    ref, cand = _Dataset(Path(reference_dir)), _Dataset(Path(candidate_dir))
    cand_index_by_ts = {cand.samples[i]["timestamp"]: i for i in cand.lidar_by_index}
    checked = failed = 0
    worst: List[Tuple[float, str, int]] = []
    for per_index in ref.boxes_by_instance.values():
        for k, box in per_index.items():
            if k not in ref.lidar_by_index:
                continue
            j = cand_index_by_ts.get(ref.samples[k]["timestamp"])
            if j is None:
                continue
            n_ref = points_in_box(ref.cloud(k), box)
            if n_ref < min_reference_points:
                continue
            n_cand = points_in_box(cand.cloud(j), box)
            checked += 1
            ratio = n_cand / n_ref
            if ratio < min_ratio:
                failed += 1
                worst.append((ratio, box["category"], k))
    worst.sort()
    status = "ok" if checked and failed == 0 else ("inconclusive" if not checked else "inconsistent")
    return {
        "status": status,
        "boxes_checked": checked,
        "boxes_failed": failed,
        "worst": [{"ratio": round(r, 3), "category": c, "keyframe": k} for r, c, k in worst[:10]],
        "message": f"{checked - failed}/{checked} boxes keep >= {min_ratio:.0%} of their reference lidar points",
    }


def compare_versions(dataset_dir: str, other_version_dir: str, *, min_overlap: float = 0.8) -> Dict:
    """Detect annotations copied between dataset versions by sample index instead of by time.

    For every sample of ``dataset_dir`` the box set (rounded global translations) is looked up in
    ``other_version_dir``; the time offset between the matching samples is reported. Identical
    versions give an offset of 0 s for every sample; a version whose frames were re-cut (e.g. with
    ``skip_timestamp``) while the annotations were copied by index gives a constant non-zero offset.
    Works without moving objects and complements :func:`estimate_annotation_offset`.
    """
    def box_sets(root: Path):
        ds = _Dataset(root)
        out: Dict[int, set] = {}
        for per_index in ds.boxes_by_instance.values():
            for k, box in per_index.items():
                out.setdefault(k, set()).add(tuple(np.round(box["translation"], 2)))
        return ds, out

    ds_a, sets_a = box_sets(Path(dataset_dir))
    ds_b, sets_b = box_sets(Path(other_version_dir))
    offsets: Counter = Counter()
    unmatched = 0
    for k, boxes in sets_a.items():
        if not sets_b:
            unmatched += 1
            continue
        best = max(sets_b, key=lambda j: len(boxes & sets_b[j]))
        overlap = len(boxes & sets_b[best]) / max(1, len(boxes))
        if overlap < min_overlap:
            unmatched += 1
            continue
        offsets[round((ds_b.samples[best]["timestamp"] - ds_a.samples[k]["timestamp"]) / 1e6, 1)] += 1
    matched = sum(offsets.values())
    dominant = max(offsets, key=offsets.get) if offsets else None
    if matched == 0:
        status, message = "inconclusive", "no sample box set of the dataset was found in the other version"
    elif dominant == 0.0 and offsets[0.0] == matched:
        status, message = "ok", f"all {matched} matched samples carry the same boxes at the same time in both versions"
    else:
        status = "inconsistent"
        message = (
            f"boxes of {offsets[dominant]}/{matched} samples appear in the other version {dominant:+.1f} s away "
            f"(annotations copied by sample index across re-cut versions?)"
        )
    return {
        "status": status,
        "message": message,
        "matched_samples": matched,
        "unmatched_samples": unmatched,
        "offset_histogram_s": {str(k): v for k, v in sorted(offsets.items())},
        "dominant_offset_s": dominant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", help="T4 dataset directory (holds annotation/ and data/)")
    parser.add_argument("--max-offset", type=int, default=3)
    parser.add_argument("--min-speed", type=float, default=1.0, help="min object speed in m/s to count as moving")
    parser.add_argument("--max-range", type=float, default=40.0, help="max distance from the ego for moving boxes (m)")
    parser.add_argument("--min-points", type=int, default=15, help="min lidar points a moving box must reach in some frame")
    parser.add_argument("--compare", metavar="OTHER_VERSION_DIR", help="also match box sets against another version of the same dataset")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()
    report = estimate_annotation_offset(
        args.dataset_dir, max_offset=args.max_offset, min_speed_mps=args.min_speed, max_range_m=args.max_range, min_points=args.min_points
    )
    comparison = compare_versions(args.dataset_dir, args.compare) if args.compare else None
    if args.json:
        print(json.dumps({"moving_objects": report.to_dict(), "version_comparison": comparison}, indent=2))
    else:
        print(f"{report.status.upper()}: {report.message}")
        print(f"  keyframes={report.num_keyframes} boxes={report.num_boxes} moving boxes used={report.num_moving_boxes}")
        print(f"  points inside moving boxes by lidar offset (keyframes): {report.points_by_offset}")
        if comparison:
            print(f"VERSION COMPARISON {comparison['status'].upper()}: {comparison['message']}  offsets={comparison['offset_histogram_s']}")
    statuses = [report.status] + ([comparison["status"]] if comparison else [])
    raise SystemExit(2 if "inconsistent" in statuses else (0 if "ok" in statuses else 3))


if __name__ == "__main__":
    main()
