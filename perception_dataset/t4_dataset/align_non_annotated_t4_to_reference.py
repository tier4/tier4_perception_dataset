from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from glob import glob
import json
import logging
import os
import os.path as osp
from pathlib import Path
import shutil
from statistics import median
from typing import Any

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger

JsonRow = dict[str, Any]
# (reference_index, candidate_index, timestamp_diff_us)
TimestampMatch = tuple[int, int, int]


class AlignNonAnnotatedT4ToReferenceConverter(AbstractConverter[list[dict[str, Any]]]):
    """Trim a non-annotated T4 dataset to the keyframe cadence of a reference
    annotated T4 dataset and remap the reference annotations onto it.

    ``input_base`` (the non-annotated candidate) and ``reference_base`` (the
    annotated reference) follow the same layout as the other T4 converters: a
    directory containing one sub-directory per scene, each holding either
    ``annotation/`` and ``data/`` directly or a ``t4_dataset/`` sub-directory
    that does. Scenes are paired by directory name (or one-to-one when each base
    holds a single scene). Outputs are written under ``output_base/<scene>``.
    """

    def __init__(
        self,
        input_base: str,
        reference_base: str,
        output_base: str,
        *,
        max_abs_diff_ms: float = 0.1,
        max_frame_drop_ratio: float = 0.1,
        copy_data: bool = False,
        write_alignment_report: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(input_base=input_base, output_base=output_base)
        self._input_base = Path(input_base)
        self._reference_base = Path(reference_base)
        self._output_base = Path(output_base)
        self._max_abs_diff_ms = max_abs_diff_ms
        self._max_frame_drop_ratio = max_frame_drop_ratio
        self._copy_data = copy_data
        self._write_alignment_report = write_alignment_report
        self._logger = logger or configure_logger(modname=__name__)

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def convert(self) -> list[dict[str, Any]]:
        candidate_scenes = self._discover_scenes(self._input_base)
        reference_scenes = self._discover_scenes(self._reference_base)

        if not candidate_scenes:
            raise FileNotFoundError(f"no T4 dataset scenes found in {self._input_base}")
        if not reference_scenes:
            raise FileNotFoundError(
                f"no reference T4 dataset scenes found in {self._reference_base}"
            )

        reference_by_name = dict(reference_scenes)
        single_reference_dir = reference_scenes[0][1] if len(reference_scenes) == 1 else None

        reports: list[dict[str, Any]] = []
        for scene_name, candidate_dir in candidate_scenes:
            if scene_name in reference_by_name:
                reference_dir = reference_by_name[scene_name]
            elif single_reference_dir is not None and len(candidate_scenes) == 1:
                reference_dir = single_reference_dir
            else:
                raise FileNotFoundError(
                    f"no matching reference scene for {scene_name} in {self._reference_base}"
                )

            output_dir = self._output_base / scene_name
            self._logger.info(
                f"[BEGIN] Aligning non-annotated T4 ({candidate_dir}) "
                f"to reference ({reference_dir}) into {output_dir}"
            )
            reports.append(self._align_scene(candidate_dir, reference_dir, output_dir))
            self._logger.info(
                f"[DONE] Aligning non-annotated T4 ({candidate_dir}) "
                f"to reference ({reference_dir}) into {output_dir}"
            )

        return reports

    # ------------------------------------------------------------------ #
    # scene discovery (mirrors the other T4 converters' IO expectations)
    # ------------------------------------------------------------------ #
    @classmethod
    def _t4_dataset_dir(cls, scene_dir: Path) -> Path | None:
        nested = scene_dir / "t4_dataset"
        if (nested / "annotation").is_dir():
            return nested
        if (scene_dir / "annotation").is_dir():
            return scene_dir
        return None

    @classmethod
    def _discover_scenes(cls, base_dir: Path) -> list[tuple[str, Path]]:
        # ``base_dir`` may itself be a single scene directory.
        own = cls._t4_dataset_dir(base_dir)
        if own is not None:
            return [(base_dir.name, own)]

        scenes: list[tuple[str, Path]] = []
        for child in sorted(Path(path) for path in glob(osp.join(str(base_dir), "*"))):
            if not child.is_dir():
                continue
            t4_dir = cls._t4_dataset_dir(child)
            if t4_dir is not None:
                scenes.append((child.name, t4_dir))
        return scenes

    # ------------------------------------------------------------------ #
    # per-scene alignment
    # ------------------------------------------------------------------ #
    def _align_scene(
        self, candidate_dir: Path, reference_dir: Path, output_dir: Path
    ) -> dict[str, Any]:
        candidate_tables = self._load_tables(candidate_dir)
        reference_tables = self._load_tables(reference_dir)

        candidate_samples = self._samples_sorted_by_timestamp(candidate_tables)
        reference_samples = self._samples_sorted_by_timestamp(reference_tables)
        if not reference_samples:
            raise RuntimeError(f"reference {reference_dir} has no samples")

        matches, unmatched = self._match_samples_by_timestamp(
            candidate_samples, reference_samples, max_abs_diff_ms=self._max_abs_diff_ms
        )

        if not matches:
            raise RuntimeError(
                f"alignment produced no matched keyframes for {candidate_dir} "
                f"(reference keyframes={len(reference_samples)}); nothing to output"
            )

        # Reference keyframes outside the candidate's covered time span (before the
        # first / after the last matched keyframe) are trimmed, not "dropped":
        # start/end timestamp mismatches between bags are largely inevitable. Only
        # interior gaps — unmatched keyframes within the covered span — count
        # against the drop ratio.
        leading, interior_unmatched, trailing = self._classify_unmatched(matches, unmatched)
        span_size = matches[-1][0] - matches[0][0] + 1
        interior_drop_ratio = len(interior_unmatched) / span_size
        if leading or trailing:
            self._logger.info(
                f"trimmed {len(leading)} leading + {len(trailing)} trailing reference "
                f"keyframe(s) outside the candidate time span for {candidate_dir}"
            )
        if interior_drop_ratio > self._max_frame_drop_ratio:
            raise RuntimeError(
                f"alignment dropped {len(interior_unmatched)}/{span_size} reference "
                f"keyframes ({interior_drop_ratio:.1%}) within the covered span for "
                f"{candidate_dir}, exceeding the allowed {self._max_frame_drop_ratio:.1%}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        key_candidate_indices = [candidate_index for _, candidate_index, _ in matches]
        sample_token_map = {
            reference_samples[reference_index]["token"]: candidate_samples[candidate_index][
                "token"
            ]
            for reference_index, candidate_index, _ in matches
        }
        reference_sample_data_token_map = self._reference_sample_data_token_map(
            candidate_tables, reference_tables, candidate_samples, reference_samples, matches
        )

        sample_rows, sample_data_rows = self._build_candidate_tables(
            candidate_dir=candidate_dir,
            output_dir=output_dir,
            candidate_tables=candidate_tables,
            candidate_samples=candidate_samples,
            key_candidate_indices=key_candidate_indices,
        )
        sample_annotations, instances = self._remap_sample_annotations(
            reference_tables.get("sample_annotation", []),
            reference_tables.get("instance", []),
            sample_token_map,
            {row["token"]: index for index, row in enumerate(sample_rows)},
        )

        self._write_output_tables(
            output_dir=output_dir,
            candidate_tables=candidate_tables,
            reference_tables=reference_tables,
            sample_rows=sample_rows,
            sample_data_rows=sample_data_rows,
            sample_annotations=sample_annotations,
            instances=instances,
            reference_sample_token_map=sample_token_map,
            reference_sample_data_token_map=reference_sample_data_token_map,
        )
        self._copy_non_annotation_files(candidate_dir, output_dir)

        report = self._build_report(
            candidate_dir=candidate_dir,
            reference_dir=reference_dir,
            output_dir=output_dir,
            candidate_samples=candidate_samples,
            reference_samples=reference_samples,
            matches=matches,
            unmatched=unmatched,
            leading_unmatched=leading,
            interior_unmatched=interior_unmatched,
            trailing_unmatched=trailing,
            span_size=span_size,
            interior_drop_ratio=interior_drop_ratio,
            sample_rows=sample_rows,
            sample_data_rows=sample_data_rows,
            sample_annotations=sample_annotations,
        )
        if self._write_alignment_report:
            self._save_json(output_dir / "alignment_report.json", report)
        return report

    # ------------------------------------------------------------------ #
    # sample selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _samples_sorted_by_timestamp(tables: dict[str, list[JsonRow]]) -> list[JsonRow]:
        # In the T4 format every ``sample`` is a keyframe and its ``timestamp`` is
        # the keyframe (LiDAR) timestamp, so alignment works directly off it.
        return sorted(tables["sample"], key=lambda row: row["timestamp"])

    @staticmethod
    def _classify_unmatched(
        matches: list[TimestampMatch], unmatched: list[JsonRow]
    ) -> tuple[list[JsonRow], list[JsonRow], list[JsonRow]]:
        """Split unmatched reference keyframes by where they fall relative to the
        matched span: ``leading`` (before the first match), ``interior`` (within
        the span), ``trailing`` (after the last match).
        """
        first_matched = matches[0][0]
        last_matched = matches[-1][0]
        leading, interior, trailing = [], [], []
        for row in unmatched:
            reference_index = row["reference_index"]
            if reference_index < first_matched:
                leading.append(row)
            elif reference_index > last_matched:
                trailing.append(row)
            else:
                interior.append(row)
        return leading, interior, trailing

    @staticmethod
    def _match_samples_by_timestamp(
        candidate_samples: list[JsonRow],
        reference_samples: list[JsonRow],
        *,
        max_abs_diff_ms: float,
    ) -> tuple[list[TimestampMatch], list[JsonRow]]:
        """Greedily match each reference keyframe to the nearest (monotonic)
        candidate sample within ``max_abs_diff_ms``. Reference keyframes without
        a close-enough candidate are reported as unmatched (dropped).
        """
        matches: list[TimestampMatch] = []
        unmatched: list[JsonRow] = []
        max_abs_diff_us = int(max_abs_diff_ms * 1000)
        candidate_index = 0
        num_candidates = len(candidate_samples)

        for reference_index, reference_sample in enumerate(reference_samples):
            reference_timestamp = reference_sample["timestamp"]
            best_index: int | None = None
            best_abs_diff: int | None = None

            while candidate_index < num_candidates:
                diff = candidate_samples[candidate_index]["timestamp"] - reference_timestamp
                abs_diff = abs(diff)
                if best_abs_diff is None or abs_diff < best_abs_diff:
                    best_index, best_abs_diff = candidate_index, abs_diff
                if diff >= 0:
                    break
                candidate_index += 1

            if best_index is None or best_abs_diff is None or best_abs_diff > max_abs_diff_us:
                unmatched.append(
                    {
                        "reference_index": reference_index,
                        "reference_timestamp": reference_timestamp,
                        "candidate_index": best_index,
                        "candidate_timestamp": (
                            candidate_samples[best_index]["timestamp"]
                            if best_index is not None
                            else None
                        ),
                        "timestamp_diff_us": (
                            candidate_samples[best_index]["timestamp"] - reference_timestamp
                            if best_index is not None
                            else None
                        ),
                    }
                )
                continue

            diff = candidate_samples[best_index]["timestamp"] - reference_timestamp
            matches.append((reference_index, best_index, diff))
            candidate_index = best_index + 1

        return matches, unmatched

    # ------------------------------------------------------------------ #
    # output table construction
    # ------------------------------------------------------------------ #
    def _build_candidate_tables(
        self,
        *,
        candidate_dir: Path,
        output_dir: Path,
        candidate_tables: dict[str, list[JsonRow]],
        candidate_samples: list[JsonRow],
        key_candidate_indices: list[int],
    ) -> tuple[list[JsonRow], list[JsonRow]]:
        candidate_start = key_candidate_indices[0]
        candidate_end = key_candidate_indices[-1]
        kept_samples = candidate_samples[candidate_start : candidate_end + 1]
        kept_sample_tokens = {row["token"] for row in kept_samples}
        candidate_index_by_sample_token = {
            row["token"]: index for index, row in enumerate(candidate_samples)
        }
        key_candidate_index_set = set(key_candidate_indices)
        key_sample_tokens = {candidate_samples[index]["token"] for index in key_candidate_indices}

        sample_rows = [deepcopy(candidate_samples[index]) for index in key_candidate_indices]
        self._update_chain(sample_rows)
        scene_token = sample_rows[0]["scene_token"]
        for row in sample_rows:
            row["scene_token"] = scene_token

        sample_data_rows: list[JsonRow] = []
        by_channel: dict[str, list[JsonRow]] = defaultdict(list)

        for row in candidate_tables["sample_data"]:
            if row["sample_token"] not in kept_sample_tokens:
                continue
            copied = deepcopy(row)
            candidate_index = candidate_index_by_sample_token[row["sample_token"]]
            next_key_index = next(
                index for index in key_candidate_indices if index >= candidate_index
            )
            output_index = candidate_index - candidate_start
            channel = self._channel_from_filename(row["filename"])
            copied["filename"] = self._renamed_data_filename(row["filename"], output_index)
            copied["sample_token"] = candidate_samples[next_key_index]["token"]
            copied["is_key_frame"] = candidate_index in key_candidate_index_set

            self._relink_or_copy(candidate_dir / row["filename"], output_dir / copied["filename"])
            if copied.get("info_filename"):
                copied["info_filename"] = self._renamed_data_filename(
                    row["info_filename"], output_index
                )
                self._relink_or_copy(
                    candidate_dir / row["info_filename"], output_dir / copied["info_filename"]
                )

            sample_data_rows.append(copied)
            by_channel[channel].append(copied)

        missing_key_tokens = key_sample_tokens - {
            row["sample_token"] for row in sample_data_rows if row.get("is_key_frame")
        }
        if missing_key_tokens:
            raise RuntimeError(f"missing keyframe sample_data for samples: {missing_key_tokens}")

        for channel_rows in by_channel.values():
            channel_rows.sort(key=lambda row: (row["timestamp"], row["filename"]))
            self._update_chain(channel_rows)

        sample_data_rows.sort(key=lambda row: (row["timestamp"], row["filename"]))
        return sample_rows, sample_data_rows

    def _reference_sample_data_token_map(
        self,
        candidate_tables: dict[str, list[JsonRow]],
        reference_tables: dict[str, list[JsonRow]],
        candidate_samples: list[JsonRow],
        reference_samples: list[JsonRow],
        matches: list[TimestampMatch],
    ) -> dict[str, str]:
        grouped_candidate = self._group_sample_data_by_sample(candidate_tables["sample_data"])
        grouped_reference = self._group_sample_data_by_sample(reference_tables["sample_data"])
        token_map: dict[str, str] = {}
        for reference_index, candidate_index, _ in matches:
            reference_rows = grouped_reference[reference_samples[reference_index]["token"]]
            candidate_rows = grouped_candidate[candidate_samples[candidate_index]["token"]]
            candidate_by_channel = {
                self._channel_from_filename(row["filename"]): row for row in candidate_rows
            }
            for reference_row in reference_rows:
                channel = self._channel_from_filename(reference_row["filename"])
                if channel in candidate_by_channel:
                    token_map[reference_row["token"]] = candidate_by_channel[channel]["token"]
        return token_map

    def _remap_sample_annotations(
        self,
        reference_annotations: list[JsonRow],
        reference_instances: list[JsonRow],
        sample_token_map: dict[str, str],
        output_sample_order: dict[str, int],
    ) -> tuple[list[JsonRow], list[JsonRow]]:
        remapped_annotations: list[JsonRow] = []
        for row in reference_annotations:
            if row["sample_token"] not in sample_token_map:
                continue
            copied = deepcopy(row)
            copied["sample_token"] = sample_token_map[row["sample_token"]]
            remapped_annotations.append(copied)

        by_instance: dict[str, list[JsonRow]] = defaultdict(list)
        for row in remapped_annotations:
            by_instance[row["instance_token"]].append(row)
        for rows in by_instance.values():
            rows.sort(key=lambda row: output_sample_order[row["sample_token"]])
            self._update_chain(rows)

        instance_by_token = {row["token"]: row for row in reference_instances}
        remapped_instances: list[JsonRow] = []
        for instance_token, rows in by_instance.items():
            copied = deepcopy(instance_by_token[instance_token])
            copied["nbr_annotations"] = len(rows)
            copied["first_annotation_token"] = rows[0]["token"]
            copied["last_annotation_token"] = rows[-1]["token"]
            remapped_instances.append(copied)

        remapped_annotations.sort(
            key=lambda row: (output_sample_order[row["sample_token"]], row["token"])
        )
        remapped_instances.sort(key=lambda row: row["token"])
        return remapped_annotations, remapped_instances

    @staticmethod
    def _remap_optional_annotation_table(
        rows: list[JsonRow],
        *,
        sample_token_map: dict[str, str],
        sample_data_token_map: dict[str, str],
    ) -> list[JsonRow]:
        remapped: list[JsonRow] = []
        for row in rows:
            copied = deepcopy(row)
            if "sample_token" in copied:
                if copied["sample_token"] not in sample_token_map:
                    continue
                copied["sample_token"] = sample_token_map[copied["sample_token"]]
            if "sample_data_token" in copied:
                if copied["sample_data_token"] not in sample_data_token_map:
                    continue
                copied["sample_data_token"] = sample_data_token_map[copied["sample_data_token"]]
            remapped.append(copied)
        return remapped

    def _write_output_tables(
        self,
        *,
        output_dir: Path,
        candidate_tables: dict[str, list[JsonRow]],
        reference_tables: dict[str, list[JsonRow]],
        sample_rows: list[JsonRow],
        sample_data_rows: list[JsonRow],
        sample_annotations: list[JsonRow],
        instances: list[JsonRow],
        reference_sample_token_map: dict[str, str],
        reference_sample_data_token_map: dict[str, str],
    ) -> None:
        output_annotation_dir = output_dir / "annotation"
        used_ego_pose_tokens = {row["ego_pose_token"] for row in sample_data_rows}

        tables: dict[str, list[JsonRow]] = {
            "sample": sample_rows,
            "sample_data": sample_data_rows,
            "ego_pose": [
                row for row in candidate_tables["ego_pose"] if row["token"] in used_ego_pose_tokens
            ],
            "calibrated_sensor": candidate_tables["calibrated_sensor"],
            "sensor": candidate_tables["sensor"],
            "vehicle_state": candidate_tables.get("vehicle_state", []),
            "log": candidate_tables.get("log", []),
            "map": candidate_tables.get("map", []),
        }

        scenes = deepcopy(candidate_tables["scene"])
        if scenes:
            scenes[0]["nbr_samples"] = len(sample_rows)
            scenes[0]["first_sample_token"] = sample_rows[0]["token"] if sample_rows else ""
            scenes[0]["last_sample_token"] = sample_rows[-1]["token"] if sample_rows else ""
        tables["scene"] = scenes

        for name in ["attribute", "category", "visibility"]:
            tables[name] = reference_tables.get(name, candidate_tables.get(name, []))
        tables["sample_annotation"] = sample_annotations
        tables["instance"] = instances

        for name in ["object_ann", "surface_ann"]:
            if name in reference_tables:
                tables[name] = self._remap_optional_annotation_table(
                    reference_tables[name],
                    sample_token_map=reference_sample_token_map,
                    sample_data_token_map=reference_sample_data_token_map,
                )

        for name, rows in tables.items():
            self._save_json(output_annotation_dir / f"{name}.json", rows)

    def _copy_non_annotation_files(self, candidate_dir: Path, output_dir: Path) -> None:
        for path in candidate_dir.iterdir():
            if path.name in {"annotation", "data"}:
                continue
            destination = output_dir / path.name
            if path.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(path, destination)
            else:
                shutil.copy2(path, destination)

    def _build_report(
        self,
        *,
        candidate_dir: Path,
        reference_dir: Path,
        output_dir: Path,
        candidate_samples: list[JsonRow],
        reference_samples: list[JsonRow],
        matches: list[TimestampMatch],
        unmatched: list[JsonRow],
        leading_unmatched: list[JsonRow],
        interior_unmatched: list[JsonRow],
        trailing_unmatched: list[JsonRow],
        span_size: int,
        interior_drop_ratio: float,
        sample_rows: list[JsonRow],
        sample_data_rows: list[JsonRow],
        sample_annotations: list[JsonRow],
    ) -> dict[str, Any]:
        used_diffs = [diff for _, _, diff in matches]
        max_abs_diff_us = max(abs(diff) for diff in used_diffs)
        if max_abs_diff_us > self._max_abs_diff_ms * 1000:
            self._logger.warning(
                "selected alignment exceeds tolerance: "
                f"max_abs_diff={max_abs_diff_us / 1000:.3f} ms"
            )
        alignment_results = [
            {
                "reference_index": reference_index,
                "candidate_index": candidate_index,
                "reference_timestamp": reference_samples[reference_index]["timestamp"],
                "candidate_timestamp": candidate_samples[candidate_index]["timestamp"],
                "timestamp_diff_us": diff,
            }
            for reference_index, candidate_index, diff in matches
        ]
        return {
            "candidate_dir": str(candidate_dir),
            "reference_dir": str(reference_dir),
            "output_dir": str(output_dir),
            "max_abs_diff_ms": self._max_abs_diff_ms,
            "max_frame_drop_ratio": self._max_frame_drop_ratio,
            "num_reference_keyframes": len(reference_samples),
            "num_keyframes": len(matches),
            "num_unmatched_reference_keyframes": len(unmatched),
            "num_trimmed_leading_keyframes": len(leading_unmatched),
            "num_trimmed_trailing_keyframes": len(trailing_unmatched),
            "num_interior_dropped_keyframes": len(interior_unmatched),
            "covered_span_keyframes": span_size,
            "interior_frame_drop_ratio": interior_drop_ratio,
            "num_samples": len(sample_rows),
            "num_sample_data": len(sample_data_rows),
            "num_sample_annotations": len(sample_annotations),
            "first_timestamp_diff_us": used_diffs[0],
            "median_abs_timestamp_diff_us": median(abs(diff) for diff in used_diffs),
            "max_abs_timestamp_diff_us": max_abs_diff_us,
            "timestamp_diffs_us": used_diffs,
            "alignment_results": alignment_results,
            "unmatched_reference_results": unmatched,
        }

    # ------------------------------------------------------------------ #
    # small reusable helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _channel_from_filename(filename: str) -> str:
        parts = Path(filename).parts
        if len(parts) < 3 or parts[0] != "data":
            raise ValueError(f"unexpected sample_data filename: {filename}")
        return parts[1]

    @staticmethod
    def _renamed_data_filename(filename: str, output_index: int) -> str:
        parts = Path(filename).parts
        if len(parts) < 3 or parts[0] != "data":
            raise ValueError(f"unexpected data filename: {filename}")
        name = parts[-1]
        extension = name[name.find(".") :] if "." in name else ""
        return f"data/{parts[1]}/{output_index:05d}{extension}"

    @staticmethod
    def _group_sample_data_by_sample(sample_data: list[JsonRow]) -> dict[str, list[JsonRow]]:
        grouped: dict[str, list[JsonRow]] = defaultdict(list)
        for row in sample_data:
            grouped[row["sample_token"]].append(row)
        return grouped

    @staticmethod
    def _update_chain(rows: list[JsonRow]) -> None:
        for index, row in enumerate(rows):
            row["prev"] = rows[index - 1]["token"] if index > 0 else ""
            row["next"] = rows[index + 1]["token"] if index + 1 < len(rows) else ""

    def _relink_or_copy(self, src: Path, dst: Path) -> None:
        if not src.exists():
            raise FileNotFoundError(f"sample_data file does not exist: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        if self._copy_data:
            shutil.copy2(src, dst)
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    @staticmethod
    def _load_tables(dataset_dir: Path) -> dict[str, list[JsonRow]]:
        annotation_dir = dataset_dir / "annotation"
        if not annotation_dir.is_dir():
            raise FileNotFoundError(f"annotation directory does not exist: {annotation_dir}")
        tables: dict[str, list[JsonRow]] = {}
        for path in annotation_dir.glob("*.json"):
            with path.open() as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} must contain a JSON list")
            tables[path.stem] = data
        return tables

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2)
