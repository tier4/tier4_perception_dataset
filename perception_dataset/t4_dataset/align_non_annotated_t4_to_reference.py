from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.utils.logger import configure_logger


LIDAR_CHANNEL = "LIDAR_CONCAT"
DEFAULT_KEYFRAME_STRIDE = 10
DEFAULT_MAX_ABS_DIFF_MS = 1.0


JsonRow = dict[str, Any]
TimestampMatch = tuple[int, int, int]


def load_json(path: Path) -> list[JsonRow]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def annotation_dir(dataset_dir: Path) -> Path:
    anno_dir = dataset_dir / "annotation"
    if not anno_dir.is_dir():
        raise FileNotFoundError(f"annotation directory does not exist: {anno_dir}")
    return anno_dir


def t4_scene_dir(path: Path) -> Path | None:
    if (path / "annotation").is_dir():
        return path
    t4_dataset_dir = path / "t4_dataset"
    if (t4_dataset_dir / "annotation").is_dir():
        return t4_dataset_dir
    return None


def scene_name_from_dir(path: Path) -> str:
    return path.parent.name if path.name == "t4_dataset" else path.name


def discover_t4_scenes(base_dir: Path) -> tuple[bool, list[tuple[str, Path]]]:
    scene_dir = t4_scene_dir(base_dir)
    if scene_dir is not None:
        return True, [(scene_name_from_dir(scene_dir), scene_dir)]

    scenes: list[tuple[str, Path]] = []
    for child_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        scene_dir = t4_scene_dir(child_dir)
        if scene_dir is not None:
            scenes.append((scene_name_from_dir(scene_dir), scene_dir))
    return False, scenes


def channel_from_filename(filename: str) -> str:
    parts = Path(filename).parts
    if len(parts) < 3 or parts[0] != "data":
        raise ValueError(f"unexpected sample_data filename: {filename}")
    return parts[1]


def extension_from_filename(filename: str) -> str:
    name = Path(filename).name
    return name[name.find(".") :] if "." in name else ""


def renamed_data_filename(filename: str, output_index: int) -> str:
    parts = Path(filename).parts
    if len(parts) < 3 or parts[0] != "data":
        raise ValueError(f"unexpected data filename: {filename}")
    return f"data/{parts[1]}/{output_index:05d}{extension_from_filename(filename)}"


def group_sample_data_by_sample(sample_data: list[JsonRow]) -> dict[str, list[JsonRow]]:
    grouped: dict[str, list[JsonRow]] = defaultdict(list)
    for row in sample_data:
        grouped[row["sample_token"]].append(row)
    return grouped


def lidar_rows(sample_data: list[JsonRow]) -> list[JsonRow]:
    rows = [
        row
        for row in sample_data
        if channel_from_filename(row["filename"]) == LIDAR_CHANNEL
    ]
    return sorted(rows, key=lambda row: row["timestamp"])


def reference_key_samples(
    reference_samples: list[JsonRow], reference_sample_data: list[JsonRow]
) -> list[JsonRow]:
    samples_by_token = {row["token"]: row for row in reference_samples}
    keys = [
        row
        for row in lidar_rows(reference_sample_data)
        if row.get("is_key_frame", False) and row["sample_token"] in samples_by_token
    ]
    return [samples_by_token[row["sample_token"]] for row in keys]


def candidate_samples_with_lidar(
    candidate_samples: list[JsonRow], candidate_sample_data: list[JsonRow]
) -> list[JsonRow]:
    lidar_by_sample = {row["sample_token"]: row for row in lidar_rows(candidate_sample_data)}
    rows = [row for row in candidate_samples if row["token"] in lidar_by_sample]
    return sorted(rows, key=lambda row: lidar_by_sample[row["token"]]["timestamp"])


def score_alignment(
    candidate_samples: list[JsonRow],
    reference_samples: list[JsonRow],
    *,
    stride: int,
    max_candidate_offset: int,
    max_reference_offset: int,
) -> tuple[int, int, list[int]]:
    best: tuple[tuple[float, float, float, int], int, int, list[int]] | None = None
    candidate_limit = min(max_candidate_offset + 1, len(candidate_samples))
    reference_limit = min(max_reference_offset + 1, len(reference_samples))

    for candidate_start in range(candidate_limit):
        for reference_start in range(reference_limit):
            diffs: list[int] = []
            k = 0
            while (
                candidate_start + stride * k < len(candidate_samples)
                and reference_start + k < len(reference_samples)
            ):
                candidate_timestamp = candidate_samples[candidate_start + stride * k][
                    "timestamp"
                ]
                reference_timestamp = reference_samples[reference_start + k][
                    "timestamp"
                ]
                diffs.append(candidate_timestamp - reference_timestamp)
                k += 1
            if not diffs:
                continue
            abs_diffs = [abs(diff) for diff in diffs]
            rank = (abs(diffs[0]), median(abs_diffs), max(abs_diffs), -len(diffs))
            if best is None or rank < best[0]:
                best = (rank, candidate_start, reference_start, diffs)

    if best is None:
        raise RuntimeError("could not find a candidate/reference alignment")
    _, candidate_start, reference_start, diffs = best
    return candidate_start, reference_start, diffs


def match_reference_samples_by_timestamp(
    candidate_samples: list[JsonRow],
    reference_samples: list[JsonRow],
    *,
    candidate_start: int,
    reference_start: int,
    max_abs_diff_ms: float,
) -> tuple[list[TimestampMatch], list[JsonRow]]:
    matches: list[TimestampMatch] = []
    unmatched: list[JsonRow] = []
    candidate_index = candidate_start
    max_abs_diff_us = int(max_abs_diff_ms * 1000)

    for reference_index in range(reference_start, len(reference_samples)):
        reference_timestamp = reference_samples[reference_index]["timestamp"]
        best_index: int | None = None
        best_abs_diff: int | None = None

        while candidate_index < len(candidate_samples):
            diff = candidate_samples[candidate_index]["timestamp"] - reference_timestamp
            abs_diff = abs(diff)
            if best_abs_diff is None or abs_diff < best_abs_diff:
                best_index = candidate_index
                best_abs_diff = abs_diff
            if diff >= 0:
                break
            candidate_index += 1

        if best_index is None or best_abs_diff is None or best_abs_diff > max_abs_diff_us:
            miss: JsonRow = {
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
            unmatched.append(miss)
            if candidate_index >= len(candidate_samples):
                break
            continue

        diff = candidate_samples[best_index]["timestamp"] - reference_timestamp
        matches.append((reference_index, best_index, diff))
        candidate_index = best_index + 1

    return matches, unmatched


def relink_or_copy(src: Path, dst: Path, *, copy_data: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_data:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def update_chain(rows: list[JsonRow]) -> None:
    for index, row in enumerate(rows):
        row["prev"] = rows[index - 1]["token"] if index > 0 else ""
        row["next"] = rows[index + 1]["token"] if index + 1 < len(rows) else ""


def build_filtered_candidate_tables(
    *,
    candidate_dir: Path,
    output_dir: Path,
    candidate_tables: dict[str, list[JsonRow]],
    candidate_samples: list[JsonRow],
    candidate_start: int,
    candidate_end: int,
    key_candidate_indices: list[int],
    copy_data: bool,
) -> tuple[list[JsonRow], list[JsonRow], dict[str, str]]:
    kept_samples = candidate_samples[candidate_start : candidate_end + 1]
    kept_sample_tokens = {row["token"] for row in kept_samples}
    candidate_index_by_sample_token = {
        row["token"]: index for index, row in enumerate(candidate_samples)
    }
    key_candidate_index_set = set(key_candidate_indices)
    key_sample_tokens = {candidate_samples[index]["token"] for index in key_candidate_indices}

    sample_rows = [deepcopy(candidate_samples[index]) for index in key_candidate_indices]
    update_chain(sample_rows)

    scene_token = sample_rows[0]["scene_token"]
    for row in sample_rows:
        row["scene_token"] = scene_token

    sample_data_rows: list[JsonRow] = []
    sample_data_token_map: dict[str, str] = {}
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
        channel = channel_from_filename(row["filename"])
        copied["filename"] = renamed_data_filename(row["filename"], output_index)
        copied["sample_token"] = candidate_samples[next_key_index]["token"]
        copied["is_key_frame"] = candidate_index in key_candidate_index_set

        src = candidate_dir / row["filename"]
        dst = output_dir / copied["filename"]
        if src.exists():
            relink_or_copy(src, dst, copy_data=copy_data)
        else:
            raise FileNotFoundError(f"sample_data file does not exist: {src}")

        if copied.get("info_filename"):
            src_info = candidate_dir / row["info_filename"]
            copied["info_filename"] = renamed_data_filename(
                row["info_filename"], output_index
            )
            dst_info = output_dir / copied["info_filename"]
            if src_info.exists():
                relink_or_copy(src_info, dst_info, copy_data=copy_data)
            else:
                raise FileNotFoundError(f"sample_data info file does not exist: {src_info}")

        sample_data_token_map[row["token"]] = copied["token"]
        sample_data_rows.append(copied)
        by_channel[channel].append(copied)

    missing_key_tokens = key_sample_tokens - {
        row["sample_token"] for row in sample_data_rows if row.get("is_key_frame")
    }
    if missing_key_tokens:
        raise RuntimeError(f"missing keyframe sample_data for samples: {missing_key_tokens}")

    for channel_rows in by_channel.values():
        channel_rows.sort(key=lambda row: (row["timestamp"], row["filename"]))
        update_chain(channel_rows)

    sample_data_rows.sort(key=lambda row: (row["timestamp"], row["filename"]))
    return sample_rows, sample_data_rows, sample_data_token_map


def remap_sample_annotations(
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
        update_chain(rows)

    instance_by_token = {row["token"]: row for row in reference_instances}
    remapped_instances: list[JsonRow] = []
    for instance_token, rows in by_instance.items():
        source = instance_by_token[instance_token]
        copied = deepcopy(source)
        copied["nbr_annotations"] = len(rows)
        copied["first_annotation_token"] = rows[0]["token"]
        copied["last_annotation_token"] = rows[-1]["token"]
        remapped_instances.append(copied)

    remapped_annotations.sort(
        key=lambda row: (output_sample_order[row["sample_token"]], row["token"])
    )
    remapped_instances.sort(key=lambda row: row["token"])
    return remapped_annotations, remapped_instances


def remap_optional_annotation_table(
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


def filter_ego_pose(
    candidate_tables: dict[str, list[JsonRow]], sample_data: list[JsonRow]
) -> list[JsonRow]:
    used_tokens = {row["ego_pose_token"] for row in sample_data}
    return [row for row in candidate_tables["ego_pose"] if row["token"] in used_tokens]


def write_output_tables(
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
    tables: dict[str, list[JsonRow]] = {}

    tables["sample"] = sample_rows
    tables["sample_data"] = sample_data_rows
    tables["ego_pose"] = filter_ego_pose(candidate_tables, sample_data_rows)
    tables["calibrated_sensor"] = candidate_tables["calibrated_sensor"]
    tables["sensor"] = candidate_tables["sensor"]
    tables["vehicle_state"] = candidate_tables.get("vehicle_state", [])
    tables["log"] = candidate_tables.get("log", [])
    tables["map"] = candidate_tables.get("map", [])

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
            tables[name] = remap_optional_annotation_table(
                reference_tables[name],
                sample_token_map=reference_sample_token_map,
                sample_data_token_map=reference_sample_data_token_map,
            )

    for name, rows in tables.items():
        save_json(output_annotation_dir / f"{name}.json", rows)


def copy_non_annotation_files(candidate_dir: Path, output_dir: Path) -> None:
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


def align_dataset(
    *,
    candidate_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    stride: int,
    max_candidate_offset: int,
    max_reference_offset: int,
    max_abs_diff_ms: float,
    copy_data: bool,
    overwrite: bool,
    write_alignment_report: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    candidate_annotation_dir = annotation_dir(candidate_dir)
    reference_annotation_dir = annotation_dir(reference_dir)

    candidate_tables = {
        path.stem: load_json(path) for path in candidate_annotation_dir.glob("*.json")
    }
    reference_tables = {
        path.stem: load_json(path) for path in reference_annotation_dir.glob("*.json")
    }

    candidate_samples = candidate_samples_with_lidar(
        candidate_tables["sample"], candidate_tables["sample_data"]
    )
    reference_samples = reference_key_samples(
        reference_tables["sample"], reference_tables["sample_data"]
    )
    candidate_start, reference_start, _ = score_alignment(
        candidate_samples,
        reference_samples,
        stride=stride,
        max_candidate_offset=max_candidate_offset,
        max_reference_offset=max_reference_offset,
    )
    timestamp_matches, unmatched_reference_results = match_reference_samples_by_timestamp(
        candidate_samples,
        reference_samples,
        candidate_start=candidate_start,
        reference_start=reference_start,
        max_abs_diff_ms=max_abs_diff_ms,
    )

    if not timestamp_matches:
        raise RuntimeError("alignment produced no samples")

    num_keyframes = len(timestamp_matches)
    key_candidate_indices = [candidate_index for _, candidate_index, _ in timestamp_matches]
    output_candidate_start = key_candidate_indices[0]
    candidate_end = key_candidate_indices[-1]
    sample_token_map = {
        reference_samples[reference_index]["token"]: candidate_samples[candidate_index]["token"]
        for reference_index, candidate_index, _ in timestamp_matches
    }

    grouped_candidate_sample_data = group_sample_data_by_sample(
        candidate_tables["sample_data"]
    )
    grouped_reference_sample_data = group_sample_data_by_sample(
        reference_tables["sample_data"]
    )
    reference_sample_data_token_map: dict[str, str] = {}
    for reference_index, candidate_index, _ in timestamp_matches:
        reference_rows = grouped_reference_sample_data[reference_samples[reference_index]["token"]]
        candidate_rows = grouped_candidate_sample_data[candidate_samples[candidate_index]["token"]]
        candidate_by_channel = {
            channel_from_filename(row["filename"]): row for row in candidate_rows
        }
        for reference_row in reference_rows:
            channel = channel_from_filename(reference_row["filename"])
            if channel in candidate_by_channel:
                reference_sample_data_token_map[reference_row["token"]] = (
                    candidate_by_channel[channel]["token"]
                )

    sample_rows, sample_data_rows, candidate_sample_data_token_map = (
        build_filtered_candidate_tables(
            candidate_dir=candidate_dir,
            output_dir=output_dir,
            candidate_tables=candidate_tables,
            candidate_samples=candidate_samples,
            candidate_start=output_candidate_start,
            candidate_end=candidate_end,
            key_candidate_indices=key_candidate_indices,
            copy_data=copy_data,
        )
    )
    sample_annotations, instances = remap_sample_annotations(
        reference_tables.get("sample_annotation", []),
        reference_tables.get("instance", []),
        sample_token_map,
        {row["token"]: index for index, row in enumerate(sample_rows)},
    )

    write_output_tables(
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
    copy_non_annotation_files(candidate_dir, output_dir)

    used_diffs = [diff for _, _, diff in timestamp_matches]
    max_abs_diff_us = max(abs(diff) for diff in used_diffs)
    if max_abs_diff_us > max_abs_diff_ms * 1000:
        print(
            "warning: selected alignment exceeds tolerance: "
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
        for reference_index, candidate_index, diff in timestamp_matches
    ]
    report = {
        "candidate_dir": str(candidate_dir),
        "reference_dir": str(reference_dir),
        "output_dir": str(output_dir),
        "stride": stride,
        "candidate_start_index": candidate_start,
        "output_candidate_start_index": output_candidate_start,
        "reference_start_index": reference_start,
        "num_keyframes": num_keyframes,
        "num_unmatched_reference_keyframes": len(unmatched_reference_results),
        "num_samples": len(sample_rows),
        "num_sample_data": len(sample_data_rows),
        "num_sample_annotations": len(sample_annotations),
        "max_abs_diff_ms": max_abs_diff_ms,
        "first_candidate_timestamp": candidate_samples[key_candidate_indices[0]]["timestamp"],
        "first_reference_timestamp": reference_samples[reference_start]["timestamp"],
        "first_timestamp_diff_us": used_diffs[0],
        "median_abs_timestamp_diff_us": median(abs(diff) for diff in used_diffs),
        "max_abs_timestamp_diff_us": max_abs_diff_us,
        "timestamp_diffs_us": used_diffs,
        "key_candidate_indices": key_candidate_indices,
        "alignment_results": alignment_results,
        "unmatched_reference_results": unmatched_reference_results,
        "candidate_sample_data_token_map_size": len(candidate_sample_data_token_map),
    }
    if write_alignment_report:
        save_json(output_dir / "alignment_report.json", report)
    return report


class AlignNonAnnotatedT4ToReferenceConverter(AbstractConverter[list[dict[str, Any]]]):
    def __init__(
        self,
        input_base: str,
        reference_base: str,
        output_base: str,
        *,
        stride: int = DEFAULT_KEYFRAME_STRIDE,
        max_candidate_offset: int = 30,
        max_reference_offset: int = 10,
        max_abs_diff_ms: float = DEFAULT_MAX_ABS_DIFF_MS,
        copy_data: bool = False,
        write_alignment_report: bool = True,
        overwrite_mode: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(input_base=input_base, output_base=output_base)
        self._input_base = Path(input_base)
        self._reference_base = Path(reference_base)
        self._output_base = Path(output_base)
        self._stride = stride
        self._max_candidate_offset = max_candidate_offset
        self._max_reference_offset = max_reference_offset
        self._max_abs_diff_ms = max_abs_diff_ms
        self._copy_data = copy_data
        self._write_alignment_report = write_alignment_report
        self._overwrite_mode = overwrite_mode
        self._logger = configure_logger(modname=__name__) if logger is None else logger

    def convert(self) -> list[dict[str, Any]]:
        input_is_single_scene, input_scenes = discover_t4_scenes(self._input_base)
        _, reference_scenes = discover_t4_scenes(self._reference_base)

        if not input_scenes:
            raise FileNotFoundError(f"no T4 dataset scenes found in {self._input_base}")
        if not reference_scenes:
            raise FileNotFoundError(
                f"no reference T4 dataset scenes found in {self._reference_base}"
            )

        reference_by_name = {scene_name: scene_dir for scene_name, scene_dir in reference_scenes}
        single_reference_dir = reference_scenes[0][1] if len(reference_scenes) == 1 else None

        reports: list[dict[str, Any]] = []
        for scene_name, candidate_dir in input_scenes:
            if scene_name in reference_by_name:
                reference_dir = reference_by_name[scene_name]
            elif single_reference_dir is not None and len(input_scenes) == 1:
                reference_dir = single_reference_dir
            else:
                raise FileNotFoundError(
                    f"no matching reference scene for {scene_name} in {self._reference_base}"
                )

            output_dir = (
                self._output_base if input_is_single_scene else self._output_base / scene_name
            )
            self._logger.info(
                "[BEGIN] Aligning non-annotated T4 "
                f"({candidate_dir}) to reference ({reference_dir}) into {output_dir}"
            )
            reports.append(
                align_dataset(
                    candidate_dir=candidate_dir,
                    reference_dir=reference_dir,
                    output_dir=output_dir,
                    stride=self._stride,
                    max_candidate_offset=self._max_candidate_offset,
                    max_reference_offset=self._max_reference_offset,
                    max_abs_diff_ms=self._max_abs_diff_ms,
                    copy_data=self._copy_data,
                    overwrite=self._overwrite_mode,
                    write_alignment_report=self._write_alignment_report,
                )
            )
            self._logger.info(
                "[DONE] Aligning non-annotated T4 "
                f"({candidate_dir}) to reference ({reference_dir}) into {output_dir}"
            )

        return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim a non-annotated T4 dataset to the timestamp cadence of a "
            "reference annotated T4 dataset and remap the reference annotations."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=DEFAULT_KEYFRAME_STRIDE)
    parser.add_argument("--max-candidate-offset", type=int, default=30)
    parser.add_argument("--max-reference-offset", type=int, default=10)
    parser.add_argument("--max-abs-diff-ms", type=float, default=DEFAULT_MAX_ABS_DIFF_MS)
    parser.add_argument(
        "--copy-data",
        action="store_true",
        help="copy data files instead of hardlinking them when possible",
    )
    parser.add_argument(
        "--no-alignment-report",
        action="store_true",
        help="do not write alignment_report.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = AlignNonAnnotatedT4ToReferenceConverter(
        input_base=str(args.candidate),
        reference_base=str(args.reference),
        output_base=str(args.output),
        stride=args.stride,
        max_candidate_offset=args.max_candidate_offset,
        max_reference_offset=args.max_reference_offset,
        max_abs_diff_ms=args.max_abs_diff_ms,
        copy_data=args.copy_data,
        write_alignment_report=not args.no_alignment_report,
        overwrite_mode=args.overwrite,
    )
    print(json.dumps(converter.convert(), indent=2))


if __name__ == "__main__":
    main()
