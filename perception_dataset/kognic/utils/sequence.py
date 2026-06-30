"""Shared T4 sequence discovery for the Kognic converters.

Both ``T4ToKognicConverter`` and ``T4ToOpenLabelConverter`` walk an
``input_base`` for T4 sequence roots and pair each with an output directory
under ``output_base``. The logic lives here so the two converters stay in sync.
"""

from pathlib import Path
from typing import List, Tuple

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)

__all__ = ["is_sequence_root", "find_sequence_roots", "iter_scene_pairs"]

# Annotation tables every T4 sequence root must contain.
_REQUIRED_ANNOTATIONS = (
    "sensor.json",
    "calibrated_sensor.json",
    "sample.json",
    "sample_data.json",
    "ego_pose.json",
)


def is_sequence_root(path: Path) -> bool:
    """True if *path* is a T4 sequence root (``annotation/`` + ``data/`` with the required tables)."""
    annotation_dir = path / "annotation"
    data_dir = path / "data"
    return (
        path.is_dir()
        and annotation_dir.is_dir()
        and data_dir.is_dir()
        and all((annotation_dir / name).exists() for name in _REQUIRED_ANNOTATIONS)
    )


def find_sequence_roots(root: Path) -> List[Path]:
    """All T4 sequence roots at or below *root* (skipping ``extracted_data``)."""
    if is_sequence_root(root):
        return [root]

    return sorted(
        path
        for path in root.rglob("*")
        if is_sequence_root(path) and "extracted_data" not in path.parts
    )


def iter_scene_pairs(input_base: Path, output_base: Path) -> List[Tuple[Path, Path]]:
    """Pair every discovered T4 sequence root under *input_base* with its output directory.

    If *input_base* is itself a sequence root it is paired with
    ``output_base/<name>``. Otherwise each immediate subdirectory is searched:
    a single sequence root maps to ``output_base/<item>``, while multiple roots
    nest under ``output_base/<item>/<seq_root>``.
    """
    input_base = Path(input_base)
    output_base = Path(output_base)

    if is_sequence_root(input_base):
        return [(input_base, output_base / input_base.name)]

    pairs: List[Tuple[Path, Path]] = []
    for item in sorted(p for p in input_base.iterdir() if p.is_dir()):
        if item.name == "extracted_data":
            continue

        seq_roots = find_sequence_roots(item)
        if not seq_roots:
            logger.warning(f"No T4 sequence root found under {item}; skipping")
            continue

        if len(seq_roots) == 1:
            pairs.append((seq_roots[0], output_base / item.name))
        else:
            for seq_root in seq_roots:
                pairs.append((seq_root, output_base / item.name / seq_root.name))

    return pairs
