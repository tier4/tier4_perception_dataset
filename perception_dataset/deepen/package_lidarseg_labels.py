"""Package lidarseg (paint-3D) annotations for import into Deepen via the GUI.

This consumes the output of the `convert_annotated_t4_to_deepen` task:
`<scene_name>_lidarseg.json` and the per-frame label binaries under `lidarseg/`.
Scenes are discovered automatically from the `*_lidarseg.json` files in
`--input_base`.

For each scene, the per-frame binaries are concatenated in numeric `file_id`
order into a single per-point uint8 buffer and written as a GUI-importable
package:

    <package_dir>/<scene_name>/
        binary_export.bin
        compressed_binary_export.deflate
        metadata.json

Import the package in Deepen from the dataset's three-dot menu -> Import Labels.
"""

import argparse
from glob import glob
import json
import os
import os.path as osp
from typing import Any, Dict, List, Tuple
import zlib

from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


def load_lidarseg_scene(input_base: str, scene_name: str) -> Tuple[bytes, Dict[str, Any]]:
    """Concatenate the per-frame label binaries of one scene.

    Returns the raw uint8 buffer (frames in numeric `file_id` order) and the
    paint metadata (`format`, `paint_categories`, `frame_sizes`).
    """
    anno_file = osp.join(input_base, f"{scene_name}_lidarseg.json")
    with open(anno_file) as f:
        annos_info: List[Dict[str, Any]] = json.load(f)

    if len(annos_info) == 0:
        raise ValueError(f"{anno_file} contains no lidarseg annotations.")

    paint_categories = annos_info[0]["paint_categories"]
    for anno in annos_info:
        if anno["paint_categories"] != paint_categories:
            raise ValueError(f"Inconsistent paint_categories across frames in {anno_file}.")

    annos_info = sorted(annos_info, key=lambda anno: int(anno["file_id"].split(".")[0]))

    buffer = bytearray()
    frame_sizes: List[int] = []
    for anno in annos_info:
        bin_path = osp.join(input_base, anno["lidarseg_anno_file"])
        with open(bin_path, "rb") as f:
            frame_buffer = f.read()
        if len(frame_buffer) != anno["total_lidar_points"]:
            raise ValueError(
                f"{bin_path}: size {len(frame_buffer)} does not match "
                f"total_lidar_points {anno['total_lidar_points']}."
            )
        buffer.extend(frame_buffer)
        frame_sizes.append(len(frame_buffer))

    paint_metadata = {
        "format": "deflate-compression",
        "paint_categories": paint_categories,
        "frame_sizes": frame_sizes,
    }
    return bytes(buffer), paint_metadata


def write_package(package_dir: str, scene_name: str, buffer: bytes, paint_metadata: Dict) -> None:
    """Write a GUI-importable package: raw binary, deflate binary and metadata.json."""
    scene_dir = osp.join(package_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    with open(osp.join(scene_dir, "binary_export.bin"), "wb") as f:
        f.write(buffer)
    with open(osp.join(scene_dir, "compressed_binary_export.deflate"), "wb") as f:
        f.write(zlib.compress(buffer))
    # NOTE: the GUI package metadata does not include frame_sizes.
    metadata = {
        "format": paint_metadata["format"],
        "paint_categories": paint_metadata["paint_categories"],
    }
    with open(osp.join(scene_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Package written to {scene_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_base",
        type=str,
        required=True,
        help="Directory containing <scene_name>_lidarseg.json and lidarseg/ binaries "
        "(the output_base of convert_annotated_t4_to_deepen).",
    )
    parser.add_argument(
        "--package_dir",
        type=str,
        default="./deepen_packages",
        help="Output directory; one package sub-directory is written per scene.",
    )
    args = parser.parse_args()

    scene_names = sorted(
        osp.basename(path)[: -len("_lidarseg.json")]
        for path in glob(osp.join(args.input_base, "*_lidarseg.json"))
    )
    if not scene_names:
        raise SystemExit(f"No *_lidarseg.json found in {args.input_base}.")

    for scene_name in scene_names:
        buffer, paint_metadata = load_lidarseg_scene(args.input_base, scene_name)
        logger.info(
            f"{scene_name}: {len(paint_metadata['frame_sizes'])} frames, "
            f"{len(buffer)} points, {len(paint_metadata['paint_categories'])} paint categories."
        )
        write_package(args.package_dir, scene_name, buffer, paint_metadata)


if __name__ == "__main__":
    main()
