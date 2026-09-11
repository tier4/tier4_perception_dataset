import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from t4_devkit.schema.tables import Category

from perception_dataset.kognic.openlabel_to_t4_converter import (
    OpenLabelToT4Converter,
    _SampleIndex,
    _decode_rle_labels,
    _remap_labels,
)
from perception_dataset.t4_dataset.table_handler import TableHandler


def _converter(tmp_path: Path) -> OpenLabelToT4Converter:
    """Create an OpenLabelToT4Converter instance with temporary paths.

    Args:
        tmp_path (Path): Temporary directory path provided by pytest.

    Returns:
        OpenLabelToT4Converter: An instance of OpenLabelToT4Converter initialized with temporary paths.
    """
    return OpenLabelToT4Converter(
        input_base=str(tmp_path / "input"),
        output_base=str(tmp_path / "output"),
        annotation_base=str(tmp_path / "annotations"),
    )


def _write_openlabel(path: Path, annotation_type: str) -> None:
    """Write a minimal OpenLabel JSON file with the specified annotation type.

    Args:
        path (Path): Path to the OpenLabel JSON file to be created.
        annotation_type (str): Type of annotation, either "boxes" or "segmentation".
    """
    metadata = {"annotation_type": "semseg"} if annotation_type == "segmentation" else {}
    path.write_text(json.dumps({"openlabel": {"metadata": metadata, "frames": {}}}))


def test_annotation_selection_applies_boxes_before_segmentation(tmp_path: Path):
    """Test that annotation selection applies boxes before segmentation.

    The candidates are deliberately supplied in the opposite order. Applying
    boxes first lets their categories seed ``category.json`` before the
    segmentation ontology adds and reconciles its category indices.

    Args:
        tmp_path (Path): Pytest directory used to store the temporary
            OpenLABEL files.
    """
    converter = _converter(tmp_path)
    boxes = tmp_path / "boxes.json"
    segmentation = tmp_path / "segmentation.json"
    _write_openlabel(boxes, "boxes")
    _write_openlabel(segmentation, "segmentation")

    assert converter._select_by_type(tmp_path / "scene", [segmentation, boxes]) == [
        boxes,
        segmentation,
    ]


def test_annotation_selection_skips_ambiguous_type_but_keeps_other_type(tmp_path: Path):
    """Test that one ambiguous annotation type does not discard another type.

    Two box files claim the same scene, so choosing either one would be
    arbitrary. The converter must skip those boxes while retaining the single,
    unambiguous segmentation file.

    Args:
        tmp_path (Path): Pytest directory used to store the candidate
            OpenLABEL files.
    """
    converter = _converter(tmp_path)
    boxes_a = tmp_path / "boxes-a.json"
    boxes_b = tmp_path / "boxes-b.json"
    segmentation = tmp_path / "segmentation.json"
    _write_openlabel(boxes_a, "boxes")
    _write_openlabel(boxes_b, "boxes")
    _write_openlabel(segmentation, "segmentation")

    assert converter._select_by_type(
        tmp_path / "scene", [boxes_a, segmentation, boxes_b]
    ) == [segmentation]


def test_lidar_sample_selection_prefers_exact_sample_timestamp():
    """Test that the LiDAR record with the exact sample timestamp is selected.

    A sample can own both its keyframe and nearby intermediate sweeps. This
    case ensures the record captured at the sample timestamp wins, preventing
    annotations from being attached to a neighbouring sweep.
    """
    sample = [SimpleNamespace(token="sample", timestamp=100)]
    sample_data = [
        SimpleNamespace(
            token="sweep",
            sample_token="sample",
            channel="LIDAR_CONCAT",
            timestamp=101,
            is_key_frame=False,
        ),
        SimpleNamespace(
            token="keyframe",
            sample_token="sample",
            channel="LIDAR_CONCAT",
            timestamp=100,
            is_key_frame=True,
        ),
    ]

    selected = OpenLabelToT4Converter._select_lidar_sample_data(
        sample, sample_data, "LIDAR_CONCAT"
    )

    assert selected["sample"].token == "keyframe"


def test_lidar_sample_selection_rejects_ambiguous_keyframes():
    """Test that a LiDAR sample selection rejects ambiguous keyframes.

    Ambiguous means that two LiDAR records belong to the same sample, both are
    marked as keyframes, and neither has the exact sample timestamp. There is
    no safe way to choose between them, so the sample must be excluded instead
    of relying on table order.
    """
    sample = [SimpleNamespace(token="sample", timestamp=100)]
    sample_data = [
        SimpleNamespace(
            token=token,
            sample_token="sample",
            channel="LIDAR_CONCAT",
            timestamp=timestamp,
            is_key_frame=True,
        )
        for token, timestamp in (("first", 99), ("second", 101))
    ]

    selected = OpenLabelToT4Converter._select_lidar_sample_data(
        sample, sample_data, "LIDAR_CONCAT"
    )

    assert selected == {}


def test_source_lidar_timestamp_maps_to_its_concat_sample(tmp_path: Path):
    """Test that a source LiDAR capture time maps back to its fused sample.

    A source sensor can be captured slightly before or after the fused concat
    timestamp. The converter reads the source timestamp from
    ``LIDAR_CONCAT_INFO`` and indexes it to the same T4 sample, preventing a
    valid per-sensor annotation from being treated as an unmatched frame.

    Args:
        tmp_path (Path): Pytest directory used to create the temporary concat
            metadata file.
    """

    info_path = tmp_path / "data/LIDAR_CONCAT_INFO/0.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "stamp": {"sec": 2, "nanosec": 0},
                "sources": [
                    {
                        "sensor_token": "front",
                        "idx_begin": 0,
                        "length": 1,
                        "stamp": {"sec": 2, "nanosec": 50_000_000},
                    }
                ]
            }
        )
    )
    record = {"info_filename": "data/LIDAR_CONCAT_INFO/0.json"}
    entry = ("sample", {"token": "pose"})
    by_timestamp = {}

    OpenLabelToT4Converter._index_source_timestamps(
        tmp_path,
        {"sample": record},
        {"sample": entry},
        by_timestamp,
    )

    assert by_timestamp[2_050_000] == entry


def test_sample_index_does_not_fall_back_to_positional_external_id():
    """Test that a frame number cannot replace a missing sensor timestamp.

    The OpenLABEL frame has an external ID that looks like a valid list index,
    but it contains no LiDAR stream URI. The match must fail instead of using
    that position and silently attaching the annotation to an unrelated sample.
    """
    index = _SampleIndex(
        by_timestamp_us={1_000_000: ("sample", {"token": "pose"})},
        sample_data_by_sample={"sample": {"token": "sample-data"}},
    )
    frame = {"frame_properties": {"external_id": "0", "streams": {}}}

    assert index.match(frame, "0", "LIDAR_CONCAT") is None


def test_segmentation_categories_reconcile_existing_bbox_indices():
    """Test that segmentation categories safely reuse a table created by boxes.

    Existing box categories occupy indices needed by background and the
    segmentation ontology. The converter must reserve the segmentation indices,
    reuse categories with matching names, and move unrelated box categories so
    every stored point label resolves to exactly one category.
    """
    categories = TableHandler(Category)
    categories.insert_into_table(name="car", description="", index=0)
    categories.insert_into_table(name="truck", description="", index=1)

    mapping = OpenLabelToT4Converter._assign_segmentation_categories(
        categories, {1: "car", 2: "road"}
    )

    by_name = {record.name: record.index for record in categories.to_records()}
    assert mapping == {1: 1, 2: 2}
    assert by_name["background"] == 0
    assert by_name["car"] == 1
    assert by_name["road"] == 2
    assert by_name["truck"] > 2
    assert len(set(by_name.values())) == len(by_name)


def test_single_tagged_lidar_stream_is_written_into_its_source_slice(tmp_path: Path):
    """Test that one tagged LiDAR stream is written only into its concat slice.

    Even though the frame contains only one RLE block, its stream tag says the
    labels belong to the rear sensor. The first three front-sensor points must
    remain background while the labels fill the two-point rear slice.

    Args:
        tmp_path (Path): Pytest directory used to create the temporary concat
            metadata file.
    """
    converter = _converter(tmp_path)
    info_path = tmp_path / "data/LIDAR_CONCAT_INFO/0.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"sensor_token": "front", "idx_begin": 0, "length": 3},
                    {"sensor_token": "rear", "idx_begin": 3, "length": 2},
                ]
            }
        )
    )
    sample_data = {
        "filename": "data/LIDAR_CONCAT/0.pcd.bin",
        "info_filename": "data/LIDAR_CONCAT_INFO/0.json",
    }

    labels = converter._stitch_stream_labels(
        tmp_path,
        sample_data,
        {"LIDAR_REAR": "#2V1"},
        {1: 9},
        num_points=5,
        channel_by_sensor_token={"front": "LIDAR_FRONT", "rear": "LIDAR_REAR"},
        frame_key="0",
    )

    assert labels is not None
    assert labels.tolist() == [0, 0, 0, 9, 9]


@pytest.mark.parametrize("value", ["garbage#2V1", "#2V1trailing", ""])
def test_rle_decoder_rejects_malformed_content(value: str):
    """Test that the RLE decoder rejects text outside the expected token format.

    Each value is empty or contains extra text around an otherwise valid token.
    Rejecting the whole value prevents partially decoded labels from being
    silently aligned with the wrong points.

    Args:
        value (str): Malformed RLE value supplied by the parametrized test.
    """
    with pytest.raises(ValueError, match="not fully covered"):
        _decode_rle_labels(value, max_points=10, frame_key="0")


def test_rle_decoder_rejects_oversized_count_before_expansion():
    """Test that an RLE declaring more labels than points is rejected early.

    The declared run is far larger than the five-point target cloud. The
    decoder must compare counts before calling ``numpy.repeat``, preventing
    corrupted or hostile input from causing an excessive memory allocation.
    """
    with pytest.raises(ValueError, match="refusing to expand"):
        _decode_rle_labels("#1000000000V1", max_points=5, frame_key="0")


def test_sparse_large_raw_label_is_remapped_without_large_lookup_table():
    """Test that a very large raw class ID is remapped with bounded memory.

    The label value is one billion even though the frame has only two points.
    The converter must map the values that actually occur rather than allocate
    a lookup table whose size is controlled by the largest raw label.
    """
    labels = np.array([0, 1_000_000_000], dtype=np.int64)

    assert _remap_labels(labels, {1_000_000_000: 7}, "0").tolist() == [0, 7]
