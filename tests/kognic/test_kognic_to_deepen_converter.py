import json
from pathlib import Path

import numpy as np
import pytest

from perception_dataset.kognic.kognic_to_deepen_converter import KognicToDeepenConverter


def _write_openlabel(path: Path) -> None:
    document = {
        "openlabel": {
            "metadata": {"schema_version": "1.0.0"},
            "ontologies": {"0": {"classifications": {"2": "road", "9": "PassengerCar"}}},
            "objects": {
                "car-uuid": {
                    "name": "car",
                    "type": "PassengerCar",
                    "object_data": {
                        "text": [{"name": "vehicle_state", "val": "driving"}],
                        "num": [{"name": "classification_id", "val": 42}],
                    },
                },
                "segmentation": {
                    "name": "lidarseg",
                    "type": "3DPointCloudSegmentation",
                },
            },
            "frames": {
                "0": {
                    "frame_properties": {"external_id": "12"},
                    "objects": {
                        "car-uuid": {
                            "object_data": {
                                "cuboid": [
                                    {
                                        "val": [1, 2, 3, 0, 0, 0, 1, 4, 5, 6],
                                        "attributes": {
                                            "text": [{"name": "stream", "val": "LIDAR"}]
                                        },
                                    }
                                ],
                                "bbox": [
                                    {
                                        "val": [50, 40, 20, 10],
                                        "attributes": {
                                            "text": [{"name": "stream", "val": "CAM_FRONT"}]
                                        },
                                    }
                                ],
                                "boolean": [{"name": "occluded", "val": False}],
                            }
                        },
                        "segmentation": {
                            "object_data": {
                                "binary": [
                                    {
                                        "name": "labels",
                                        "encoding": "rle",
                                        "val": "#2V2#1V0",
                                        "attributes": {
                                            "text": [{"name": "stream", "val": "LIDAR_A"}]
                                        },
                                    },
                                    {
                                        "name": "labels",
                                        "encoding": "rle",
                                        "val": "#2V42",
                                        "attributes": {
                                            "text": [{"name": "stream", "val": "LIDAR_B"}]
                                        },
                                    },
                                ]
                            }
                        },
                    },
                }
            },
        }
    }
    path.write_text(json.dumps(document))


def test_converts_bbox_and_semantic_segmentation(tmp_path: Path):
    input_path = tmp_path / "scene.json"
    output_path = tmp_path / "output"
    _write_openlabel(input_path)

    result = KognicToDeepenConverter(
        input_base=str(input_path),
        output_base=str(output_path),
        dataset_id="dataset-id",
        category_map={"PassengerCar": "car"},
        sensor_map={"CAM_FRONT": "camera_0"},
        lidar_streams=["LIDAR_B", "LIDAR_A"],
    ).convert()

    labels = json.loads((output_path / "scene.json").read_text())["labels"]
    assert len(labels) == 2
    cuboid = next(label for label in labels if label["label_type"] == "3d_bbox")
    assert cuboid["dataset_id"] == "dataset-id"
    assert cuboid["label_category_id"] == "car"
    assert cuboid["label_id"] == "car:1"
    assert cuboid["attributes"] == {"vehicle_state": "driving", "occluded": False}
    assert cuboid["three_d_bbox"]["w"] == 4
    assert cuboid["three_d_bbox"]["l"] == 5
    assert cuboid["three_d_bbox"]["h"] == 6
    assert cuboid["three_d_bbox"]["quaternion"]["z"] == pytest.approx(2**-0.5)
    bbox = next(label for label in labels if label["label_type"] == "box")
    assert bbox["sensor_id"] == "camera_0"
    assert bbox["box"] == [40, 35, 20, 10]

    lidarseg = json.loads((output_path / "scene_lidarseg.json").read_text())
    assert lidarseg[0]["file_id"] == "12.pcd"
    assert lidarseg[0]["paint_categories"] == ["road", "car"]
    assert lidarseg[0]["total_lidar_points"] == 5
    labels_bin = np.fromfile(output_path / lidarseg[0]["lidarseg_anno_file"], dtype=np.uint8)
    assert labels_bin.tolist() == [2, 2, 1, 1, 0]
    assert result.items[0].bbox_annotation_path == str(output_path / "scene.json")
    assert result.items[0].lidarseg_annotation_path == str(output_path / "scene_lidarseg.json")


def test_rejects_invalid_segmentation_rle(tmp_path: Path):
    input_path = tmp_path / "scene.json"
    _write_openlabel(input_path)
    document = json.loads(input_path.read_text())
    binary = document["openlabel"]["frames"]["0"]["objects"]["segmentation"]["object_data"][
        "binary"
    ][0]
    binary["val"] = "#2V2-invalid"
    input_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="invalid Kognic RLE"):
        KognicToDeepenConverter(str(input_path), str(tmp_path / "output")).convert()
