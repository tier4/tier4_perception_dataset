import json
import os.path as osp
from typing import Any, Dict

from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from perception_dataset.t4_dataset.classes import (
    AttributeTable,
    CategoryTable,
    InstanceTable,
    ObjectAnnTable,
    SampleAnnotationTable,
    SurfaceAnnTable,
    VisibilityTable,
)


def _load_json(filepath: str) -> Any:
    with open(filepath) as f:
        data = json.load(f)
    return data


class AnnotationFilesUpdater(AnnotationFilesGenerator):
    def __init__(self, with_camera: bool = True, description: Dict[str, Dict[str, str]] = ...):
        super().__init__(with_camera, description)
        self.description = description

    def convert_one_scene(
        self,
        input_dir: str,
        output_dir: str,
        scene_anno_dict: dict[str, list[dict[str, Any]]],
        dataset_name: str,
    ) -> None:
        anno_dir = osp.join(input_dir, "annotation")
        if not osp.exists(anno_dir):
            raise ValueError(f"Annotations files doesn't exist in {anno_dir}")

        # Load existence annotation files
        self._init_table_from_json(anno_dir=anno_dir)

        super().convert_one_scene(
            input_dir=input_dir,
            output_dir=output_dir,
            scene_anno_dict=scene_anno_dict,
            dataset_name=dataset_name,
        )

    def _init_table_from_json(self, anno_dir: str) -> None:
        self._attribute_table = AttributeTable.from_json(
            filepath=osp.join(anno_dir, AttributeTable.FILENAME),
            name_to_description={},
            default_value="",
        )

        self._category_table = CategoryTable.from_json(
            filepath=osp.join(anno_dir, CategoryTable.FILENAME),
            name_to_description={},
            default_value="",
        )

        self._instance_table = InstanceTable.from_json(
            filepath=osp.join(anno_dir, InstanceTable.FILENAME)
        )

        self._visibility_table = VisibilityTable.from_json(
            filepath=osp.join(anno_dir, VisibilityTable.FILENAME),
            level_to_description=self.description.get(
                "visibility",
                {
                    "v0-40": "visibility of whole object is between 0 and 40%",
                    "v40-60": "visibility of whole object is between 40 and 60%",
                    "v60-80": "visibility of whole object is between 60 and 80%",
                    "v80-100": "visibility of whole object is between 80 and 100%",
                    "none": "visibility isn't available",
                },
            ),
            default_value="",
        )

        if osp.exists(osp.join(anno_dir, SampleAnnotationTable.FILENAME)):
            self._sample_annotation_table = SampleAnnotationTable.from_json(
                osp.join(anno_dir, SampleAnnotationTable.FILENAME)
            )

        if osp.exists(osp.join(anno_dir, ObjectAnnTable.FILENAME)):
            self._object_ann_table = ObjectAnnTable.from_json(
                osp.join(anno_dir, ObjectAnnTable.FILENAME)
            )

        if osp.exists(osp.join(anno_dir, SurfaceAnnTable.FILENAME)):
            self._surface_ann_table = SurfaceAnnTable.from_json(
                osp.join(anno_dir, SurfaceAnnTable.FILENAME)
            )
