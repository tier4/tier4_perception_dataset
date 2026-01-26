import json
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List

from t4_devkit.schema.tables import (
    Attribute,
    Category,
    Instance,
    ObjectAnn,
    SampleAnnotation,
    SurfaceAnn,
    Visibility,
)

from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator
from perception_dataset.t4_dataset.table_handler import TableHandler, get_schema_name
from perception_dataset.t4_dataset.resolver.duplicated_annotation_remover import (
    DuplicatedAnnotationRemover,
)
from perception_dataset.t4_dataset.resolver.keyframe_consistency_resolver import (
    KeyFrameConsistencyResolver,
)


def _load_json(filepath: str) -> Any:
    with open(filepath) as f:
        data = json.load(f)
    return data


class AnnotationFilesUpdater(AnnotationFilesGenerator):
    def __init__(
        self,
        with_camera: bool = True,
        description: Dict[str, Dict[str, str]] = ...,
        surface_categories: List[str] = [],
    ):
        super().__init__(with_camera, description, surface_categories)
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

        # Remove duplicated annotations
        DuplicatedAnnotationRemover().remove_duplicated_annotation(output_dir)
        # fix non-keyframe (no-labeled frame) in t4 dataset
        KeyFrameConsistencyResolver().inspect_and_fix_t4_segment(Path(output_dir))

    def _init_table_from_json(self, anno_dir: str) -> None:
        self._attribute_table = TableHandler.from_json(
            schema_type=Attribute,
            filepath=osp.join(anno_dir, "attribute.json"),
        )

        self._category_table = TableHandler.from_json(
            schema_type=Category,
            filepath=osp.join(anno_dir, "category.json"),
        )

        self._instance_table = TableHandler.from_json(
            schema_type=Instance,
            filepath=osp.join(anno_dir, "instance.json"),
        )

        self._visibility_table = TableHandler.from_json(
            schema_type=Visibility,
            filepath=osp.join(anno_dir, "visibility.json"),
        )
        
        # Ensure default visibility levels exist
        for level, desc in self.description.get(
            "visibility",
            {
                "v0-40": "visibility of whole object is between 0 and 40%",
                "v40-60": "visibility of whole object is between 40 and 60%",
                "v60-80": "visibility of whole object is between 60 and 80%",
                "v80-100": "visibility of whole object is between 80 and 100%",
                "none": "visibility isn't available",
            },
        ).items():
            if not self._visibility_table.get_token_from_field("level", level):
                self._visibility_table.insert_into_table(
                    level=level,
                    description=desc,
                )

        sample_annotation_filepath = osp.join(anno_dir, f"{get_schema_name(SampleAnnotation)}.json")
        if osp.exists(sample_annotation_filepath):
            self._sample_annotation_table = TableHandler.from_json(
                schema_type=SampleAnnotation,
                filepath=sample_annotation_filepath,
            )

        objectann_filepath = osp.join(anno_dir, f"{get_schema_name(ObjectAnn)}.json")
        if osp.exists(objectann_filepath):
            self._object_ann_table = TableHandler.from_json(
                schema_type=ObjectAnn,
                filepath=objectann_filepath,
            )

        surfaceann_filepath = osp.join(anno_dir, f"{get_schema_name(SurfaceAnn)}.json")
        if osp.exists(surfaceann_filepath):
            self._surface_ann_table = TableHandler.from_json(
                schema_type=SurfaceAnn,
                filepath=surfaceann_filepath,
            )
