import os.path as osp
from typing import Any

from perception_dataset.t4_dataset.annotation_files_generator import AnnotationFilesGenerator


class AnnotationFilesUpdater(AnnotationFilesGenerator):
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
        self._attribute_table.insert_from_json(osp.join(anno_dir, self._attribute_table.FILENAME))
        self._category_table.insert_from_json(osp.join(anno_dir, self._category_table.FILENAME))
        self._instance_table.insert_from_json(osp.join(anno_dir, self._instance_table.FILENAME))
        self._sample_annotation_table.insert_from_json(
            osp.join(anno_dir, self._sample_annotation_table.FILENAME)
        )
        self._object_ann_table.insert_from_json(
            osp.join(anno_dir, self._object_ann_table.FILENAME)
        )
        self._surface_ann_table.insert_from_json(
            osp.join(anno_dir, self._surface_ann_table.FILENAME)
        )

        super().convert_one_scene(
            input_dir=input_dir,
            output_dir=output_dir,
            scene_anno_dict=scene_anno_dict,
            dataset_name=dataset_name,
        )
