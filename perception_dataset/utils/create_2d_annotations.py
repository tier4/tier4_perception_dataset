from typing import List

import numpy as np
from t4_devkit import Tier4
from t4_devkit.schema.tables import Instance, ObjectAnn, SampleAnnotation, SampleData

from perception_dataset.t4_dataset.table_handler import TableHandler


def create_2d_annotations(
    dataroot: str,
    camera_sensor_channels: List[str],
    annotation_table: TableHandler[SampleAnnotation],
    sample_data_table: TableHandler[SampleData],
    object_ann_table: TableHandler[ObjectAnn],
    instance_table: TableHandler[Instance],
):
    t4_dataset = Tier4(data_root=dataroot, verbose=False)
    for sample in t4_dataset.sample:
        for camera_sensor in camera_sensor_channels:
            camera_channel = camera_sensor["channel"]
            camera_token = sample.data[camera_channel]
            _, boxes, camera_intrinsic = t4_dataset.get_sample_data(camera_token)

            # map 3d cuboids to 2d bounding boxes
            for box in boxes:
                corners = box.corners()
                x = corners[0, :]
                y = corners[1, :]
                z = corners[2, :]
                x_y_z = np.array((x, y, z))
                orthographic = np.dot(camera_intrinsic, x_y_z)
                perspective_x = orthographic[0] / orthographic[2]
                perspective_y = orthographic[1] / orthographic[2]

                min_x = int(np.min(perspective_x))
                max_x = int(np.max(perspective_x))
                min_y = int(np.min(perspective_y))
                max_y = int(np.max(perspective_y))

                sample_annotation_record = annotation_table.get_record_from_token(box.token)
                instance_token = sample_annotation_record.instance_token
                attribute_tokens = sample_annotation_record.attribute_tokens
                category_token = instance_table.get_record_from_token(
                    instance_token
                ).category_token
                sample_data_record = sample_data_table.get_record_from_token(camera_token)
                imsize = [sample_data_record.width, sample_data_record.height]

                object_ann_table.insert_into_table(
                    sample_data_token=camera_token,
                    instance_token=instance_token,
                    category_token=category_token,
                    attribute_tokens=attribute_tokens,
                    bbox=[min_x, min_y, max_x, max_y],
                    mask={"size": imsize},
                )
