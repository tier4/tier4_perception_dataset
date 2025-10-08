from typing import List

import numpy as np
from t4_devkit import Tier4

from perception_dataset.t4_dataset.classes.instance import InstanceTable
from perception_dataset.t4_dataset.classes.object_ann import ObjectAnnRecord, ObjectAnnTable
from perception_dataset.t4_dataset.classes.sample_annotation import SampleAnnotationTable
from perception_dataset.t4_dataset.classes.sample_data import SampleDataTable


def create_2d_annotations(
    dataroot: str,
    camera_sensor_channels: List[str],
    annotation_table: SampleAnnotationTable,
    sample_data_table: SampleDataTable,
    object_ann_table: ObjectAnnTable,
    instance_table: InstanceTable,
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

                sample_annotation_record = annotation_table._token_to_record[box.token]
                instance_token = sample_annotation_record._instance_token
                attribute_tokens = sample_annotation_record._attribute_tokens
                category_token = instance_table._token_to_record[instance_token]._category_token
                imsize = [
                    sample_data_table._token_to_record[camera_token].width,
                    sample_data_table._token_to_record[camera_token].height,
                ]

                object_ann_record = ObjectAnnRecord(
                    sample_data_token=camera_token,
                    instance_token=instance_token,
                    category_token=category_token,
                    attribute_tokens=attribute_tokens,
                    bbox=[min_x, min_y, max_x, max_y],
                    mask={"size": imsize},
                )
                object_ann_table._token_to_record[object_ann_record.token] = object_ann_record
