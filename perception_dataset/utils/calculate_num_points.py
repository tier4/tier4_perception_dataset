import numpy as np
from nuscenes import NuScenes

from perception_dataset.t4_dataset.classes.sample_annotation import (
    SampleAnnotationRecord,
    SampleAnnotationTable,
)
from perception_dataset.utils import box_np_ops
from perception_dataset.utils.logger import configure_logger

logger = configure_logger(modname=__name__)


def calculate_num_points(
    dataroot: str, lidar_sensor_channel: str, annotation_table: SampleAnnotationTable
):
    """Calcluate number of points in each box and overwrite the annotation table"""
    nusc = NuScenes(version="annotation", dataroot=dataroot, verbose=False)
    for sample in nusc.sample:
        if not lidar_sensor_channel in sample["data"]:
            continue
        lidar_token = sample["data"][lidar_sensor_channel]
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, 5)

        # taken from awml_det3d/dataset_converter/t4dataset_converter.py
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

        indices = box_np_ops.points_in_rbbox(
            points[:, :3],
            gt_boxes[:, :7],
        )
        num_points = indices.sum(0)

        for box, num in zip(boxes, num_points):
            # Create new record with num_lidar_pts and overwrite the original one
            record: SampleAnnotationRecord = annotation_table._token_to_record[box.token]
            new_record = SampleAnnotationRecord(
                sample_token=record._sample_token,
                instance_token=record._instance_token,
                attribute_tokens=record._attribute_tokens,
                visibility_token=record._visibility_token,
                translation=record._translation,
                size=record._size,
                rotation=record._rotation,
                num_lidar_pts=int(num),
                num_radar_pts=record._num_radar_pts,
            )
            new_record._token = box.token  # overwrite record token with old one
            annotation_table._token_to_record[box.token] = new_record

    # connect next/prev tokens
    for instance in nusc.instance:
        try:
            prev_sample_data: str = annotation_table._token_to_record[
                instance["first_annotation_token"]
            ]
            annotation_data_list = [
                v
                for v in annotation_table._token_to_record.values()
                if v._instance_token == instance["token"]
            ]
            annotation_data_list[0].prev = ""
            for sample_data_i in range(1, len(annotation_data_list)):
                cur_sample_data: str = annotation_data_list[sample_data_i]
                if prev_sample_data._instance_token != cur_sample_data._instance_token:
                    prev_sample_data.next_token = ""
                    cur_sample_data.prev_token = ""
                else:
                    prev_sample_data.next_token = cur_sample_data.token
                    cur_sample_data.prev_token = prev_sample_data.token
                prev_sample_data: str = cur_sample_data
        except KeyError as e:
            logger.error(f"no key {e} in annotation table")
