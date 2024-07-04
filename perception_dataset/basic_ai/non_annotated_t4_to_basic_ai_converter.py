from concurrent.futures import ProcessPoolExecutor
import glob
import open3d as o3d
import os
import os.path as osp
import shutil
import time
from typing import Any, Dict
from pathlib import Path

from nptyping import NDArray
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import SENSOR_ENUM
# from perception_dataset.deepen.json_format import ConfigData, ImageData
from perception_dataset.utils.logger import configure_logger
from perception_dataset.basic_ai.utils import BasicAiCameraConfig, save_pcd

logger = configure_logger(modname=__name__)


class NonAnnotatedT4ToBasicAiConverter(AbstractConverter):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        camera_sensors: list,
        annotation_hz: int = 10,
        workers_number: int = 32,
    ):
        super().__init__(input_base, output_base)

        self._camera_sensor_types = []
        self._annotation_hz = annotation_hz
        self._workers_number = workers_number
        if isinstance(camera_sensors, list):
            for cam in camera_sensors:
                self._camera_sensor_types.append(SENSOR_ENUM[cam["channel"]])

    def convert(self):
        start_time = time.time()

        for scene_dir in glob.glob(osp.join(self._input_base, "*")):
            if not osp.isdir(scene_dir):
                continue

            out_dir = osp.join(self._output_base, osp.basename(scene_dir).replace(".", "-"))
            self._convert_one_scene(
                scene_dir,
                out_dir,
            )
            shutil.make_archive(f"{out_dir}", "zip", root_dir=out_dir)

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.1f} [sec]")

    def _convert_one_scene(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        nusc = NuScenes(version="annotation", dataroot=input_dir, verbose=False)

        logger.info(f"Converting {input_dir} to {output_dir}")

        with ProcessPoolExecutor(max_workers=self._workers_number) as executor:
            future_list = []
            for frame_index, sample in enumerate(nusc.sample):
                if frame_index % int(10 / self._annotation_hz) != 0:
                    continue
                future = executor.submit(
                    self._convert_one_frame, input_dir, output_dir, frame_index
                )
                future_list.append(future)
            [x.result() for x in future_list]
        logger.info(f"Done Conversion: {input_dir} to {output_dir}")

    def _convert_one_frame(self, input_dir: str, output_dir: str, frame_index: int):
        if frame_index % 10 == 0:
            logger.info(f"frame index: {frame_index}")
        nusc = NuScenes(version="annotation", dataroot=input_dir, verbose=False)
        sample = nusc.sample[frame_index]

        # save lidar
        lidar_token: str = sample["data"][SENSOR_ENUM.LIDAR_CONCAT.value["channel"]]
        lidar_path: str = nusc.get_sample_data(lidar_token)[0]
        lidar_data_dict: Dict[str, Any] = self._get_data(nusc, lidar_token)

        pointcloud: LidarPointCloud = LidarPointCloud.from_file(lidar_path)
        pointcloud.transform(lidar_data_dict["sensor2global_transform"])
        points: NDArray = pointcloud.points.T  # (-1, 4)
        points = self._transform_points_from_global_to_lidar(points, lidar_data_dict["global2sensor_transform"])

        output_pcd_file_path = Path(output_dir) / "lidar_point_cloud_0" / f"data{frame_index + 1}.pcd"
        output_pcd_file_path.parent.mkdir(parents=True, exist_ok=True)
        save_pcd(points, output_pcd_file_path)

        # save camera images and camera config
        camera_config = BasicAiCameraConfig()
        for index, camera_sensor_type in enumerate(self._camera_sensor_types):
            camera_channel = camera_sensor_type.value["channel"]

            if camera_channel in sample["data"].keys():
                camera_token = sample["data"][camera_channel]
            else:
                sample_data = [s for s in nusc.sample_data if s["sample_token"] == sample["token"]]
                for sensor in sample_data:
                    if sensor["channel"] == camera_channel:
                        camera_token = sensor["token"]
                        break
                logger.warning(
                    f"camera: {camera_channel} not found in frame {frame_index}, skipping this frame..."
                )
                return

            input_image_file_path, _, cam_intrinsic = nusc.get_sample_data(camera_token)
            camera_data_dict: Dict[str, Any] = self._get_data(nusc, camera_token)

            output_image_file_path = Path(output_dir) / f"camera_image_{index}" / f"data{frame_index + 1}.jpg"
            output_image_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(input_image_file_path, output_image_file_path)

            camera_external = (camera_data_dict["ego2sensor_transform"] @ lidar_data_dict["sensor2ego_transform"])
            camera_config.set_camera_config(
                camera_index=index,
                camera_internal={
                    "fx": cam_intrinsic[0, 0],
                    "cx": cam_intrinsic[0, 2],
                    "fy": cam_intrinsic[1, 1],
                    "cy": cam_intrinsic[1, 2],
                },
                camera_external=camera_external.reshape(-1).tolist(),
                row_major=True,
                distortionK=[],
                distortionP=[],
                distortionInvP=[],
                width=camera_data_dict["width"],
                height=camera_data_dict["height"],
            )
        
        output_camera_config_path = Path(output_dir) / "camera_config" / f"data{frame_index + 1}.json"
        output_camera_config_path.parent.mkdir(parents=True, exist_ok=True)
        camera_config.dump_json(output_camera_config_path)


    def _get_data(self, nusc: NuScenes, sensor_channel_token: str) -> Dict[str, Any]:
        sd_record = nusc.get("sample_data", sensor_channel_token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        ep_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        sensor2ego_transform = transform_matrix(
            translation=cs_record["translation"],
            rotation=Quaternion(cs_record["rotation"]),
        )
        ego2sensor_transform = transform_matrix(
            translation=cs_record["translation"],
            rotation=Quaternion(cs_record["rotation"]),
            inverse=True,
        )
        ego2global_transform = transform_matrix(
            translation=ep_record["translation"],
            rotation=Quaternion(ep_record["rotation"]),
        )
        global2ego_transform = transform_matrix(
            translation=ep_record["translation"],
            rotation=Quaternion(ep_record["rotation"]),
            inverse=True
        )
        sensor2global_transform = ego2global_transform @ sensor2ego_transform
        sensor2global_translation = sensor2global_transform[:3, 3]
        sensor2global_rotation = np.array(list(Quaternion(matrix=sensor2global_transform[:3, :3])))

        global2sensor_transform = ego2sensor_transform @ global2ego_transform

        ret_dict = {
            "fileformat": sd_record["fileformat"],
            "unix_timestamp": self._timestamp_to_sec(sd_record["timestamp"]),
            "sensor2global_transform": sensor2global_transform,
            "sensor2global_translation": sensor2global_translation,
            "sensor2global_rotation": sensor2global_rotation,
            "global2sensor_transform": global2sensor_transform,
            "sensor2ego_transform": sensor2ego_transform,
            "ego2sensor_transform": ego2sensor_transform,
            "width": sd_record["width"],
            "height": sd_record["height"],
        }

        return ret_dict

    def _transform_points_from_global_to_lidar(self, points, global_to_lidar):
        """Transform points
        """
        points_out = np.ones_like(points)
        points_out[:, :3] = points[:, :3]

        points_out = (global_to_lidar @ points_out.T).T
        points_out[:, 3] = points[:, 3]
        return points_out

    def _timestamp_to_sec(self, timestamp: int) -> float:
        return float(timestamp) * 1e-6
