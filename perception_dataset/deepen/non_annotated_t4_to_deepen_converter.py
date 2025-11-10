from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import glob
import os
import os.path as osp
import shutil
import time
from typing import Any, Dict

from nptyping import NDArray
import numpy as np
from pyquaternion import Quaternion
from t4_devkit import Tier4
from t4_devkit.dataclass import LidarPointCloud
from t4_devkit.schema import CalibratedSensor, EgoPose, SampleData

from perception_dataset.abstract_converter import AbstractConverter
from perception_dataset.constants import SENSOR_ENUM
from perception_dataset.deepen.json_format import ConfigData, ImageData
from perception_dataset.utils.logger import configure_logger
from perception_dataset.utils.transform import transform_matrix

logger = configure_logger(modname=__name__)


@dataclass
class NonAnnotatedT4ToDeepenConverterOutputItem:
    output_path: str


@dataclass
class NonAnnotatedT4ToDeepenConverterOutput:
    items: list[NonAnnotatedT4ToDeepenConverterOutputItem]


class NonAnnotatedT4ToDeepenConverter(AbstractConverter[NonAnnotatedT4ToDeepenConverterOutput]):
    def __init__(
        self,
        input_base: str,
        output_base: str,
        camera_sensors: list,
        annotation_hz: int = 10,
        workers_number: int = 32,
        drop_camera_token_not_found: bool = False,
        save_intensity: bool = False,
    ):
        super().__init__(input_base, output_base)

        self._save_intensity = save_intensity

        self._camera_sensor_types = []
        self._annotation_hz = annotation_hz
        self._workers_number = workers_number
        self._drop_camera_token_not_found = drop_camera_token_not_found
        if isinstance(camera_sensors, list):
            for cam in camera_sensors:
                self._camera_sensor_types.append(SENSOR_ENUM[cam["channel"]])

    def convert(self) -> NonAnnotatedT4ToDeepenConverterOutput:
        start_time = time.time()

        output_items: list[NonAnnotatedT4ToDeepenConverterOutputItem] = []
        for scene_dir in glob.glob(osp.join(self._input_base, "*")):
            if not osp.isdir(scene_dir):
                continue

            out_dir = osp.join(self._output_base, osp.basename(scene_dir).replace(".", "-"))
            self._convert_one_scene(
                scene_dir,
                out_dir,
            )
            output_path = shutil.make_archive(f"{out_dir}", "zip", root_dir=out_dir)
            output_items.append(
                NonAnnotatedT4ToDeepenConverterOutputItem(
                    output_path=output_path,
                )
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.1f} [sec]")
        return NonAnnotatedT4ToDeepenConverterOutput(
            items=output_items,
        )

    def _convert_one_scene(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        t4_dataset = Tier4(data_root=input_dir, verbose=False)

        logger.info(f"Converting {input_dir} to {output_dir}")

        with ProcessPoolExecutor(max_workers=self._workers_number) as executor:
            future_list = []
            for frame_index, sample in enumerate(t4_dataset.sample):
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
        t4_dataset = Tier4(data_root=input_dir, verbose=False)
        sample = t4_dataset.sample[frame_index]
        camera_only_mode = SENSOR_ENUM.LIDAR_CONCAT.value["channel"] not in sample.data
        if not camera_only_mode:
            lidar_token: str = sample.data[SENSOR_ENUM.LIDAR_CONCAT.value["channel"]]
            lidar_path: str = t4_dataset.get_sample_data(lidar_token)[0]
            data_dict: Dict[str, Any] = self._get_data(t4_dataset, lidar_token)

            pointcloud: LidarPointCloud = LidarPointCloud.from_file(lidar_path)
            pointcloud.transform(data_dict["sensor2global_transform"])
            points: NDArray = pointcloud.points.T  # (-1, 4)

            config_data = ConfigData(
                frame_index=frame_index,
                unix_timestamp=data_dict["unix_timestamp"],
                points=points,
                device_position=data_dict["sensor2global_translation"],
                device_heading=data_dict["sensor2global_rotation"],
                save_intensity=self._save_intensity,
            )

        for camera_sensor_type in self._camera_sensor_types:
            camera_channel = camera_sensor_type.value["channel"]
            camera_token: str | None = self._get_camera_token(camera_channel, sample, t4_dataset)

            if camera_token is None:
                if self._drop_camera_token_not_found:
                    logger.warning(
                        f"Skipping.. Camera token not found for {camera_channel} in frame {frame_index}. Dropping this frame"
                    )
                    return
                else:
                    logger.error(
                        f"Camera token not found for {camera_channel} in frame {frame_index}. "
                    )
                    continue
            camera_path, _, cam_intrinsic = t4_dataset.get_sample_data(camera_token)
            data_dict: Dict[str, Any] = self._get_data(t4_dataset, camera_token)

            image_data = ImageData(
                frame_index=frame_index,
                channel=camera_channel,
                fileformat=data_dict["fileformat"],
                unix_timestamp=data_dict["unix_timestamp"],
                device_position=data_dict["sensor2global_translation"],
                device_heading=data_dict["sensor2global_rotation"],
                camera_intrinsic_matrix=cam_intrinsic,
            )
            image_data.make_directory(output_dir)
            image_data.save(camera_path, output_dir)
            if camera_only_mode:
                config_data = ConfigData(
                    frame_index=frame_index,
                    unix_timestamp=data_dict["unix_timestamp"],
                    points=None,
                    device_position=data_dict["sensor2global_translation"],
                    device_heading=data_dict["sensor2global_rotation"],
                )
            config_data.add_image_data(image_data)

        config_data.save_json(output_dir)

    @staticmethod
    def _get_camera_token(camera_channel: str, sample, t4_dataset) -> str | None:
        """Get camera token for `camera_channel` in the given `sample` data from a Tier4 dataset.
        Args:
            camera_channel (str): Camera channel name e.g. CAM_FRONT to look for
            sample: Sample data for a specific frame from Tier4 = `t4_dataset.sample[frame_index]`
            t4_dataset (Tier4): Tier4 dataset
        Return:
            camera_token: str | None
                None if not found
        """
        camera_token: str | None = None
        if camera_channel in sample.data.keys():
            camera_token = sample.data[camera_channel]
        else:
            sample_data = [s for s in t4_dataset.sample_data if s.sample_token == sample.token]
            for sensor in sample_data:
                if sensor.channel == camera_channel:
                    camera_token = sensor.token
                    break
        return camera_token

    def _get_data(self, t4_dataset: Tier4, sensor_channel_token: str) -> Dict[str, Any]:
        sd_record: SampleData = t4_dataset.get("sample_data", sensor_channel_token)
        cs_record: CalibratedSensor = t4_dataset.get(
            "calibrated_sensor", sd_record.calibrated_sensor_token
        )
        ep_record: EgoPose = t4_dataset.get("ego_pose", sd_record.ego_pose_token)

        sensor2ego_transform = transform_matrix(
            translation=cs_record.translation,
            rotation=cs_record.rotation,
        )
        ego2global_transform = transform_matrix(
            translation=ep_record.translation,
            rotation=ep_record.rotation,
        )

        sensor2global_transform = ego2global_transform @ sensor2ego_transform
        sensor2global_translation = sensor2global_transform[:3, 3]
        sensor2global_rotation = np.array(list(Quaternion(matrix=sensor2global_transform[:3, :3])))

        ret_dict = {
            "fileformat": sd_record.fileformat,
            "unix_timestamp": self._timestamp_to_sec(sd_record.timestamp),
            "sensor2global_transform": sensor2global_transform,
            "sensor2global_translation": sensor2global_translation,
            "sensor2global_rotation": sensor2global_rotation,
        }

        return ret_dict

    def _timestamp_to_sec(self, timestamp: int) -> float:
        return float(timestamp) * 1e-6
