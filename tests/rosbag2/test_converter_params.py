import pytest
from pydantic import ValidationError

from perception_dataset.rosbag2.converter_params import (
    LidarSensor,
    LidarSourceMapping,
    Rosbag2ConverterParams,
)


class TestRosbag2ConverterParams:
    def test_validation_pass(self):
        Rosbag2ConverterParams(
            task="task",
            input_base="input_base",
            output_base="output_base",
            object_msg_type="DetectedObjects",
            num_load_frames=1,
            skip_timestamp=1.0,
        )

    def test_validation_error(self):
        with pytest.raises(ValidationError) as e:
            Rosbag2ConverterParams(
                task="task",
                input_base="input_base",
                output_base="output_base",
                object_msg_type="invalid",
                num_load_frames=1,
                skip_timestamp=1.0,
            )

        assert len(e.value.errors()) == 1
        assert e.value.errors()[0]["loc"] == ("object_msg_type",)


class TestLidarSensor:
    def test_validation_pass_minimum(self):
        LidarSensor(
            topic="task",
            channel="input_base",
        )

    def test_validation_pass_all_defined(self):
        LidarSensor(
            topic="/sensing/lidar/concatenated/pointcloud",
            channel="LIDAR_CONCAT",
            lidar_info_topic="/sensing/lidar/concatenated/pointcloud_info",
            lidar_info_channel="LIDAR_CONCAT_INFO",
            accept_no_info=False,
            lidar_sources_mapping=[
                LidarSourceMapping(
                    topic="/sensing/lidar/rear_upper/pointcloud_before_sync",
                    channel="LIDAR_REAR_UPPER",
                    frame_id="rear_upper/lidar_base_link",
                )
            ],
        )

    def test_validation_error_required_params(self):
        with pytest.raises(ValidationError) as e:
            LidarSensor(
                topic="/sensing/lidar/concatenated/pointcloud",
                channel="LIDAR_CONCAT",
                lidar_info_topic="/sensing/lidar/concatenated/pointcloud_info",
                lidar_info_channel=None,
                accept_no_info=None,
                lidar_sources_mapping=[],
            )

        assert len(e.value.errors()) == 1
        assert e.value.errors()[0]["type"] == "value_error"
        assert (
            e.value.errors()[0]["msg"]
            == "Value error, If any of lidar_info_topic, lidar_info_channel, accept_no_info, or lidar_sources_mapping is defined, all must be defined. Defined: ['lidar_info_topic']. Undefined: ['lidar_info_channel', 'accept_no_info', 'lidar_sources_mapping']"
        )
