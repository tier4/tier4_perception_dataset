import logging

import numpy as np
from pydantic import ValidationError
import pytest

from perception_dataset.rosbag2.converter_params import (
    LidarSensor,
    LidarSourceMapping,
    Rosbag2ConverterParams,
)

CONVERTER_PARAMS_LOGGER = "perception_dataset.rosbag2.converter_params"


@pytest.fixture
def isolated_converter_params_logger(monkeypatch):
    converter_params_logger = logging.getLogger(CONVERTER_PARAMS_LOGGER)
    monkeypatch.setattr(converter_params_logger, "handlers", [])
    monkeypatch.setattr(converter_params_logger, "propagate", True)
    return converter_params_logger


def make_params(**kwargs):
    return Rosbag2ConverterParams(
        task="task",
        input_base="input_base",
        output_base="output_base",
        object_msg_type="DetectedObjects",
        num_load_frames=1,
        skip_timestamp=1.0,
        camera_sensors=[{"topic": "/camera"}],
        **kwargs,
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

    def test_jpeg_defaults(self):
        params = make_params()

        assert params.jpeg_quality == 95
        assert params.jpeg_optimize is False

    def test_jpeg_valid_opt_in_values(self):
        assert make_params(jpeg_quality=85).jpeg_quality == 85
        assert make_params(jpeg_optimize=True).jpeg_optimize is True

    @pytest.mark.parametrize("jpeg_quality", [True, False, np.bool_(False), np.bool_(True)])
    def test_jpeg_quality_rejects_booleans(self, jpeg_quality):
        with pytest.raises(ValidationError):
            make_params(jpeg_quality=jpeg_quality)

    @pytest.mark.parametrize(("jpeg_quality", "expected_quality"), [(1, 1), (100, 100)])
    def test_jpeg_quality_boundaries_do_not_warn(
        self,
        caplog,
        isolated_converter_params_logger,
        jpeg_quality,
        expected_quality,
    ):
        with caplog.at_level(logging.WARNING, logger=CONVERTER_PARAMS_LOGGER):
            params = make_params(jpeg_quality=jpeg_quality)

        assert params.jpeg_quality == expected_quality
        assert caplog.records == []

    @pytest.mark.parametrize(
        ("jpeg_quality", "expected_quality", "expected_warning"),
        [
            (0, 1, "jpeg_quality must be in [1, 100], got 0, replaced to 1."),
            (150, 100, "jpeg_quality must be in [1, 100], got 150, replaced to 100."),
        ],
    )
    def test_jpeg_quality_clamps_out_of_range_values_with_a_warning(
        self,
        caplog,
        isolated_converter_params_logger,
        jpeg_quality,
        expected_quality,
        expected_warning,
    ):
        with caplog.at_level(logging.WARNING, logger=CONVERTER_PARAMS_LOGGER):
            params = make_params(jpeg_quality=jpeg_quality)

        assert params.jpeg_quality == expected_quality
        assert caplog.messages == [expected_warning]
        assert [record.name for record in caplog.records] == [CONVERTER_PARAMS_LOGGER]


class TestLidarSensor:
    def test_validation_pass_minimum(self):
        LidarSensor(
            topic="task",
            channel="input_base",
            num_lidar_feats=7,
            output_pointcloud_format="pcd",
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

    def test_validation_error_num_lidar_feats(self):
        with pytest.raises(ValidationError) as e:
            LidarSensor(
                topic="task",
                channel="input_base",
                num_lidar_feats=6,
            )

        assert len(e.value.errors()) == 1
        assert e.value.errors()[0]["type"] == "value_error"

    def test_validation_error_output_pointcloud_format(self):
        with pytest.raises(ValidationError) as e:
            LidarSensor(
                topic="task",
                channel="input_base",
                output_pointcloud_format="ply",
            )

        assert len(e.value.errors()) == 1
        assert e.value.errors()[0]["type"] == "value_error"
