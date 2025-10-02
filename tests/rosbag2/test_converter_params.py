from pydantic import ValidationError
import pytest

from perception_dataset.rosbag2.converter_params import Rosbag2ConverterParams


def test_validation_pass():
    Rosbag2ConverterParams(
        task="task",
        input_base="input_base",
        output_base="output_base",
        object_msg_type="DetectedObjects",
        num_load_frames=1,
        skip_timestamp=1.0,
    )


def test_validation_error():
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


def test_ins_topic_mapping():
    """Test that ins_topic_mapping can be specified and defaults to None."""
    params = Rosbag2ConverterParams(
        task="task",
        input_base="input_base",
        output_base="output_base",
        object_msg_type="DetectedObjects",
        num_load_frames=1,
        skip_timestamp=1.0,
    )
    assert params.ins_topic_mapping is None

    custom_mapping = {
        "imu": "/custom/imu",
        "nav_sat_fix": "/custom/nav_sat_fix",
        "odometry": "/custom/odometry",
        "velocity": "/custom/velocity",
    }
    params_with_mapping = Rosbag2ConverterParams(
        task="task",
        input_base="input_base",
        output_base="output_base",
        object_msg_type="DetectedObjects",
        num_load_frames=1,
        skip_timestamp=1.0,
        ins_topic_mapping=custom_mapping,
    )
    assert params_with_mapping.ins_topic_mapping == custom_mapping


def test_vehicle_status_topic_mapping():
    """Test that vehicle_status_topic_mapping can be specified and defaults to None."""
    params = Rosbag2ConverterParams(
        task="task",
        input_base="input_base",
        output_base="output_base",
        object_msg_type="DetectedObjects",
        num_load_frames=1,
        skip_timestamp=1.0,
    )
    assert params.vehicle_status_topic_mapping is None

    custom_mapping = {
        "actuation_status": "/custom/actuation_status",
        "control_mode": "/custom/control_mode",
        "door_status": "/custom/door_status",
        "gear_status": "/custom/gear_status",
        "hazard_lights_status": "/custom/hazard_lights_status",
        "steering_status": "/custom/steering_status",
        "steering_wheel_status": "/custom/steering_wheel_status",
        "turn_indicators_status": "/custom/turn_indicators_status",
        "velocity_status": "/custom/velocity_status",
    }
    params_with_mapping = Rosbag2ConverterParams(
        task="task",
        input_base="input_base",
        output_base="output_base",
        object_msg_type="DetectedObjects",
        num_load_frames=1,
        skip_timestamp=1.0,
        vehicle_status_topic_mapping=custom_mapping,
    )
    assert params_with_mapping.vehicle_status_topic_mapping == custom_mapping
