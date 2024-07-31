import pytest
from pydantic import ValidationError

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
