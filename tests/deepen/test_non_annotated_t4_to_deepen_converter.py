import pytest

from perception_dataset.deepen.non_annotated_t4_to_deepen_converter import (
    NonAnnotatedT4ToDeepenConverter,
)


class TestNonAnnotatedT4ToDeepenConverter:
    @pytest.fixture(scope="function")
    def converter_for_test(self):
        # TODO(yukke42): test with files
        return NonAnnotatedT4ToDeepenConverter(input_base="", output_base="", camera_sensors=[])

    def test__convert_one_scene(self):
        # TODO(yukke42): impl test__convert_one_scene
        pass

    def test__get_data(self):
        # TODO(yukke42): impl test__get_data
        pass

    @pytest.mark.parametrize("timestamp, expected_value", [(1624164470899887, 1624164470.899887)])
    def test__timestamp_to_sec(
        self,
        mocker,
        scope_function,
        timestamp: int,
        expected_value: float,
        converter_for_test: NonAnnotatedT4ToDeepenConverter,
    ):
        timestamp_f = converter_for_test._timestamp_to_sec(timestamp)
        assert isinstance(timestamp_f, float)
        assert timestamp_f == pytest.approx(expected_value)
