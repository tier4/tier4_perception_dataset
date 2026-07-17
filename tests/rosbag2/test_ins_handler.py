import pytest

from perception_dataset.ros2.oxts_msgs.ins_handler import INSHandler


class TestINSHandlerGetTopicMapping:
    """`ins_topic_mapping` must be provided explicitly.

    The OXTS topic set was the DRS-specific default and is no longer treated as an
    implicit fallback, so an explicit, vehicle-specific mapping is now required.
    """

    VALID_MAPPING = {
        "imu": "/sensing/imu/tamagawa/imu_raw",
        "nav_sat_fix": "/sensing/gnss/septentrio/nav_sat_fix",
        "odometry": "/localization/kinematic_state",
    }

    def test_returns_given_mapping(self):
        assert INSHandler.get_topic_mapping(self.VALID_MAPPING) == self.VALID_MAPPING

    def test_none_raises(self):
        # No implicit (OXTS) default: a mapping must be provided explicitly.
        with pytest.raises(ValueError):
            INSHandler.get_topic_mapping(None)

    def test_missing_key_raises(self):
        mapping = {"imu": "/a", "odometry": "/b"}  # missing nav_sat_fix
        with pytest.raises(ValueError):
            INSHandler.get_topic_mapping(mapping)

    def test_extra_key_raises(self):
        mapping = dict(self.VALID_MAPPING, extra="/c")
        with pytest.raises(ValueError):
            INSHandler.get_topic_mapping(mapping)

    def test_no_oxts_default_attribute(self):
        # The OXTS-specific default mapping must no longer exist.
        assert not hasattr(INSHandler, "DEFAULT_TOPIC_MAPPING")
