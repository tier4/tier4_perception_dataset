from t4_devkit.schema import VisibilityLevel


def test_visibility_level() -> None:
    """Test VisibilityLevel."""

    levels = ("full", "most", "partial", "none", "unavailable")

    # check all enum members are covered by above names
    members: list[str] = [m.value for m in VisibilityLevel]
    assert set(members) == set(levels)

    # check each member can construct
    for value in levels:
        _ = VisibilityLevel(value)


def test_visibility_level_deprecated() -> None:
    """Test VisibilityLevel with deprecated format."""

    levels = {"v80-100": "full", "v60-80": "most", "v40-60": "partial", "v0-40": "none"}

    for value, expect in levels.items():
        level = VisibilityLevel.from_value(value)
        expect_level = VisibilityLevel(expect)
        assert level == expect_level
