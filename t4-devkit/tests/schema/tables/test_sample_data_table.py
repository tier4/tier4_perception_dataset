from t4_devkit.schema import FileFormat


def test_fileformat() -> None:
    """Test FileFormat enum."""

    fileformats = ("jpg", "png", "pcd", "bin", "pcd.bin")

    # check all enum members are covered by above names
    members: list[str] = [m.value for m in FileFormat]
    assert set(members) == set(fileformats)

    # check .values() method
    assert set(FileFormat.values()) == set(fileformats)

    # check each member can construct and its method is valid
    for value in fileformats:
        # check is_member() method
        assert FileFormat.is_member(value)

        member = FileFormat(value)

        # check as_ext() returns .value
        assert member.as_ext() == f".{value}"
