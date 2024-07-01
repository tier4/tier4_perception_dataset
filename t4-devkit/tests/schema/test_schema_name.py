from t4_devkit.schema import SchemaName


def test_schema_name() -> None:
    """Test schema name definition is valid."""

    # {name: is_optional}
    schema_names = {
        "attribute": False,
        "calibrated_sensor": False,
        "category": False,
        "ego_pose": False,
        "instance": False,
        "log": False,
        "map": False,
        "sample": False,
        "sample_annotation": False,
        "sample_data": False,
        "visibility": False,
        "sensor": False,
        "scene": False,
        "object_ann": True,
        "surface_ann": True,
        "keypoint": True,
    }

    # check all enum members are covered by above names
    members: list[str] = [m.value for m in SchemaName]
    assert set(members) == set(schema_names.keys())

    # check each member can construct and its method is valid
    for name, is_optional in schema_names.items():
        member = SchemaName(name)

        # check filename returns name.json
        assert member.filename == f"{name}.json"

        # check is_optional()
        assert member.is_optional() == is_optional
