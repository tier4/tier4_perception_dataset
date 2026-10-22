"""The sample Kognic conversion config carries everything the current pre-label task definitions need."""

from pathlib import Path

import yaml

from perception_dataset.kognic.openlabel.openlabel_attributes import (
    DEFAULT_OCCLUSION_TO_VISIBILITY,
    DEFAULT_T4_ATTRIBUTE_NAME_TO_KOGNIC,
    attribute_to_text,
    invert_attribute_map,
    occlusion_to_visibility_level,
    to_t4_attribute_name,
)
from perception_dataset.kognic.t4_to_openlabel import T4ToOpenLabelConverter

is_attribute_excluded = T4ToOpenLabelConverter.attribute_in_scoped_groups

SAMPLE_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "convert_annotated_t4_to_kognic_sample.yaml"
)
IMPORT_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "convert_kognic_annotation_to_t4_sample.yaml"
)

TASK_DEFINITION_CLASSES = {
    "traffic_cone",
    "stroller",
    "bicycle_rack",
    "debris",
    "car",
    "animal",
    "trailer",
    "train",
    "construction_sign",
    "personal_mobility",
    "bus",
    "forklift",
    "pushable_pullable",
    "construction_vehicle",
    "kart",
    "barrier",
    "other_vehicle",
    "pedestrian",
    "emergency_vehicle",
    "bicycle",
    "motorcycle",
    "truck",
    "vehicle_protruding_object",
    "open_door",
}


def sample_conversion() -> dict:
    with SAMPLE_CONFIG.open() as f:
        return yaml.safe_load(f)["conversion"]


def test_sample_config_excludes_the_groups_current_task_definitions_reject():
    assert {
        "emergency_vehicle_lights_state",
        "pedestrian.extremities_state",
        "personal_mobility.extremities_state",
        "personal_mobility.pedestrian_state",
        "stroller.pedestrian_state",
        "stroller.extremities_state",
    } <= set(sample_conversion()["exclude_attributes"])


def test_sample_config_maps_categories_onto_task_definition_classes():
    category_map = sample_conversion()["category_map"]
    assert set(category_map.values()) <= TASK_DEFINITION_CLASSES
    assert (
        category_map["police_car"]
        == category_map["ambulance"]
        == category_map["fire_truck"]
        == "emergency_vehicle"
    )
    assert category_map["wheelchair"] == "personal_mobility"


def test_sample_config_resolves_multi_valued_and_static_attributes():
    conversion = sample_conversion()
    assert conversion["attribute_value_priority"]["extremities_state"] == [
        "open_door",
        "protruding_object",
        "none",
    ]
    assert conversion["attribute_value_priority"]["vehicle_state"] == ["parked", "driving"]
    assert conversion["static_attributes"] == ["forklift.extremities_state"]
    assert conversion["include_attributes"] is True


def test_sample_exclusions_drop_personal_mobility_pedestrian_state_only_for_that_class():
    exclude = set(sample_conversion()["exclude_attributes"])
    assert is_attribute_excluded(
        "pedestrian_state.walking", "personal_mobility", "personal_mobility", exclude
    )
    assert not is_attribute_excluded(
        "pedestrian_state.walking", "pedestrian", "pedestrian", exclude
    )
    # the mapped class counts too: wheelchair -> personal_mobility
    assert is_attribute_excluded(
        "pedestrian_state.walking", "wheelchair", "personal_mobility", exclude
    )


def test_converter_accepts_the_sample_config(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    conversion = sample_conversion()
    converter = T4ToOpenLabelConverter(
        str(tmp_path / "in"),
        str(tmp_path / "out"),
        category_map=conversion["category_map"],
        include_attributes=conversion["include_attributes"],
        exclude_attributes=conversion["exclude_attributes"],
        attribute_value_priority=conversion["attribute_value_priority"],
        static_attributes=conversion["static_attributes"],
    )
    assert "personal_mobility.pedestrian_state" in converter._exclude_attributes
    assert converter._static_attributes == {"forklift.extremities_state"}
    assert converter._attribute_value_priority["vehicle_state"] == ["parked", "driving"]


def test_category_attributes_add_a_property_per_t4_category(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    converter = T4ToOpenLabelConverter(
        str(tmp_path / "in"),
        str(tmp_path / "out"),
        category_map={"police_car": "emergency_vehicle"},
        category_attributes={"police_car": {"emergency_vehicle_type": "police_car"}},
    )
    assert converter._category_attributes == {
        "police_car": {"emergency_vehicle_type": "police_car"}
    }
    assert sample_conversion()["category_attributes"]["ambulance"] == {
        "emergency_vehicle_type": "ambulance"
    }


def test_task_definition_class_check_names_the_unmapped_category(tmp_path):
    import pytest

    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    conversion = sample_conversion()
    assert set(conversion["task_definition_classes"]) == TASK_DEFINITION_CLASSES
    converter = T4ToOpenLabelConverter(
        str(tmp_path / "in"),
        str(tmp_path / "out"),
        category_map=conversion["category_map"],
        task_definition_classes=conversion["task_definition_classes"],
    )
    converter._check_task_definition_classes(["car", "police_car", "wheelchair"], "scene-ok")
    with pytest.raises(ValueError, match="golf_cart -> golf_cart"):
        converter._check_task_definition_classes(["car", "golf_cart"], "scene-bad")
    # without the list the check is off
    T4ToOpenLabelConverter(
        str(tmp_path / "in"), str(tmp_path / "out")
    )._check_task_definition_classes(["golf_cart"], "x")


def test_attribute_map_is_configurable_and_inverted_on_import():
    assert invert_attribute_map() == {"rider_state": "two_wheel_vehicle_state"}
    custom = {"two_wheel_vehicle_state": "rider_state", "vehicle_state": "motion_state"}
    assert attribute_to_text("vehicle_state.parked", custom).name == "motion_state"
    assert (
        attribute_to_text("vehicle_state.parked").name == "vehicle_state"
    )  # default map leaves it
    assert to_t4_attribute_name("motion_state", custom) == "vehicle_state"
    assert to_t4_attribute_name("rider_state") == "two_wheel_vehicle_state"
    assert to_t4_attribute_name("unknown_property", custom) == "unknown_property"
    import pytest

    with pytest.raises(ValueError):
        invert_attribute_map({"a": "same", "b": "same"})


def test_occlusion_to_visibility_is_configurable():
    assert occlusion_to_visibility_level("light") == "most"
    assert occlusion_to_visibility_level("light", {"light": "partial"}) == "partial"
    assert occlusion_to_visibility_level("weird") == "unavailable"
    assert occlusion_to_visibility_level(None, {"none": "full"}) == "unavailable"


def test_both_sample_configs_carry_the_same_spelling_maps():
    with IMPORT_CONFIG.open() as f:
        import_conversion = yaml.safe_load(f)["conversion"]
    export_conversion = sample_conversion()
    assert (
        export_conversion["attribute_map"]
        == import_conversion["attribute_map"]
        == DEFAULT_T4_ATTRIBUTE_NAME_TO_KOGNIC
    )
    assert import_conversion["occlusion_to_visibility"] == DEFAULT_OCCLUSION_TO_VISIBILITY
