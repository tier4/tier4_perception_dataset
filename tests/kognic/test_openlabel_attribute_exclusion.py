"""Unit tests for class-scoped attribute exclusion in the OpenLABEL pre-annotation converter."""

from perception_dataset.kognic.openlabel.openlabel_attributes import (
    attribute_to_text,
    choose_static_value,
    normalize_attribute_value,
    resolve_multi_value_attribute,
)
from perception_dataset.kognic.t4_to_openlabel import T4ToOpenLabelConverter

is_attribute_excluded = T4ToOpenLabelConverter.attribute_in_scoped_groups

PRIORITY = {
    "extremities_state": ["open_door", "protruding_object", "none"],
    "vehicle_state": ["parked", "driving"],
}


def test_global_group_exclusion_applies_to_every_class():
    excluded = {"emergency_vehicle_lights_state"}
    assert is_attribute_excluded("emergency_vehicle_lights_state.on", "car", "car", excluded)
    assert is_attribute_excluded("emergency_vehicle_lights_state.on", "bus", "bus", excluded)
    assert not is_attribute_excluded("occlusion_state.none", "car", "car", excluded)


def test_class_scoped_exclusion_matches_t4_category_or_mapped_class():
    excluded = {"pedestrian.extremities_state", "personal_mobility.extremities_state"}
    assert is_attribute_excluded("extremities_state.none", "pedestrian", "pedestrian", excluded)
    # wheelchair mapped to personal_mobility via category_map
    assert is_attribute_excluded(
        "extremities_state.none", "wheelchair", "personal_mobility", excluded
    )
    # vehicles keep their extremities state
    assert not is_attribute_excluded("extremities_state.open_door", "car", "car", excluded)
    assert not is_attribute_excluded("extremities_state.protruding_object", "bus", "bus", excluded)


def test_attribute_without_group_is_never_excluded():
    assert not is_attribute_excluded(
        "standalone", "pedestrian", "pedestrian", {"pedestrian.standalone", "standalone"}
    )


def test_multi_value_attribute_keeps_highest_priority_value():
    assert (
        resolve_multi_value_attribute("extremities_state.protruding_object;none", PRIORITY)
        == "extremities_state.protruding_object"
    )
    assert (
        resolve_multi_value_attribute("extremities_state.protruding_object;open_door", PRIORITY)
        == "extremities_state.open_door"
    )
    assert (
        resolve_multi_value_attribute(
            "extremities_state.none;open_door;protruding_object", PRIORITY
        )
        == "extremities_state.open_door"
    )
    assert (
        resolve_multi_value_attribute("vehicle_state.driving;parked", PRIORITY)
        == "vehicle_state.parked"
    )


def test_single_value_and_groupless_attributes_are_unchanged():
    assert (
        resolve_multi_value_attribute("extremities_state.open_door", PRIORITY)
        == "extremities_state.open_door"
    )
    assert resolve_multi_value_attribute("standalone", PRIORITY) == "standalone"
    assert resolve_multi_value_attribute("standalone;other", PRIORITY) == "standalone;other"


def test_multi_value_without_priority_keeps_first_listed_value():
    assert (
        resolve_multi_value_attribute("pedestrian_state.sitting;standing", PRIORITY)
        == "pedestrian_state.sitting"
    )
    assert (
        resolve_multi_value_attribute("extremities_state.foo;bar", PRIORITY)
        == "extremities_state.foo"
    )


def test_resolved_attribute_becomes_single_valued_text_property():
    text = attribute_to_text(
        resolve_multi_value_attribute("extremities_state.protruding_object;none", PRIORITY)
    )
    assert (text.name, text.val) == ("extremities_state", "protruding_object")


def test_static_value_prefers_priority_then_frequency():
    assert (
        choose_static_value(["none", "none", "open_door"], PRIORITY["extremities_state"])
        == "open_door"
    )
    assert (
        choose_static_value(["none", "protruding_object", "none"], PRIORITY["extremities_state"])
        == "protruding_object"
    )
    assert choose_static_value(["standing", "sitting", "sitting"], []) == "sitting"
    assert choose_static_value(["a", "b"], []) == "a"  # tie -> first seen


def test_attribute_values_are_normalized_to_task_definition_spelling():
    assert normalize_attribute_value("protruding object") == "protruding_object"
    assert normalize_attribute_value("  open_door ") == "open_door"
    text = attribute_to_text("extremities_state.protruding object")
    assert (text.name, text.val) == ("extremities_state", "protruding_object")
    assert (
        resolve_multi_value_attribute("extremities_state.protruding object;none", PRIORITY)
        == "extremities_state.protruding_object"
    )
