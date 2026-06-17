"""Attribute / visibility convention mapping shared by the T4 <-> Kognic
OpenLABEL converters.

T4 stores per-box class properties as ``attribute`` rows named
``<group>.<value>`` (e.g. ``vehicle_state.driving``) plus a separate
``visibility`` level, while Kognic exposes them as OpenLABEL ``text`` properties
and an ``occlusion_state``. The name maps and the occlusion<->visibility map
live here so the forward (``T4ToOpenLabelConverter``) and inverse
(``OpenLabelToT4Converter``) directions stay consistent.
"""

from typing import Optional

from kognic.openlabel.models import models as openlabel

# TODO: the visibility mapping is a best effort based on the occlusion_state
# values in T4 dataset and kognic format. It should be revisited and
# standardized in the future.
OCCLUSION_TO_VISIBILITY = {
    "none": "full",
    "light": "most",
    "most": "partial",
    "full": "none",
}

# TODO: property names that T4 and Kognic spell differently. The value
# (e.g. ``with_rider``/``without_rider``) is preserved as-is. The two maps are
# inverses of each other and should be revisited and standardized in the future.
T4_ATTRIBUTE_NAME_TO_KOGNIC = {
    "two_wheel_vehicle_state": "rider_state",
}
KOGNIC_ATTRIBUTE_NAME_TO_T4 = {
    kognic_name: t4_name for t4_name, kognic_name in T4_ATTRIBUTE_NAME_TO_KOGNIC.items()
}


def attribute_to_text(attribute_name: str) -> openlabel.Text:
    """Split a T4 attribute like ``vehicle_state.driving`` into an OpenLABEL
    ``text`` property, remapping the group name to its Kognic spelling."""
    name, _, value = attribute_name.rpartition(".")
    if not name:
        return openlabel.Text(name=attribute_name, val="true")
    name = T4_ATTRIBUTE_NAME_TO_KOGNIC.get(name, name)
    return openlabel.Text(name=name, val=value)


def to_t4_attribute_name(kognic_name: str) -> str:
    """Map a Kognic property name back to its T4 attribute-group spelling."""
    return KOGNIC_ATTRIBUTE_NAME_TO_T4.get(kognic_name, kognic_name)


def occlusion_to_visibility_level(occlusion: Optional[str]) -> str:
    """Map a Kognic ``occlusion_state`` value to a T4 visibility level."""
    if not occlusion:
        return "unavailable"
    return OCCLUSION_TO_VISIBILITY.get(occlusion, "unavailable")
