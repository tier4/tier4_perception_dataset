"""Attribute / visibility convention mapping shared by the T4 <-> Kognic
OpenLABEL converters.

T4 stores per-box class properties as ``attribute`` rows named
``<group>.<value>`` (e.g. ``vehicle_state.driving``) plus a separate
``visibility`` level, while Kognic exposes them as OpenLABEL ``text`` properties
and an ``occlusion_state``. The name maps and the occlusion<->visibility map
live here so the forward (``T4ToOpenLabelConverter``) and inverse
(``OpenLabelToT4Converter``) directions stay consistent.
"""

import logging
from typing import Dict, List, Optional

from kognic.openlabel.models import models as openlabel

# Defaults for the two spelling maps between T4 and the Kognic task definitions. Both are
# configurable (``attribute_map`` / ``occlusion_to_visibility`` in the conversion configs);
# the values below are what the current pre-label projects use. The visibility mapping is a best
# effort based on the occlusion_state values on both sides.
DEFAULT_OCCLUSION_TO_VISIBILITY = {
    "none": "full",
    "light": "most",
    "most": "partial",
    "full": "none",
}

# Property names that T4 and Kognic spell differently, T4 -> Kognic. The value (e.g.
# ``with_rider``/``without_rider``) is preserved as-is. The Kognic -> T4 direction is derived with
# ``invert_attribute_map``.
DEFAULT_T4_ATTRIBUTE_NAME_TO_KOGNIC = {
    "two_wheel_vehicle_state": "rider_state",
}

# backwards-compatible aliases
OCCLUSION_TO_VISIBILITY = DEFAULT_OCCLUSION_TO_VISIBILITY
T4_ATTRIBUTE_NAME_TO_KOGNIC = DEFAULT_T4_ATTRIBUTE_NAME_TO_KOGNIC


def invert_attribute_map(name_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Kognic -> T4 property names from a T4 -> Kognic map (``None`` = the default map).

    Raises ``ValueError`` when two T4 names map to one Kognic name, since the inverse would be
    ambiguous when importing annotations.
    """
    forward = DEFAULT_T4_ATTRIBUTE_NAME_TO_KOGNIC if name_map is None else name_map
    inverse: Dict[str, str] = {}
    for t4_name, kognic_name in forward.items():
        if kognic_name in inverse and inverse[kognic_name] != t4_name:
            raise ValueError(
                f"attribute_map is not invertible: {inverse[kognic_name]!r} and {t4_name!r} both map to {kognic_name!r}"
            )
        inverse[kognic_name] = t4_name
    return inverse


def normalize_attribute_value(value: str) -> str:
    """Spell a T4 attribute value the way the Kognic task definition does.

    Some source annotations carry ``"protruding object"`` or trailing blanks where the definition says
    ``protruding_object``; Kognic rejects the unknown spelling ("non-allowed value").

    Args:
        value (str): Raw value part of a T4 attribute name.

    Returns:
        str: Trimmed value with inner whitespace runs replaced by ``_``.
    """
    return "_".join(value.split())


def attribute_to_text(
    attribute_name: str, name_map: Optional[Dict[str, str]] = None
) -> openlabel.Text:
    """Convert a T4 attribute name to an OpenLABEL text property.

    ``name_map`` renames attribute groups T4 and Kognic spell differently (T4 -> Kognic);
    ``None`` uses :data:`DEFAULT_T4_ATTRIBUTE_NAME_TO_KOGNIC`.

    Args:
        attribute_name (str): T4 attribute in ``group.value`` form.

    Returns:
        openlabel.Text: Text property with the Kognic-compatible group name.
    """
    name, _, value = attribute_name.rpartition(".")
    if not name:
        return openlabel.Text(name=attribute_name, val="true")
    forward = DEFAULT_T4_ATTRIBUTE_NAME_TO_KOGNIC if name_map is None else name_map
    name = forward.get(name.strip(), name.strip())
    return openlabel.Text(name=name, val=normalize_attribute_value(value))


logger = logging.getLogger(__name__)


def resolve_multi_value_attribute(
    attribute_name: str, value_priority: Dict[str, List[str]]
) -> str:
    """Reduce a multi-valued T4 attribute (``group.a;b``) to a single ``group.value``.

    Args:
        attribute_name (str): T4 attribute in ``group.value`` form; ``value`` may hold several
            ``;``-separated values.
        value_priority (Dict[str, List[str]]): Per group, the values in decreasing priority
            (``{"extremities_state": ["open_door", "protruding_object", "none"]}``).

    Returns:
        str: ``attribute_name`` unchanged when it is single-valued, otherwise ``group.<chosen>``
            where ``chosen`` is the highest-priority value present; without a priority for the
            group the first listed value is kept (and a warning logged).
    """
    # Some T4 sources encode a multi-select attribute as one ``group.a;b`` row (seen: ``extremities_state.protruding_object;none``).
    # Kognic properties are single-valued, so the pre-annotation keeps one of them, chosen by per-group priority.
    multi_value_seperator = ";"

    group, sep, value = attribute_name.rpartition(".")
    if not sep or multi_value_seperator not in value:
        return attribute_name
    values = [
        normalize_attribute_value(v) for v in value.split(multi_value_seperator) if v.strip()
    ]
    if not values:
        return attribute_name
    chosen = next((v for v in value_priority.get(group, []) if v in values), None)
    if chosen is None:
        chosen = values[0]
        logger.warning(
            f"Multi-valued attribute {attribute_name!r} has no value priority for group {group!r}; keeping {chosen!r}"
        )
    return f"{group}.{chosen}"


def choose_static_value(values: List[str], priority: List[str]) -> str:
    """Pick the single value a per-object (static) Kognic property gets from per-frame T4 values.

    Args:
        values (List[str]): The attribute values seen on one object across frames (non-empty).
        priority (List[str]): Values in decreasing priority; the first one present wins.

    Returns:
        str: The highest-priority value present, else the most frequent one (first seen breaks ties).
    """
    present = set(values)
    for candidate in priority:
        if candidate in present:
            return candidate
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda value: (counts[value], -values.index(value)))


def to_t4_attribute_name(kognic_name: str, name_map: Optional[Dict[str, str]] = None) -> str:
    """Map a Kognic property name to its T4 attribute-group spelling.

    Args:
        kognic_name (str): Kognic property name.
        name_map (Optional[Dict[str, str]]): T4 -> Kognic map (the same one the forward
            conversion uses); it is inverted here. ``None`` uses the default map.

    Returns:
        str: Corresponding T4 attribute-group name.
    """
    return invert_attribute_map(name_map).get(kognic_name, kognic_name)


def occlusion_to_visibility_level(
    occlusion: Optional[str], occlusion_to_visibility: Optional[Dict[str, str]] = None
) -> str:
    """Map a Kognic occlusion state to a T4 visibility level.

    Args:
        occlusion (Optional[str]): Kognic ``occlusion_state`` value.
        occlusion_to_visibility (Optional[Dict[str, str]]): Occlusion -> visibility map;
            ``None`` uses :data:`DEFAULT_OCCLUSION_TO_VISIBILITY`.

    Returns:
        str: Corresponding T4 visibility level, or ``unavailable``.
    """
    if not occlusion:
        return "unavailable"
    mapping = (
        DEFAULT_OCCLUSION_TO_VISIBILITY
        if occlusion_to_visibility is None
        else occlusion_to_visibility
    )
    return mapping.get(occlusion, "unavailable")
