"""Frozen task vocabulary, record formats and query templates."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


VALUE_TYPES = ("num", "word")
TRAIN_TEMPLATE_IDS = ("Q1", "Q3")
HELDOUT_TEMPLATE_IDS = ("Q2", "Q4")
ALL_TEMPLATE_IDS = TRAIN_TEMPLATE_IDS + HELDOUT_TEMPLATE_IDS

KEY_POOL = (
    "garden", "cotton", "harbor", "violet", "marble", "copper",
    "willow", "cedar", "meadow", "lantern", "orchard", "velvet",
)

WORD_MODIFIERS = (
    "ancient", "faded", "rustic", "polished",
    "antique", "weathered", "ornate", "vintage",
)
WORD_COLORS = (
    "amber", "silver", "blue", "golden",
    "crimson", "ivory", "teal", "gray",
)
WORD_OBJECTS = (
    "bookcase", "sundial", "teapot", "lighthouse",
    "typewriter", "hourglass", "windmill", "armchair",
)

HAYSTACK_TEXT = (
    "This passage contains ordinary background prose about unrelated everyday events."
)
HAYSTACK_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class ValueLanguage:
    singular: str
    plural: str
    record_kind: str


VALUE_LANGUAGE = {
    "num": ValueLanguage("access code", "access codes", "access-code"),
    "word": ValueLanguage(
        "verification phrase", "verification phrases", "verification-phrase"
    ),
}

QUERY_TEMPLATES = {
    "Q1": (
        "The document contains record-key-to-{record_kind} entries. Retrieve the "
        "{items} associated with these record keys: {keys}. Use the keys in the "
        "listed order. {answer_format}"
    ),
    "Q2": (
        "The document contains many {record_kind} records. Look up the {items} for "
        "these requested record keys: {keys}. Use the keys in the listed order. "
        "{answer_format}"
    ),
    "Q3": (
        "According to the {record_kind} records in the document, what {items} "
        "correspond to these record keys: {keys}? Use the keys in the listed order. "
        "{answer_format}"
    ),
    "Q4": (
        "In the document, find the {record_kind} records for these record keys and "
        "retrieve their stored {items}: {keys}. Use the keys in the listed order. "
        "{answer_format}"
    ),
}

INDEPENDENT_QUERY_TEMPLATES = {
    "Q1": (
        "The document contains record-key-to-{record_kind} entries. Retrieve the "
        "{item} associated with record key \"{key}\". {value_shape}Output only "
        "the {item}."
    ),
    "Q2": (
        "The document contains many {record_kind} records. Look up the {item} for "
        "record key \"{key}\". {value_shape}Output only the {item}."
    ),
    "Q3": (
        "According to the {record_kind} records in the document, what {item} "
        "corresponds to record key \"{key}\"? {value_shape}Output only the "
        "{item}."
    ),
    "Q4": (
        "In the document, find the {record_kind} record for key \"{key}\" and "
        "retrieve its stored {item}. {value_shape}Output only the {item}."
    ),
}


def render_query(template_id: str, value_type: str, keys: Sequence[str]) -> str:
    if template_id not in QUERY_TEMPLATES:
        raise ValueError(f"unknown query template: {template_id}")
    if value_type not in VALUE_LANGUAGE:
        raise ValueError(f"unknown value type: {value_type}")
    if not keys:
        raise ValueError("query must contain at least one key")
    language = VALUE_LANGUAGE[value_type]
    count = len(keys)
    item_name = language.singular if count == 1 else language.plural
    if value_type == "num":
        value_shape = (
            "Each access code consists of exactly three three-digit groups "
            "separated by single spaces. "
        )
    else:
        value_shape = (
            "Each verification phrase consists of exactly three words separated "
            "by single spaces. "
        )
    answer_format = (
        f"{value_shape}Return exactly {count} {item_name} in the requested-key "
        f"order. Output only the {item_name}"
    )
    if count > 1:
        answer_format += (
            ", separating different items with a semicolon followed by one space."
        )
    else:
        answer_format += "."
    return QUERY_TEMPLATES[template_id].format(
        record_kind=language.record_kind,
        items=language.plural,
        answer_format=answer_format,
        keys="; ".join(keys),
    )


def render_independent_query(template_id: str, value_type: str, key: str) -> str:
    """Render an order-free query that asks for exactly one stored value."""
    if template_id == "X1":
        return (
            f'Find the identifier paired with record identifier "{key}" in the '
            "mapping. Output only the paired identifier."
        )
    if template_id == "X2":
        return (
            f'Find the stored token that begins with "{key[:5]}" and ends with '
            f'"{key[-5:]}". Output only the complete token.'
        )
    if template_id not in INDEPENDENT_QUERY_TEMPLATES:
        raise ValueError(f"unknown query template: {template_id}")
    if value_type not in VALUE_LANGUAGE:
        raise ValueError(f"unknown value type: {value_type}")
    if not key:
        raise ValueError("query key must not be empty")
    language = VALUE_LANGUAGE[value_type]
    if value_type == "num":
        value_shape = (
            "The access code consists of exactly three three-digit groups "
            "separated by single spaces. "
        )
    else:
        value_shape = (
            "The verification phrase consists of exactly three words separated "
            "by single spaces. "
        )
    return INDEPENDENT_QUERY_TEMPLATES[template_id].format(
        record_kind=language.record_kind,
        item=language.singular,
        key=key,
        value_shape=value_shape,
    )


def render_needle(value_type: str, key: str, value: str) -> str:
    if value_type not in VALUE_LANGUAGE:
        raise ValueError(f"unknown value type: {value_type}")
    singular = VALUE_LANGUAGE[value_type].singular
    return f'\nThe {singular} associated with record key "{key}" is "{value}".\n'


def generate_values(value_type: str, count: int, rng: random.Random) -> tuple[str, ...]:
    if count <= 0 or count > 8:
        raise ValueError("value count must be in [1, 8]")
    if value_type == "num":
        chunks = rng.sample(range(1000), 3 * count)
        return tuple(
            " ".join(f"{number:03d}" for number in chunks[index : index + 3])
            for index in range(0, len(chunks), 3)
        )
    if value_type == "word":
        modifiers = rng.sample(WORD_MODIFIERS, count)
        colors = rng.sample(WORD_COLORS, count)
        objects = rng.sample(WORD_OBJECTS, count)
        return tuple(
            f"{modifier} {color} {object_name}"
            for modifier, color, object_name in zip(
                modifiers, colors, objects, strict=True
            )
        )
    raise ValueError(f"unknown value type: {value_type}")


def render_answer(values_in_query_order: Sequence[str]) -> str:
    if not values_in_query_order:
        raise ValueError("answer must contain at least one value")
    return "; ".join(values_in_query_order)
