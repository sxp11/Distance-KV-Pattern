"""Tokenizer-level invariants for the frozen data design."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

from ...core.layout import BlockLayout
from .templates import (
    ALL_TEMPLATE_IDS,
    HAYSTACK_TEXT,
    KEY_POOL,
    VALUE_LANGUAGE,
    WORD_COLORS,
    WORD_MODIFIERS,
    WORD_OBJECTS,
    render_needle,
    render_query,
)


FORBIDDEN_HAYSTACK_TERMS = (
    "record",
    "key",
    "label",
    "value",
    "code",
    "phrase",
    *KEY_POOL,
    *WORD_MODIFIERS,
    *WORD_COLORS,
    *WORD_OBJECTS,
)


def tokenizer_names_compatible(manifest_name: str, runtime_name: str) -> bool:
    """Accept a relocated copy of the same tokenizer.

    Manifests intentionally record the tokenizer location for provenance.  A
    relocation of the project/model tree must not invalidate an otherwise
    identical manifest, so absolute paths are compared first and then by a
    conservative normalized model-directory identity.  The materializer still
    validates every manifest needle's token length/boundaries and the complete
    answer grammar, which catches an actually different tokenizer.
    """

    if manifest_name == runtime_name:
        return True

    def identity(value: str) -> str:
        name = Path(value).name.lower()
        # Hugging Face model mirrors often differ only by a vendor prefix.
        name = re.sub(r"^(meta[-_])", "", name)
        return re.sub(r"[^a-z0-9]+", "", name)

    manifest_identity = identity(manifest_name)
    runtime_identity = identity(runtime_name)
    return bool(manifest_identity) and manifest_identity == runtime_identity


def validate_tokenizer_contract(tokenizer: Any, layout: BlockLayout) -> dict[str, int]:
    """Fail early if a tokenizer violates the agreed fixed-length design."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("the first implementation requires an Instruct chat template")

    bad_keys = {
        key: len(tokenizer.encode(key, add_special_tokens=False))
        for key in KEY_POOL
        if len(tokenizer.encode(key, add_special_tokens=False)) != 2
    }
    if bad_keys:
        raise ValueError(f"keys are not exactly two standalone tokens: {bad_keys}")

    lower_haystack = HAYSTACK_TEXT.lower()
    collisions = [
        term for term in FORBIDDEN_HAYSTACK_TERMS if term.lower() in lower_haystack
    ]
    if collisions:
        raise ValueError(f"haystack contains forbidden task vocabulary: {collisions}")
    if any(character.isdigit() for character in HAYSTACK_TEXT):
        raise ValueError("haystack must not contain digits")

    numeric_chunk_tokens = len(
        tokenizer.encode("000", add_special_tokens=False)
    )
    for number in range(1000):
        chunk = f"{number:03d}"
        if len(tokenizer.encode(chunk, add_special_tokens=False)) != numeric_chunk_tokens:
            raise ValueError(f"numeric chunks have inconsistent token lengths: {chunk}")
        if len(tokenizer.encode(" " + chunk, add_special_tokens=False)) != numeric_chunk_tokens + 1:
            raise ValueError(f"space-prefixed numeric chunk is unstable: {chunk}")
        delimiter_ids = tokenizer.encode(";", add_special_tokens=False)
        prefixed_ids = tokenizer.encode("; " + chunk, add_special_tokens=False)
        if prefixed_ids[: len(delimiter_ids)] != delimiter_ids:
            raise ValueError(f"delimiter-prefixed numeric chunk is unstable: {chunk}")
        value_ids = prefixed_ids[len(delimiter_ids) :]
        while value_ids and tokenizer.decode([value_ids[0]]).isspace():
            value_ids = value_ids[1:]
        if tokenizer.decode(value_ids).lstrip(" ") != chunk:
            raise ValueError(f"numeric value boundary is unstable: {chunk}")

    word_phrases = tuple(
        f"{modifier} {color} {object_name}"
        for modifier, color, object_name in itertools.product(
            WORD_MODIFIERS, WORD_COLORS, WORD_OBJECTS
        )
    )
    for phrase in word_phrases:
        if len(tokenizer.encode(phrase, add_special_tokens=False)) != 5:
            raise ValueError(f"word phrase is not five tokens: {phrase}")
        delimiter_ids = tokenizer.encode(";", add_special_tokens=False)
        prefixed_ids = tokenizer.encode("; " + phrase, add_special_tokens=False)
        if prefixed_ids[: len(delimiter_ids)] != delimiter_ids:
            raise ValueError(f"delimiter-prefixed word phrase is unstable: {phrase}")
        value_ids = prefixed_ids[len(delimiter_ids) :]
        while value_ids and tokenizer.decode([value_ids[0]]).isspace():
            value_ids = value_ids[1:]
        if tokenizer.decode(value_ids).lstrip(" ") != phrase:
            raise ValueError(f"word value boundary is unstable: {phrase}")

    representative_num = "007 042 519"
    max_needle_tokens = 0
    for key in KEY_POOL:
        for value_type, value in (
            ("num", representative_num),
            ("word", word_phrases[0]),
        ):
            length = len(
                tokenizer.encode(
                    render_needle(value_type, key, value),
                    add_special_tokens=False,
                )
            )
            max_needle_tokens = max(max_needle_tokens, length)
            if length > layout.usable_needle_tokens:
                raise ValueError(
                    f"{value_type} needle for {key} uses {length} tokens; "
                    f"maximum is {layout.usable_needle_tokens}"
                )

    max_query_tokens = 0
    keys = KEY_POOL[:8]
    for template_id in ALL_TEMPLATE_IDS:
        for value_type in VALUE_LANGUAGE:
            max_query_tokens = max(
                max_query_tokens,
                len(
                    tokenizer.encode(
                        render_query(template_id, value_type, keys),
                        add_special_tokens=False,
                    )
                ),
            )
    return {
        "num_keys": len(KEY_POOL),
        "num_word_phrases": len(word_phrases),
        "numeric_chunk_tokens": numeric_chunk_tokens,
        "max_needle_tokens": max_needle_tokens,
        "max_query_tokens_without_chat_wrapper": max_query_tokens,
    }
