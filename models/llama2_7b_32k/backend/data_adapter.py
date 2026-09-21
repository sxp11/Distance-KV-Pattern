"""Llama2 tokenizer materialization and fixed 32K block layout."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from distance_kv_pattern.core import BlockLayout
from distance_kv_pattern.training.data_pipeline.contracts import (
    FORBIDDEN_HAYSTACK_TERMS,
)
from distance_kv_pattern.training.data_pipeline.materialize import (
    InstructMaterializer,
    ValueTokenSpan,
)
from distance_kv_pattern.training.data_pipeline.templates import (
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


def build_llama2_layout() -> BlockLayout:
    return BlockLayout(
        total_tokens=32768,
        context_tokens=32512,
        block_size=128,
        sink_blocks=1,
        recent_blocks=8,
        needle_margin_tokens=16,
    )


class Llama2InstructMaterializer(InstructMaterializer):
    """Locate Llama2 SentencePiece answer values by character offsets."""

    def _answer_tokens(
        self,
        answer: str,
        values: Sequence[str],
        *,
        answer_start: int,
    ) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[ValueTokenSpan, ...]]:
        encoded = self.tokenizer(
            answer,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        answer_ids = tuple(encoded["input_ids"])
        offsets = tuple(tuple(offset) for offset in encoded["offset_mapping"])
        answer_mask = [False] * len(answer_ids)
        value_spans: list[ValueTokenSpan] = []
        char_cursor = 0

        for value_index, value in enumerate(values):
            char_start = char_cursor + (2 if value_index else 0)
            char_end = char_start + len(value)
            token_positions = tuple(
                token_index
                for token_index, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start
                and token_start < char_end
                and token_end > char_start
            )
            if token_positions != tuple(
                range(token_positions[0], token_positions[-1] + 1)
            ):
                raise ValueError("Llama2 value tokens are not contiguous")
            token_start = token_positions[0]
            token_end = token_positions[-1] + 1
            for token_index in token_positions:
                answer_mask[token_index] = True
            value_spans.append(
                ValueTokenSpan(
                    value_index=value_index,
                    value=value,
                    char_start=char_start,
                    char_end=char_end,
                    answer_token_start=token_start,
                    answer_token_end=token_end,
                    absolute_token_start=answer_start + token_start,
                    absolute_token_end=answer_start + token_end,
                )
            )
            char_cursor = char_end

        if char_cursor != len(answer):
            raise ValueError("answer values do not reconstruct the answer")
        return answer_ids, tuple(answer_mask), tuple(value_spans)


def validate_llama2_tokenizer_contract(
    tokenizer: Any,
    layout: BlockLayout,
) -> dict[str, int]:
    materializer = Llama2InstructMaterializer(tokenizer, layout=layout)
    lower_haystack = HAYSTACK_TEXT.lower()
    if any(term.lower() in lower_haystack for term in FORBIDDEN_HAYSTACK_TERMS):
        raise ValueError("haystack contains task vocabulary")
    if any(character.isdigit() for character in HAYSTACK_TEXT):
        raise ValueError("haystack must not contain digits")
    if any(not tokenizer.encode(key, add_special_tokens=False) for key in KEY_POOL):
        raise ValueError("record key encodes to zero tokens")

    numeric_values = tuple(
        f"{number:03d} {number:03d} {number:03d}" for number in range(1000)
    )
    word_values = tuple(
        f"{modifier} {color} {object_name}"
        for modifier, color, object_name in itertools.product(
            WORD_MODIFIERS,
            WORD_COLORS,
            WORD_OBJECTS,
        )
    )
    for value in numeric_values + word_values:
        materializer._answer_tokens(
            f"{value}; {value}",
            (value, value),
            answer_start=0,
        )

    max_needle_tokens = max(
        len(
            tokenizer.encode(
                render_needle(value_type, key, value),
                add_special_tokens=False,
            )
        )
        for key in KEY_POOL
        for value_type, values in (("num", numeric_values), ("word", word_values))
        for value in values
    )
    if max_needle_tokens > layout.usable_needle_tokens:
        raise ValueError("Llama2 needle exceeds the block interior")
    max_query_tokens = max(
        len(
            tokenizer.encode(
                render_query(template_id, value_type, KEY_POOL[:8]),
                add_special_tokens=False,
            )
        )
        for template_id in ALL_TEMPLATE_IDS
        for value_type in VALUE_LANGUAGE
    )
    numeric_chunk_tokens = len(tokenizer.encode("000", add_special_tokens=False))
    return {
        "num_keys": len(KEY_POOL),
        "num_word_phrases": len(word_values),
        "numeric_chunk_tokens": numeric_chunk_tokens,
        "max_needle_tokens": max_needle_tokens,
        "max_query_tokens_without_chat_wrapper": max_query_tokens,
    }
