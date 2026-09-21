"""Lazy Instruct-model materialization of compact manifest rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Sequence

from ...core.layout import BlockLayout
from .manifest import SCHEMA_VERSION, SamplePlan
from .contracts import tokenizer_names_compatible
from .templates import (
    HAYSTACK_SEPARATOR,
    HAYSTACK_TEXT,
    render_independent_query,
)


_CONTEXT_MARKER = "__DISTANCE_KV_CONTEXT_31F947__"
_QUERY_MARKER = "__DISTANCE_KV_QUERY_C6A205__"
_ANSWER_MARKER = "__DISTANCE_KV_ANSWER_821DB4__"


@dataclass(frozen=True, slots=True)
class ChatParts:
    prefix_ids: tuple[int, ...]
    pre_query_separator_ids: tuple[int, ...]
    post_query_ids: tuple[int, ...]
    answer_tail_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ValueTokenSpan:
    value_index: int
    value: str
    char_start: int
    char_end: int
    answer_token_start: int
    answer_token_end: int
    absolute_token_start: int
    absolute_token_end: int


@dataclass(frozen=True, slots=True)
class MaterializedSample:
    plan: SamplePlan
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    query_start: int
    query_end: int
    answer_start: int
    answer_end: int
    full_length: int
    suffix_length: int
    value_token_spans: tuple[ValueTokenSpan, ...]
    input_ids_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "sample_id": self.plan.sample_id,
            "value_type": self.plan.value_type,
            "needle_count": self.plan.needle_count,
            "template_id": self.plan.template_id,
            "target_blocks": [needle.absolute_block for needle in self.plan.needles],
            "relative_distances": [
                needle.relative_distance for needle in self.plan.needles
            ],
            "query_start": self.query_start,
            "query_end": self.query_end,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "suffix_length": self.suffix_length,
            "full_length": self.full_length,
            "supervised_tokens": sum(self.loss_mask),
            "input_ids_sha256": self.input_ids_sha256,
            "value_token_spans": [
                {
                    "value_index": span.value_index,
                    "value": span.value,
                    "char_span": [span.char_start, span.char_end],
                    "answer_token_span": [
                        span.answer_token_start,
                        span.answer_token_end,
                    ],
                    "absolute_token_span": [
                        span.absolute_token_start,
                        span.absolute_token_end,
                    ],
                }
                for span in self.value_token_spans
            ],
        }


@dataclass(frozen=True, slots=True)
class IndependentQuerySuffix:
    query_position: int
    needle_index: int
    key: str
    value: str
    relative_distance: int
    gate_index: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    query_end: int
    answer_start: int
    answer_end: int
    suffix_length: int
    input_ids_sha256: str


@dataclass(frozen=True, slots=True)
class IndependentMaterializedSample:
    plan: SamplePlan
    prompt_input_ids: tuple[int, ...]
    prompt_ids_sha256: str
    suffixes: tuple[IndependentQuerySuffix, ...]


def _split_once(text: str, marker: str) -> tuple[str, str]:
    if text.count(marker) != 1:
        raise ValueError(f"chat template did not preserve marker exactly once: {marker}")
    before, after = text.split(marker, 1)
    return before, after


def _hash_ids(input_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in input_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


class InstructMaterializer:
    """Materialize one plan with query token zero fixed at context_tokens."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        layout: BlockLayout,
        haystack_text: str = HAYSTACK_TEXT,
        haystack_separator: str = HAYSTACK_SEPARATOR,
    ) -> None:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Instruct materialization requires tokenizer.chat_template")
        self.tokenizer = tokenizer
        self.layout = layout
        self.tokenizer_name = str(getattr(tokenizer, "name_or_path", "unknown"))
        self.chat_parts = self._render_chat_parts()
        filler_text = haystack_text + haystack_separator
        self.filler_ids = tuple(tokenizer.encode(filler_text, add_special_tokens=False))
        if not self.filler_ids:
            raise ValueError("haystack filler encodes to zero tokens")
        if len(self.chat_parts.prefix_ids) > layout.block_size:
            raise ValueError(
                "chat prefix does not fit in fixed sink block: "
                f"{len(self.chat_parts.prefix_ids)} > {layout.block_size}"
            )

    def _encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def _render_chat_parts(self) -> ChatParts:
        user_content = _CONTEXT_MARKER + "\n\n" + _QUERY_MARKER
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_text, after_context = _split_once(prompt_text, _CONTEXT_MARKER)
        separator_text, post_query_text = _split_once(after_context, _QUERY_MARKER)
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": _ANSWER_MARKER},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        _, answer_tail_text = _split_once(full_text, _ANSWER_MARKER)
        return ChatParts(
            prefix_ids=self._encode(prefix_text),
            pre_query_separator_ids=self._encode(separator_text),
            post_query_ids=self._encode(post_query_text),
            answer_tail_ids=self._encode(answer_tail_text),
        )

    def _filler(self, length: int) -> list[int]:
        if length < 0:
            raise ValueError("chat prefix and separator exceed context budget")
        quotient, remainder = divmod(length, len(self.filler_ids))
        return list(self.filler_ids) * quotient + list(self.filler_ids[:remainder])

    def _answer_tokens(
        self,
        answer: str,
        values: Sequence[str],
        *,
        answer_start: int,
    ) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[ValueTokenSpan, ...]]:
        """Tokenize value segments exactly and leave delimiters unsupervised.

        Some converted Llama fast tokenizers expose unusable zero-width offset
        mappings.  Segment-level tokenization is stricter for this fixed answer
        grammar: every ``value`` or ``; value`` fragment is encoded separately,
        then the concatenation must exactly equal tokenizing the full answer.
        """
        answer_ids = tuple(self.tokenizer.encode(answer, add_special_tokens=False))
        delimiter_text = "; "
        separator_ids = tuple(
            self.tokenizer.encode(";", add_special_tokens=False)
        )
        if not separator_ids:
            raise ValueError("answer separator encodes to zero tokens")

        reconstructed_ids: list[int] = []
        answer_mask: list[bool] = []
        value_spans: list[ValueTokenSpan] = []
        char_cursor = 0
        token_cursor = 0
        for value_index, value in enumerate(values):
            if value_index == 0:
                fragment = value
                delimiter_length = 0
            else:
                fragment = delimiter_text + value
                delimiter_length = len(separator_ids)
            fragment_ids = tuple(
                self.tokenizer.encode(fragment, add_special_tokens=False)
            )
            if delimiter_length and fragment_ids[:delimiter_length] != separator_ids:
                raise ValueError(
                    "answer separator merges with the following value; change the "
                    "answer grammar or value pool"
                )
            while delimiter_length < len(fragment_ids):
                decoded_token = self.tokenizer.decode(
                    [fragment_ids[delimiter_length]],
                    skip_special_tokens=False,
                )
                if not decoded_token.isspace():
                    break
                delimiter_length += 1
            value_ids = fragment_ids[delimiter_length:]
            if not value_ids:
                raise AssertionError(f"value has no tokens: {value}")
            decoded = self.tokenizer.decode(value_ids, skip_special_tokens=False)
            if decoded.lstrip(" ") != value:
                raise AssertionError(
                    f"value token span decodes to {decoded!r}, expected {value!r}"
                )

            char_start = char_cursor + (len(delimiter_text) if value_index else 0)
            char_end = char_start + len(value)
            value_token_start = token_cursor + delimiter_length
            value_token_end = value_token_start + len(value_ids)
            value_spans.append(
                ValueTokenSpan(
                    value_index=value_index,
                    value=value,
                    char_start=char_start,
                    char_end=char_end,
                    answer_token_start=value_token_start,
                    answer_token_end=value_token_end,
                    absolute_token_start=answer_start + value_token_start,
                    absolute_token_end=answer_start + value_token_end,
                )
            )
            reconstructed_ids.extend(fragment_ids)
            answer_mask.extend([False] * delimiter_length)
            answer_mask.extend([True] * len(value_ids))
            char_cursor = char_end
            token_cursor += len(fragment_ids)

        if char_cursor != len(answer):
            raise AssertionError("answer_values do not reconstruct the answer")
        if tuple(reconstructed_ids) != answer_ids:
            raise ValueError(
                "segment tokenization differs from full-answer tokenization; "
                "the answer grammar is not boundary-stable"
            )
        if len(answer_mask) != len(answer_ids):
            raise AssertionError("answer mask length does not match answer IDs")
        return answer_ids, tuple(answer_mask), tuple(value_spans)

    def materialize(self, plan: SamplePlan) -> MaterializedSample:
        if plan.schema_version != SCHEMA_VERSION:
            raise ValueError(
                "sample schema differs from the active data contract: "
                f"{plan.schema_version} != {SCHEMA_VERSION}"
            )
        if not tokenizer_names_compatible(
            plan.tokenizer_name_or_path, self.tokenizer_name
        ):
            raise ValueError(
                "manifest tokenizer is not compatible with materializer tokenizer: "
                f"{plan.tokenizer_name_or_path!r} != {self.tokenizer_name!r}"
            )
        prefix = self.chat_parts.prefix_ids
        separator = self.chat_parts.pre_query_separator_ids
        filler_length = self.layout.context_tokens - len(prefix) - len(separator)
        context_ids = list(prefix)
        context_ids.extend(self._filler(filler_length))
        context_ids.extend(separator)
        if len(context_ids) != self.layout.context_tokens:
            raise AssertionError("pre-query context did not hit the exact anchor")

        for needle in plan.needles:
            needle_ids = tuple(
                self.tokenizer.encode(needle.text, add_special_tokens=False)
            )
            if len(needle_ids) != needle.token_length:
                raise ValueError("manifest needle length does not match tokenizer")
            if needle.token_end - needle.token_start != len(needle_ids):
                raise AssertionError("manifest needle span has the wrong width")
            block_start, block_end = self.layout.block_span(needle.absolute_block)
            if needle.token_start < block_start + self.layout.needle_margin_tokens:
                raise AssertionError("needle violates left block margin")
            if needle.token_end > block_end - self.layout.needle_margin_tokens:
                raise AssertionError("needle violates right block margin")
            context_ids[needle.token_start : needle.token_end] = needle_ids

        query_ids = tuple(self.tokenizer.encode(plan.query, add_special_tokens=False))
        query_start = self.layout.context_tokens
        query_end = query_start + len(query_ids)
        answer_start = query_end + len(self.chat_parts.post_query_ids)
        answer_ids, answer_mask, value_spans = self._answer_tokens(
            plan.answer, plan.answer_values, answer_start=answer_start
        )
        answer_end = answer_start + len(answer_ids)
        input_ids = tuple(
            context_ids
            + list(query_ids)
            + list(self.chat_parts.post_query_ids)
            + list(answer_ids)
            + list(self.chat_parts.answer_tail_ids)
        )
        suffix_length = len(input_ids) - self.layout.context_tokens
        if suffix_length > self.layout.suffix_capacity_tokens:
            raise ValueError(
                f"suffix needs {suffix_length} tokens; capacity is "
                f"{self.layout.suffix_capacity_tokens}"
            )
        if len(input_ids) > self.layout.total_tokens:
            raise AssertionError("materialized sample exceeds total token budget")

        loss_mask = [False] * len(input_ids)
        labels = [-100] * len(input_ids)
        for answer_token_index, supervised in enumerate(answer_mask):
            if not supervised:
                continue
            absolute_index = answer_start + answer_token_index
            loss_mask[absolute_index] = True
            labels[absolute_index] = answer_ids[answer_token_index]
        expected_supervised = sum(
            span.answer_token_end - span.answer_token_start for span in value_spans
        )
        if sum(loss_mask) != expected_supervised:
            raise AssertionError("loss mask does not equal the union of value spans")

        return MaterializedSample(
            plan=plan,
            input_ids=input_ids,
            labels=tuple(labels),
            loss_mask=tuple(loss_mask),
            query_start=query_start,
            query_end=query_end,
            answer_start=answer_start,
            answer_end=answer_end,
            full_length=len(input_ids),
            suffix_length=suffix_length,
            value_token_spans=value_spans,
            input_ids_sha256=_hash_ids(input_ids),
        )

    def materialize_independent_queries(
        self, plan: SamplePlan
    ) -> IndependentMaterializedSample:
        """Reuse one fixed prompt and build one single-key suffix per needle."""
        if plan.template_id in {"X1", "X2"}:
            first_needle_index = plan.query_order[0]
            first_needle = plan.needles[first_needle_index]
            prompt_plan = replace(
                plan,
                query=render_independent_query(
                    plan.template_id,
                    plan.value_type,
                    first_needle.key,
                ),
                answer=first_needle.value,
                answer_values=(first_needle.value,),
            )
            combined = self.materialize(prompt_plan)
        else:
            combined = self.materialize(plan)
        prompt_ids = combined.input_ids[: self.layout.context_tokens]
        if len(prompt_ids) != self.layout.context_tokens:
            raise AssertionError("independent prompt does not hit the fixed anchor")

        suffixes: list[IndependentQuerySuffix] = []
        triples = zip(
            plan.query_order,
            plan.query_keys,
            plan.answer_values,
            strict=True,
        )
        for query_position, (needle_index, key, value) in enumerate(triples):
            needle = plan.needles[needle_index]
            if needle.key != key or needle.value != value:
                raise AssertionError("query order does not map to the selected needle")

            query = render_independent_query(
                plan.template_id,
                plan.value_type,
                key,
            )
            query_ids = self._encode(query)
            query_end = len(query_ids)
            answer_start = query_end + len(self.chat_parts.post_query_ids)
            answer_ids, answer_mask, _ = self._answer_tokens(
                value,
                (value,),
                answer_start=answer_start,
            )
            answer_end = answer_start + len(answer_ids)
            suffix_ids = tuple(
                list(query_ids)
                + list(self.chat_parts.post_query_ids)
                + list(answer_ids)
                + list(self.chat_parts.answer_tail_ids)
            )
            if len(suffix_ids) > self.layout.suffix_capacity_tokens:
                raise ValueError(
                    "independent suffix needs "
                    f"{len(suffix_ids)} tokens; capacity is "
                    f"{self.layout.suffix_capacity_tokens}"
                )
            labels = [-100] * len(suffix_ids)
            loss_mask = [False] * len(suffix_ids)
            for answer_token_index, supervised in enumerate(answer_mask):
                if supervised:
                    suffix_index = answer_start + answer_token_index
                    labels[suffix_index] = answer_ids[answer_token_index]
                    loss_mask[suffix_index] = True
            if not any(loss_mask):
                raise AssertionError("independent suffix has no supervised tokens")

            suffixes.append(
                IndependentQuerySuffix(
                    query_position=query_position,
                    needle_index=needle_index,
                    key=key,
                    value=value,
                    relative_distance=needle.relative_distance,
                    gate_index=needle.gate_index,
                    input_ids=suffix_ids,
                    labels=tuple(labels),
                    loss_mask=tuple(loss_mask),
                    query_end=query_end,
                    answer_start=answer_start,
                    answer_end=answer_end,
                    suffix_length=len(suffix_ids),
                    input_ids_sha256=_hash_ids(suffix_ids),
                )
            )

        if len(suffixes) != plan.needle_count:
            raise AssertionError("independent branch count differs from needle count")
        return IndependentMaterializedSample(
            plan=plan,
            prompt_input_ids=prompt_ids,
            prompt_ids_sha256=_hash_ids(prompt_ids),
            suffixes=tuple(suffixes),
        )
