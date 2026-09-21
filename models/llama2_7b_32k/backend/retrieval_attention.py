"""Llama2 MHA attention extraction for retrieval initialization."""

from __future__ import annotations

import contextlib
import types
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F
from distance_kv_pattern.core import BlockLayout
from distance_kv_pattern.training.retrieval_initialization.retrieval_init import (
    ATTENTION_METRICS,
    attention_step_metrics,
    mean_step_metrics,
    prediction_query_target_pairs,
)
from transformers.cache_utils import DynamicCache


def align_llama2_value_tokens(
    tokenizer: Any,
    *,
    needle_text: str,
    value: str,
    absolute_needle_start: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    value_start = needle_text.index(value)
    value_end = value_start + len(value)
    source_encoding = tokenizer(
        needle_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    source_tokens = tuple(
        (
            absolute_needle_start + token_index,
            max(token_start, value_start) - value_start,
            min(token_end, value_end) - value_start,
        )
        for token_index, (token_start, token_end) in enumerate(
            source_encoding["offset_mapping"]
        )
        if token_end > value_start and token_start < value_end
    )
    target_encoding = tokenizer(
        value,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    target_to_source = tuple(
        tuple(
            source_position
            for source_position, source_start, source_end in source_tokens
            if source_end > target_start and source_start < target_end
        )
        for target_start, target_end in target_encoding["offset_mapping"]
    )
    source_positions = tuple(source_position for source_position, _, _ in source_tokens)
    return target_to_source, source_positions


@contextlib.contextmanager
def temporary_eager_llama_attention(model: Any) -> Iterator[None]:
    from transformers.models.llama.modeling_llama import LlamaAttention

    originals: list[tuple[Any, Any]] = []
    try:
        for layer in model.model.layers:
            module = layer.self_attn
            originals.append((module, module.forward))
            module.forward = types.MethodType(LlamaAttention.forward, module)
        yield
    finally:
        for module, original in originals:
            module.forward = original


def score_branch(
    *,
    model: Any,
    tokenizer: Any,
    cache: DynamicCache,
    prompt_length: int,
    suffix: Any,
    needle: Any,
    layout: BlockLayout,
    retrieval_k: int,
) -> dict[str, Any]:
    suffix_ids = torch.tensor(
        suffix.input_ids,
        dtype=torch.long,
        device=next(model.parameters()).device,
    ).unsqueeze(0)
    pairs = prediction_query_target_pairs(suffix.labels)
    target_to_source, source_positions = align_llama2_value_tokens(
        tokenizer,
        needle_text=needle.text,
        value=needle.value,
        absolute_needle_start=needle.token_start,
    )
    if len(target_to_source) != len(pairs):
        raise ValueError(
            "Llama2 target-token and alignment counts differ: "
            f"{len(target_to_source)} != {len(pairs)}"
        )
    block_span = layout.block_span(needle.absolute_block)
    needle_span = (needle.token_start, needle.token_end)
    first_query_index = pairs[0][0]
    if first_query_index:
        prefix = suffix_ids[:, :first_query_index]
        prefix_output = model.model(
            input_ids=prefix,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        cache = prefix_output.past_key_values
        del prefix_output, prefix

    step_metrics: list[dict[str, torch.Tensor]] = []
    predicted_ids: list[int] = []
    target_ids: list[int] = []
    nlls: list[float] = []
    with temporary_eager_llama_attention(model):
        for step_index, (query_index, target_index) in enumerate(pairs):
            expected_cache_length = prompt_length + query_index
            if int(cache.get_seq_length()) != expected_cache_length:
                raise AssertionError(
                    "suffix cache and causal prediction row are misaligned"
                )
            current_token = suffix_ids[:, query_index : query_index + 1]
            output = model.model(
                input_ids=current_token,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            cache = output.past_key_values
            if output.attentions is None:
                raise RuntimeError("temporary eager forward returned no attentions")
            step_metrics.append(
                attention_step_metrics(
                    output.attentions,
                    aligned_source_position=target_to_source[step_index],
                    value_positions=source_positions,
                    needle_span=needle_span,
                    block_span=block_span,
                    retrieval_k=retrieval_k,
                )
            )
            logits = model.lm_head(output.last_hidden_state[:, -1, :]).float()
            target = suffix_ids[:, target_index]
            nlls.append(float(F.cross_entropy(logits, target).item()))
            predicted_ids.append(int(logits.argmax(dim=-1).item()))
            target_ids.append(int(target.item()))
            del output, logits, target, current_token

    averaged = mean_step_metrics(step_metrics)
    num_layers, num_q_heads = averaged[ATTENTION_METRICS[0]].shape
    return {
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": int(model.config.num_key_value_heads),
        "source_value_positions": list(source_positions),
        "source_needle_span": list(needle_span),
        "source_block_span": list(block_span),
        "prediction_query_indices": [pair[0] for pair in pairs],
        "target_token_indices": [pair[1] for pair in pairs],
        "target_token_ids": target_ids,
        "predicted_token_ids": predicted_ids,
        "teacher_token_accuracy": sum(
            predicted == target
            for predicted, target in zip(predicted_ids, target_ids, strict=True)
        )
        / len(target_ids),
        "teacher_exact_match": predicted_ids == target_ids,
        "teacher_mean_nll": sum(nlls) / len(nlls),
        "q_head_metrics": {
            metric: averaged[metric].tolist() for metric in ATTENTION_METRICS
        },
    }
