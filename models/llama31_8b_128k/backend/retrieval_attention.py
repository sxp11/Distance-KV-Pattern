"""Llama 3.1 attention extraction for retrieval-score initialization."""

from __future__ import annotations

import contextlib
import types
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from distance_kv_pattern.core import BlockLayout
from distance_kv_pattern.training.retrieval_initialization.retrieval_init import (
    ATTENTION_METRICS,
    attention_step_metrics,
    mean_step_metrics,
    prediction_query_target_pairs,
    source_value_token_positions,
)


@contextlib.contextmanager
def temporary_eager_llama_attention(model: Any) -> Iterator[None]:
    """Temporarily expose Llama attention rows for single-token suffix scoring."""

    from transformers.models.llama.modeling_llama import LlamaAttention

    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise TypeError("temporary eager scoring requires a Llama model")
    originals: list[tuple[Any, Any]] = []
    try:
        for layer in layers:
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
    """Score one Llama teacher-forced suffix from a shared prompt cache."""

    suffix_ids = torch.tensor(
        suffix.input_ids,
        dtype=torch.long,
        device=next(model.parameters()).device,
    ).unsqueeze(0)
    pairs = prediction_query_target_pairs(suffix.labels)
    source_positions = source_value_token_positions(
        tokenizer,
        needle_text=needle.text,
        value=needle.value,
        absolute_needle_start=needle.token_start,
    )
    if len(source_positions) != len(pairs):
        raise ValueError(
            "source value and supervised target token counts differ: "
            f"{len(source_positions)} != {len(pairs)}"
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
                    "suffix cache and causal prediction row are misaligned: "
                    f"{cache.get_seq_length()} != {expected_cache_length}"
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
                    aligned_source_position=source_positions[step_index],
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
    num_kv_heads = int(model.config.num_key_value_heads)
    if num_q_heads != int(model.config.num_attention_heads):
        raise AssertionError("returned Q-head count differs from model config")
    if int(cache.get_seq_length()) != prompt_length + pairs[-1][0] + 1:
        raise AssertionError("branch cache length is inconsistent after scoring")
    return {
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
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
