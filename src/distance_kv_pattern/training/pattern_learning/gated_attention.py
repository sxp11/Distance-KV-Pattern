"""Llama-4.45.0-style attention with prompt-block retention gates.

The operation order follows ``LlamaAttention.forward``: QK scores, causal
mask, fp32 softmax, dropout, and AV aggregation. The ordinary softmax is the
only semantic change: prompt mass is multiplied by a differentiable block gate
before prompt and suffix positions share one normalization denominator.
Native GQA grouping avoids materializing ``repeat_kv`` at 128K.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class GatedAttentionOutput:
    """Grouped-query attention output and optional diagnostic probabilities."""

    output: torch.Tensor
    prompt_probabilities: torch.Tensor | None = None
    suffix_probabilities: torch.Tensor | None = None


def build_prompt_block_gate(
    learned_gate: torch.Tensor,
    *,
    sink_blocks: int,
    recent_blocks: int,
) -> torch.Tensor:
    """Add immutable sink/recent ones around learned distance gates.

    learned_gate is ordered by increasing relative distance, while prompt
    blocks are ordered from the document start. The learnable segment must
    therefore be reversed between the sink and recent blocks.
    """

    if learned_gate.ndim < 2:
        raise ValueError("learned_gate must end in [heads, distances]")
    if sink_blocks < 0 or recent_blocks < 0:
        raise ValueError("fixed block counts cannot be negative")
    prefix_shape = learned_gate.shape[:-1]
    sink = learned_gate.new_ones(*prefix_shape, sink_blocks)
    recent = learned_gate.new_ones(*prefix_shape, recent_blocks)
    return torch.cat((sink, learned_gate.flip(-1), recent), dim=-1)


def _suffix_causal_mask(
    query_length: int,
    key_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if query_length != key_length:
        raise ValueError(
            "the reference suffix path expects one K/V position per suffix query"
        )
    return torch.ones(
        query_length,
        key_length,
        dtype=torch.bool,
        device=device,
    ).tril()


def _prompt_gated_softmax(
    prompt_attn_weights: torch.Tensor,
    suffix_attn_weights: torch.Tensor,
    block_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Llama's fp32 softmax with a gate on prompt attention mass.

    This evaluates ``softmax([S_prompt + log(z), S_suffix])`` in log space and
    masks the Hard Concrete endpoint zero before exponentiation.
    """

    prompt_attn_weights = prompt_attn_weights.float()
    suffix_attn_weights = suffix_attn_weights.float()
    block_gate = block_gate.float()

    positive_gate = block_gate > 0
    safe_gate = torch.where(
        positive_gate,
        block_gate,
        torch.ones_like(block_gate),
    )
    gated_prompt_weights = (prompt_attn_weights + safe_gate.log()).masked_fill(
        ~positive_gate,
        float("-inf"),
    )
    prompt_max = gated_prompt_weights.amax(dim=(-2, -1))
    suffix_max = suffix_attn_weights.amax(dim=-1)
    row_max = torch.maximum(prompt_max, suffix_max).detach()

    prompt_weights = torch.exp(gated_prompt_weights - row_max[..., None, None])
    suffix_weights = torch.exp(suffix_attn_weights - row_max[..., None])
    denominator = (
        prompt_weights.sum(dim=(-2, -1)) + suffix_weights.sum(dim=-1)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    return (
        prompt_weights / denominator[..., None, None],
        suffix_weights / denominator[..., None],
    )


def _block_gated_attention(
    query: torch.Tensor,
    prompt_key: torch.Tensor,
    prompt_value: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    prompt_block_gate: torch.Tensor,
    *,
    binary_prompt_block_gate: torch.Tensor | None = None,
    gate_granularity: str,
    block_size: int,
    suffix_causal_mask: torch.Tensor | None = None,
    scale: float | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_probabilities: bool = False,
) -> GatedAttentionOutput:
    """Attend suffix Q heads to gated prompt KV and always-on suffix KV.

    Shapes:
      query: [batch, q_heads, suffix_queries, head_dim]
      prompt K/V: [batch, kv_heads, prompt_tokens, head_dim]
      suffix K/V: [batch, kv_heads, suffix_tokens, head_dim]
      KV-head gate: [kv_heads, prompt_blocks] or batched equivalent.
      Q-head gate: [q_heads, prompt_blocks] or batched equivalent.

    Two storage-only transformations differ from the literal HF source:

    * Q is grouped by shared KV head instead of materializing ``repeat_kv``;
    * prompt and suffix AV stay separate instead of copying the 128K cache.

    These transformations are algebraically identical and share one softmax.
    """

    tensors = (query, prompt_key, prompt_value, suffix_key, suffix_value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("query, key and value tensors must all be rank four")
    batch, num_q_heads, query_length, head_dim = query.shape
    prompt_batch, num_kv_heads, prompt_tokens, key_dim = prompt_key.shape
    if prompt_value.shape != prompt_key.shape:
        raise ValueError("prompt key and value shapes must match")
    if suffix_key.shape != suffix_value.shape:
        raise ValueError("suffix key and value shapes must match")
    if prompt_batch != batch or suffix_key.shape[0] != batch:
        raise ValueError("query and KV batch dimensions must match")
    if suffix_key.shape[1] != num_kv_heads:
        raise ValueError("prompt and suffix KV-head counts must match")
    if key_dim != head_dim or suffix_key.shape[-1] != head_dim:
        raise ValueError("query, key and value head dimensions must match")
    if num_q_heads % num_kv_heads:
        raise ValueError("Q-head count must be divisible by KV-head count")
    if block_size <= 0 or prompt_tokens % block_size:
        raise ValueError("prompt length must be divisible by block_size")
    if not 0 <= dropout_p < 1:
        raise ValueError("dropout_p must lie inside [0, 1)")

    if gate_granularity not in ("q_head", "kv_head"):
        raise ValueError("gate_granularity must be 'q_head' or 'kv_head'")
    num_prompt_units = prompt_tokens // block_size
    num_gate_heads = num_q_heads if gate_granularity == "q_head" else num_kv_heads
    expected_unbatched = (num_gate_heads, num_prompt_units)
    expected_batched = (batch, num_gate_heads, num_prompt_units)
    if prompt_block_gate.ndim == 2:
        if tuple(prompt_block_gate.shape) != expected_unbatched:
            raise ValueError(
                "prompt_block_gate has shape "
                f"{tuple(prompt_block_gate.shape)}, expected {expected_unbatched} "
                f"for {gate_granularity} gates"
            )
        block_gate = prompt_block_gate.unsqueeze(0)
    elif prompt_block_gate.ndim == 3:
        if tuple(prompt_block_gate.shape) != expected_batched:
            raise ValueError(
                "batched prompt_block_gate has shape "
                f"{tuple(prompt_block_gate.shape)}, expected {expected_batched} "
                f"for {gate_granularity} gates"
            )
        block_gate = prompt_block_gate
    else:
        raise ValueError("prompt_block_gate must be rank two or three")

    binary_block_gate = None
    if binary_prompt_block_gate is not None:
        if binary_prompt_block_gate.shape != prompt_block_gate.shape:
            raise ValueError("relaxed and binary prompt block gates must match")
        binary_block_gate = (
            binary_prompt_block_gate.unsqueeze(0)
            if binary_prompt_block_gate.ndim == 2
            else binary_prompt_block_gate
        )

    suffix_tokens = suffix_key.shape[2]
    if suffix_causal_mask is None:
        suffix_causal_mask = _suffix_causal_mask(
            query_length,
            suffix_tokens,
            device=query.device,
        )
    if suffix_causal_mask.dtype != torch.bool:
        raise ValueError("suffix_causal_mask must be boolean")
    if suffix_causal_mask.shape == (query_length, suffix_tokens):
        suffix_causal_mask = suffix_causal_mask.view(
            1, 1, 1, query_length, suffix_tokens
        )
    elif suffix_causal_mask.shape == (batch, query_length, suffix_tokens):
        suffix_causal_mask = suffix_causal_mask.view(
            batch, 1, 1, query_length, suffix_tokens
        )
    else:
        raise ValueError("suffix_causal_mask has incompatible shape")

    group_size = num_q_heads // num_kv_heads
    grouped_query = query.reshape(
        batch,
        num_kv_heads,
        group_size,
        query_length,
        head_dim,
    )
    attention_scale = head_dim ** -0.5 if scale is None else float(scale)
    prompt_scores = torch.einsum(
        "bhgqd,bhkd->bhgqk",
        grouped_query,
        prompt_key,
    ) * attention_scale
    suffix_scores = torch.einsum(
        "bhgqd,bhsd->bhgqs",
        grouped_query,
        suffix_key,
    ) * attention_scale
    suffix_scores = suffix_scores.masked_fill(~suffix_causal_mask, float("-inf"))

    gate_batch = batch if prompt_block_gate.ndim == 3 else 1
    if gate_granularity == "q_head":
        block_gate = block_gate.view(
            gate_batch,
            num_kv_heads,
            group_size,
            1,
            num_prompt_units,
        )
    else:
        block_gate = block_gate.view(
            gate_batch,
            num_kv_heads,
            1,
            1,
            num_prompt_units,
        )

    if binary_block_gate is not None:
        binary_block_gate = binary_block_gate.view(block_gate.shape)

    prompt_scores = prompt_scores.reshape(
        batch,
        num_kv_heads,
        group_size,
        query_length,
        num_prompt_units,
        block_size,
    )
    block_gate = block_gate.unsqueeze(-1)
    if binary_block_gate is not None:
        binary_block_gate = binary_block_gate.unsqueeze(-1)

    # This is the only semantic replacement for Llama's fp32 softmax line.
    prompt_probabilities, suffix_probabilities = _prompt_gated_softmax(
        prompt_scores,
        suffix_scores,
        block_gate,
    )
    if binary_block_gate is not None:
        binary_prompt_probabilities, binary_suffix_probabilities = (
            _prompt_gated_softmax(
                prompt_scores,
                suffix_scores,
                binary_block_gate,
            )
        )
        prompt_probabilities = prompt_probabilities + (
            binary_prompt_probabilities - prompt_probabilities
        ).detach()
        suffix_probabilities = suffix_probabilities + (
            binary_suffix_probabilities - suffix_probabilities
        ).detach()

    if dropout_p:
        prompt_probabilities = F.dropout(
            prompt_probabilities,
            p=dropout_p,
            training=training,
        )
        suffix_probabilities = F.dropout(
            suffix_probabilities,
            p=dropout_p,
            training=training,
        )

    prompt_probabilities_flat = prompt_probabilities.reshape(
        batch,
        num_kv_heads,
        group_size,
        query_length,
        prompt_tokens,
    )
    output_dtype = prompt_value.dtype
    prompt_output = torch.einsum(
        "bhgqk,bhkd->bhgqd",
        prompt_probabilities_flat.to(output_dtype),
        prompt_value,
    )
    suffix_output = torch.einsum(
        "bhgqs,bhsd->bhgqd",
        suffix_probabilities.to(output_dtype),
        suffix_value,
    )
    output = (prompt_output + suffix_output).reshape(
        batch,
        num_q_heads,
        query_length,
        head_dim,
    )

    if not return_probabilities:
        return GatedAttentionOutput(output=output)
    return GatedAttentionOutput(
        output=output,
        prompt_probabilities=prompt_probabilities_flat,
        suffix_probabilities=suffix_probabilities,
    )


def kv_head_block_gated_attention(
    query: torch.Tensor,
    prompt_key: torch.Tensor,
    prompt_value: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    prompt_block_gate: torch.Tensor,
    *,
    binary_prompt_block_gate: torch.Tensor | None = None,
    block_size: int,
    suffix_causal_mask: torch.Tensor | None = None,
    scale: float | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_probabilities: bool = False,
) -> GatedAttentionOutput:
    """Apply one distance gate per KV head, shared by its GQA Q heads."""

    return _block_gated_attention(
        query,
        prompt_key,
        prompt_value,
        suffix_key,
        suffix_value,
        prompt_block_gate,
        binary_prompt_block_gate=binary_prompt_block_gate,
        gate_granularity="kv_head",
        block_size=block_size,
        suffix_causal_mask=suffix_causal_mask,
        scale=scale,
        dropout_p=dropout_p,
        training=training,
        return_probabilities=return_probabilities,
    )


def q_head_block_gated_attention(
    query: torch.Tensor,
    prompt_key: torch.Tensor,
    prompt_value: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    prompt_block_gate: torch.Tensor,
    *,
    binary_prompt_block_gate: torch.Tensor | None = None,
    block_size: int,
    suffix_causal_mask: torch.Tensor | None = None,
    scale: float | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_probabilities: bool = False,
) -> GatedAttentionOutput:
    """Apply an independent distance gate to every Q head in each GQA group."""

    return _block_gated_attention(
        query,
        prompt_key,
        prompt_value,
        suffix_key,
        suffix_value,
        prompt_block_gate,
        binary_prompt_block_gate=binary_prompt_block_gate,
        gate_granularity="q_head",
        block_size=block_size,
        suffix_causal_mask=suffix_causal_mask,
        scale=scale,
        dropout_p=dropout_p,
        training=training,
        return_probabilities=return_probabilities,
    )


# Backward-compatible name for the original KV-head implementation.
grouped_block_gated_attention = kv_head_block_gated_attention
