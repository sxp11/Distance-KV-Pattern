"""Static Q-head inference for Transformers 4.45 Llama-3.1 8B GQA."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .packed_update import resolve_q_head_packed_update

LLAMA31_NUM_Q_HEADS = 32
LLAMA31_NUM_KV_HEADS = 8
LLAMA31_Q_HEADS_PER_KV_HEAD = 4


@dataclass
class Llama31LayerCache:
    packed_key: torch.Tensor
    packed_value: torch.Tensor
    head_lengths: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_k: int
    appended_length: int


@dataclass(frozen=True)
class Llama31VarlenInputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    query_length: int


def _union_llama31_q_head_keep_mask(q_head_keep_mask: torch.Tensor) -> torch.Tensor:
    num_layers, batch_size, _, context_length = q_head_keep_mask.shape
    return q_head_keep_mask.reshape(
        num_layers,
        batch_size,
        LLAMA31_NUM_KV_HEADS,
        LLAMA31_Q_HEADS_PER_KV_HEAD,
        context_length,
    ).any(dim=3)


class Llama31QHeadCache(Cache):
    """Compact per-KV-head cache using each GQA group's Q-head union."""

    def __init__(
        self,
        layers: list[Llama31LayerCache],
        *,
        logical_context_length: int,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.logical_context_length = logical_context_length

    @classmethod
    def from_dynamic_cache(
        cls,
        dense_cache: DynamicCache,
        *,
        keep_mask: torch.Tensor,
        logical_context_length: int,
    ) -> Llama31QHeadCache:
        kv_keep_mask = _union_llama31_q_head_keep_mask(keep_mask)
        layers: list[Llama31LayerCache] = []
        for layer_idx, (key, value) in enumerate(
            zip(dense_cache.key_cache, dense_cache.value_cache)
        ):
            layer_mask = kv_keep_mask[layer_idx].to(key.device)
            head_lengths = layer_mask.sum(dim=-1, dtype=torch.int32)
            flat_lengths = head_lengths.reshape(-1)
            cu_seqlens_k = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device=key.device),
                    torch.cumsum(flat_lengths, dim=0, dtype=torch.int32),
                )
            )
            layers.append(
                Llama31LayerCache(
                    packed_key=key[layer_mask].unsqueeze(1),
                    packed_value=value[layer_mask].unsqueeze(1),
                    head_lengths=head_lengths,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_k=int(head_lengths.max().item()),
                    appended_length=0,
                )
            )
        return cls(layers, logical_context_length=logical_context_length)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        index = 0 if layer_idx is None else layer_idx
        return self.logical_context_length + self.layers[index].appended_length

    def get_max_length(self) -> None:
        return None


def _build_llama31_varlen_inputs(
    query_states: torch.Tensor,
    new_key_states: torch.Tensor,
    new_value_states: torch.Tensor,
    cache: Llama31QHeadCache,
    layer_idx: int,
) -> Llama31VarlenInputs:
    layer = cache.layers[layer_idx]
    query_length = query_states.shape[2]
    head_dim = query_states.shape[-1]
    packed_update = resolve_q_head_packed_update()
    packed_key = packed_update(
        layer.packed_key,
        new_key_states,
        layer.head_lengths,
        layer.cu_seqlens_k,
    )
    packed_value = packed_update(
        layer.packed_value,
        new_value_states,
        layer.head_lengths,
        layer.cu_seqlens_k,
    )
    sequence_offsets = torch.arange(
        LLAMA31_NUM_KV_HEADS + 1,
        dtype=torch.int32,
        device=query_states.device,
    )
    return Llama31VarlenInputs(
        query=(
            query_states.reshape(
                1,
                LLAMA31_NUM_KV_HEADS,
                LLAMA31_Q_HEADS_PER_KV_HEAD,
                query_length,
                head_dim,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(
                LLAMA31_NUM_KV_HEADS * query_length,
                LLAMA31_Q_HEADS_PER_KV_HEAD,
                head_dim,
            )
        ),
        key=packed_key,
        value=packed_value,
        cu_seqlens_q=sequence_offsets * query_length,
        cu_seqlens_k=layer.cu_seqlens_k + sequence_offsets * query_length,
        max_seqlen_q=query_length,
        max_seqlen_k=layer.max_seqlen_k + query_length,
        query_length=query_length,
    )


def _resolve_flash_attn_varlen_func():
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func


def _run_llama31_varlen_attention(
    query_states: torch.Tensor,
    new_key_states: torch.Tensor,
    new_value_states: torch.Tensor,
    cache: Llama31QHeadCache,
    layer_idx: int,
) -> torch.Tensor:
    layer = cache.layers[layer_idx]
    packed = _build_llama31_varlen_inputs(
        query_states,
        new_key_states,
        new_value_states,
        cache,
        layer_idx,
    )
    output = _resolve_flash_attn_varlen_func()(
        packed.query,
        packed.key,
        packed.value,
        packed.cu_seqlens_q,
        packed.cu_seqlens_k,
        packed.max_seqlen_q,
        packed.max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    layer.packed_key = packed.key
    layer.packed_value = packed.value
    layer.head_lengths += packed.query_length
    layer.cu_seqlens_k = packed.cu_seqlens_k
    layer.max_seqlen_k = packed.max_seqlen_k
    layer.appended_length += packed.query_length
    return (
        output.reshape(
            1,
            LLAMA31_NUM_KV_HEADS,
            packed.query_length,
            LLAMA31_Q_HEADS_PER_KV_HEAD,
            -1,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(1, LLAMA31_NUM_Q_HEADS, packed.query_length, -1)
    )


def _llama31_q_head_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Cache | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: object,
):
    if not isinstance(past_key_value, Llama31QHeadCache):
        return self._llama31_q_head_original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    batch_size, query_length, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(
        batch_size, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if query_states.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attention_output = _run_llama31_varlen_attention(
        query_states,
        key_states,
        value_states,
        past_key_value,
        self.layer_idx,
    )
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(batch_size, query_length, -1)
    return self.o_proj(attention_output), None, past_key_value


def install_llama31_q_head_attention(model: LlamaForCausalLM) -> None:
    for layer in model.model.layers:
        attention = layer.self_attn
        attention._llama31_q_head_original_forward = attention.forward
        attention.forward = MethodType(_llama31_q_head_attention_forward, attention)
