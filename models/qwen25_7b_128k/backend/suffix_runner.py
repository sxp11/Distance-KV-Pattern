"""Qwen2.5 gated suffix forward locked to Transformers 4.45.0.

The prompt prefill uses the unmodified Hugging Face Qwen2 model. The suffix
forward mirrors Qwen2's 4.45.0 decoder and replaces only the attention core
with the shared block-gated attention implementation.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import __version__ as transformers_version
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    apply_rotary_pos_emb,
)

from distance_kv_pattern.core.layout import BlockLayout
from distance_kv_pattern.training.pattern_learning.gated_attention import (
    build_prompt_block_gate,
    kv_head_block_gated_attention,
    q_head_block_gated_attention,
)
from distance_kv_pattern.training.pattern_learning.runner_types import (
    GatedBranchesOutput,
    GatedSuffixOutput,
    PromptKVCache,
    causal_value_cross_entropy,
)


SUPPORTED_TRANSFORMERS_VERSION = "4.45.0"


def validate_transformers_runtime(
    detected_version: str | None = None,
) -> None:
    actual_version = (
        transformers_version if detected_version is None else detected_version
    )
    if actual_version != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen2 GatedSuffixRunner mirrors Transformers "
            f"{SUPPORTED_TRANSFORMERS_VERSION}, found {actual_version}"
        )


class GatedSuffixRunner(nn.Module):
    """Run suffix tokens through frozen Qwen2 weights and gated prompt attention."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        *,
        layout: BlockLayout | None = None,
        freeze_model: bool = True,
        gate_granularity: str = "kv_head",
    ) -> None:
        super().__init__()
        validate_transformers_runtime()
        if not isinstance(model, Qwen2ForCausalLM):
            raise TypeError("GatedSuffixRunner supports Qwen2ForCausalLM")
        if gate_granularity not in ("q_head", "kv_head"):
            raise ValueError("gate_granularity must be 'q_head' or 'kv_head'")
        self.model = model
        self.layout = layout if layout is not None else BlockLayout()
        self.gate_granularity = gate_granularity
        self.model.eval()
        if freeze_model:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

    @property
    def device(self) -> torch.device:
        return self.model.model.embed_tokens.weight.device

    @torch.no_grad()
    def dense_prefill(
        self,
        prompt_input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> PromptKVCache:
        if prompt_input_ids.ndim == 1:
            prompt_input_ids = prompt_input_ids.unsqueeze(0)
        if prompt_input_ids.ndim != 2:
            raise ValueError("prompt_input_ids must be rank one or two")
        if prompt_input_ids.shape[0] != 1:
            raise ValueError("the first implementation supports batch size one")
        if prompt_input_ids.shape[1] != self.layout.context_tokens:
            raise ValueError(
                f"prompt length {prompt_input_ids.shape[1]} differs from "
                f"layout length {self.layout.context_tokens}"
            )
        prompt_input_ids = prompt_input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.model.model(
            input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        cache = PromptKVCache.from_huggingface(
            outputs.past_key_values,
            expected_layers=self.num_layers,
        )
        return cache

    def _project_suffix_qkv(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length, _ = hidden_states.size()
        query_states = attention.q_proj(hidden_states).view(
            batch_size,
            query_length,
            attention.num_heads,
            attention.head_dim,
        ).transpose(1, 2)
        key_states = attention.k_proj(hidden_states).view(
            batch_size,
            query_length,
            attention.num_key_value_heads,
            attention.head_dim,
        ).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).view(
            batch_size,
            query_length,
            attention.num_key_value_heads,
            attention.head_dim,
        ).transpose(1, 2)
        return query_states, key_states, value_states

    def _gated_attention_forward(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        prompt_key: torch.Tensor,
        prompt_value: torch.Tensor,
        layer_gate: torch.Tensor,
        binary_layer_gate: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        query_states, key_states, value_states = self._project_suffix_qkv(
            attention,
            hidden_states,
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
        prompt_gate = build_prompt_block_gate(
            layer_gate,
            sink_blocks=self.layout.sink_blocks,
            recent_blocks=self.layout.recent_blocks,
        )
        binary_prompt_gate = (
            None
            if binary_layer_gate is None
            else build_prompt_block_gate(
                binary_layer_gate,
                sink_blocks=self.layout.sink_blocks,
                recent_blocks=self.layout.recent_blocks,
            )
        )
        gated_attention = (
            q_head_block_gated_attention
            if self.gate_granularity == "q_head"
            else kv_head_block_gated_attention
        )
        attention_result = gated_attention(
            query_states,
            prompt_key,
            prompt_value,
            key_states,
            value_states,
            prompt_gate,
            binary_prompt_block_gate=binary_prompt_gate,
            dropout_p=attention.attention_dropout,
            training=attention.training,
            block_size=self.layout.block_size,
        )
        batch_size, _, query_length, _ = attention_result.output.shape
        attention_output = (
            attention_result.output.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, query_length, attention.hidden_size)
        )
        return attention.o_proj(attention_output)

    def _decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        layer_gate: torch.Tensor,
        *,
        binary_layer_gate: torch.Tensor | None,
        decoder_layer: nn.Module,
        prompt_key: torch.Tensor,
        prompt_value: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        hidden_states = self._gated_attention_forward(
            decoder_layer.self_attn,
            hidden_states,
            prompt_key,
            prompt_value,
            layer_gate,
            binary_layer_gate,
            position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        return residual + hidden_states

    def forward_suffix(
        self,
        suffix_input_ids: torch.Tensor,
        *,
        prompt_cache: PromptKVCache,
        learned_gate: torch.Tensor,
        binary_gate: torch.Tensor | None = None,
        checkpoint_layers: bool = False,
    ) -> GatedSuffixOutput:
        if suffix_input_ids.ndim == 1:
            suffix_input_ids = suffix_input_ids.unsqueeze(0)
        suffix_input_ids = suffix_input_ids.to(self.device)
        learned_gate = learned_gate.to(self.device)
        if binary_gate is not None:
            binary_gate = binary_gate.to(self.device)
        hidden_states = self.model.model.embed_tokens(suffix_input_ids)
        query_length = suffix_input_ids.shape[1]
        position_ids = torch.arange(
            prompt_cache.sequence_length,
            prompt_cache.sequence_length + query_length,
            device=self.device,
        ).unsqueeze(0)
        position_embeddings = self.model.model.rotary_emb(
            hidden_states,
            position_ids,
        )

        for layer_index, decoder_layer in enumerate(self.model.model.layers):
            prompt_key, prompt_value = prompt_cache.layer(layer_index)
            layer_function = partial(
                self._decoder_layer_forward,
                binary_layer_gate=(
                    None if binary_gate is None else binary_gate[layer_index]
                ),
                decoder_layer=decoder_layer,
                prompt_key=prompt_key,
                prompt_value=prompt_value,
                position_embeddings=position_embeddings,
            )
            layer_gate = learned_gate[layer_index]
            if checkpoint_layers:
                hidden_states = checkpoint(
                    layer_function,
                    hidden_states,
                    layer_gate,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer_function(hidden_states, layer_gate)

        hidden_states = self.model.model.norm(hidden_states)
        return GatedSuffixOutput(
            logits=self.model.lm_head(hidden_states).float(),
            hidden_states=hidden_states,
        )

    def forward_branches(
        self,
        suffix_input_ids: Sequence[torch.Tensor],
        suffix_labels: Sequence[torch.Tensor],
        *,
        prompt_cache: PromptKVCache,
        learned_gate: torch.Tensor,
        checkpoint_layers: bool = False,
    ) -> GatedBranchesOutput:
        branch_logits: list[torch.Tensor] = []
        branch_losses: list[torch.Tensor] = []
        for input_ids, labels in zip(
            suffix_input_ids,
            suffix_labels,
            strict=True,
        ):
            output = self.forward_suffix(
                input_ids,
                prompt_cache=prompt_cache,
                learned_gate=learned_gate,
                checkpoint_layers=checkpoint_layers,
            )
            branch_logits.append(output.logits)
            branch_losses.append(
                causal_value_cross_entropy(output.logits, labels)
            )
        stacked_losses = torch.stack(branch_losses)
        return GatedBranchesOutput(
            logits=tuple(branch_logits),
            branch_losses=stacked_losses,
            task_loss=stacked_losses.mean(),
        )


class KVHeadGatedSuffixRunner(GatedSuffixRunner):
    def __init__(
        self,
        model: Qwen2ForCausalLM,
        *,
        layout: BlockLayout | None = None,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(
            model,
            layout=layout,
            freeze_model=freeze_model,
            gate_granularity="kv_head",
        )


class QHeadGatedSuffixRunner(GatedSuffixRunner):
    def __init__(
        self,
        model: Qwen2ForCausalLM,
        *,
        layout: BlockLayout | None = None,
        freeze_model: bool = True,
    ) -> None:
        super().__init__(
            model,
            layout=layout,
            freeze_model=freeze_model,
            gate_granularity="q_head",
        )
