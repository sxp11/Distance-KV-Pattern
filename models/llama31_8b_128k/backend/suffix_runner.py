"""Llama-3.1 gated suffix forward locked to Transformers 4.45.0.

The dense prompt is prefetched by the unmodified Hugging Face Llama model.
Only the query/answer suffix forward replaces the attention core so prompt KV
blocks can be weighted by layer-, Q/KV-head- and distance-specific gates.
Everything surrounding attention follows modeling_llama.py from 4.45.0.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import __version__ as transformers_version
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
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
    """Reject an unverified Transformers runtime before model work begins."""

    actual_version = (
        transformers_version if detected_version is None else detected_version
    )
    if actual_version != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "GatedSuffixRunner mirrors Transformers "
            f"{SUPPORTED_TRANSFORMERS_VERSION}, found {actual_version}"
        )


class GatedSuffixRunner(nn.Module):
    """Run suffix tokens through frozen Llama weights and gated prompt attention."""

    def __init__(
        self,
        model: LlamaForCausalLM,
        *,
        layout: BlockLayout | None = None,
        freeze_model: bool = True,
        gate_granularity: str = "kv_head",
    ) -> None:
        super().__init__()
        validate_transformers_runtime()
        if not isinstance(model, LlamaForCausalLM):
            raise TypeError("GatedSuffixRunner currently supports LlamaForCausalLM")
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
        if config.num_hidden_layers != len(self.model.model.layers):
            raise ValueError("Llama config and decoder layer count disagree")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Q-head count must be divisible by KV-head count")
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden size must be divisible by Q-head count")
        self.num_layers = config.num_hidden_layers
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        configured_head_dim = getattr(config, "head_dim", None)
        self.head_dim = int(
            configured_head_dim
            if configured_head_dim is not None
            else config.hidden_size // config.num_attention_heads
        )
        layer_head_dims = {
            int(layer.self_attn.head_dim) for layer in self.model.model.layers
        }
        if layer_head_dims != {self.head_dim}:
            raise ValueError(
                "Llama config and attention layer head dimensions disagree"
            )

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
        """Use the original model attention implementation to build prompt KV."""

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
        if cache.sequence_length != prompt_input_ids.shape[1]:
            raise AssertionError("prompt cache length differs from prompt input")
        if cache.num_kv_heads != self.num_kv_heads:
            raise AssertionError("prompt cache KV-head count differs from model")
        if cache.head_dim != self.head_dim:
            raise AssertionError("prompt cache head dimension differs from model")
        return cache

    def _project_suffix_qkv(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Copy the Q/K/V projection branch from LlamaAttention 4.45.0."""

        batch_size, query_length, _ = hidden_states.size()
        if attention.config.pretraining_tp > 1:
            key_value_slicing = (
                attention.num_key_value_heads * attention.head_dim
            ) // attention.config.pretraining_tp
            query_slices = attention.q_proj.weight.split(
                (attention.num_heads * attention.head_dim)
                // attention.config.pretraining_tp,
                dim=0,
            )
            key_slices = attention.k_proj.weight.split(
                key_value_slicing,
                dim=0,
            )
            value_slices = attention.v_proj.weight.split(
                key_value_slicing,
                dim=0,
            )
            query_states = torch.cat(
                [
                    F.linear(hidden_states, query_slices[index])
                    for index in range(attention.config.pretraining_tp)
                ],
                dim=-1,
            )
            key_states = torch.cat(
                [
                    F.linear(hidden_states, key_slices[index])
                    for index in range(attention.config.pretraining_tp)
                ],
                dim=-1,
            )
            value_states = torch.cat(
                [
                    F.linear(hidden_states, value_slices[index])
                    for index in range(attention.config.pretraining_tp)
                ],
                dim=-1,
            )
        else:
            query_states = attention.q_proj(hidden_states)
            key_states = attention.k_proj(hidden_states)
            value_states = attention.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size,
            query_length,
            attention.num_heads,
            attention.head_dim,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size,
            query_length,
            attention.num_key_value_heads,
            attention.head_dim,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size,
            query_length,
            attention.num_key_value_heads,
            attention.head_dim,
        ).transpose(1, 2)
        return query_states, key_states, value_states

    def _output_projection(
        self,
        attention: nn.Module,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        """Copy the transpose/reshape/O-projection branch from 4.45.0."""

        batch_size, _, query_length, _ = attention_output.shape
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(
            batch_size,
            query_length,
            -1,
        )
        if attention.config.pretraining_tp > 1:
            split_output = attention_output.split(
                attention.hidden_size // attention.config.pretraining_tp,
                dim=2,
            )
            output_slices = attention.o_proj.weight.split(
                attention.hidden_size // attention.config.pretraining_tp,
                dim=1,
            )
            return sum(
                F.linear(split_output[index], output_slices[index])
                for index in range(attention.config.pretraining_tp)
            )
        return attention.o_proj(attention_output)

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
        attention_kwargs = {
            "dropout_p": attention.attention_dropout,
            "training": attention.training,
            "block_size": self.layout.block_size,
        }
        attention_result = gated_attention(
            query_states,
            prompt_key,
            prompt_value,
            key_states,
            value_states,
            prompt_gate,
            binary_prompt_block_gate=binary_prompt_gate,
            **attention_kwargs,
        )
        return self._output_projection(attention, attention_result.output)

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
        """Copy LlamaDecoderLayer.forward, replacing only self-attention."""

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
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_suffix(
        self,
        suffix_input_ids: torch.Tensor,
        *,
        prompt_cache: PromptKVCache,
        learned_gate: torch.Tensor,
        binary_gate: torch.Tensor | None = None,
        checkpoint_layers: bool = False,
    ) -> GatedSuffixOutput:
        """Run one complete teacher-forced query/answer suffix."""

        if suffix_input_ids.ndim == 1:
            suffix_input_ids = suffix_input_ids.unsqueeze(0)
        if suffix_input_ids.ndim != 2 or suffix_input_ids.shape[0] != 1:
            raise ValueError("suffix_input_ids must have batch size one")
        if suffix_input_ids.shape[1] <= 0:
            raise ValueError("suffix_input_ids cannot be empty")
        if suffix_input_ids.shape[1] > self.layout.suffix_capacity_tokens:
            raise ValueError("suffix exceeds the configured token capacity")
        num_gate_heads = (
            self.num_q_heads
            if self.gate_granularity == "q_head"
            else self.num_kv_heads
        )
        expected_gate_shape = (
            self.num_layers,
            num_gate_heads,
            self.layout.num_learnable_blocks,
        )
        if tuple(learned_gate.shape) != expected_gate_shape:
            raise ValueError(
                f"learned_gate has shape {tuple(learned_gate.shape)}, "
                f"expected {expected_gate_shape}"
            )
        if prompt_cache.sequence_length != self.layout.context_tokens:
            raise ValueError("prompt cache length differs from active layout")
        if prompt_cache.batch_size != 1:
            raise ValueError("the first implementation supports batch size one")
        if prompt_cache.num_kv_heads != self.num_kv_heads:
            raise ValueError("prompt cache KV-head count differs from model")
        if prompt_cache.head_dim != self.head_dim:
            raise ValueError("prompt cache head dimension differs from model")

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
                hidden_states = layer_function(
                    hidden_states,
                    layer_gate,
                )

        hidden_states = self.model.model.norm(hidden_states)
        if self.model.config.pretraining_tp > 1:
            head_slices = self.model.lm_head.weight.split(
                self.model.vocab_size // self.model.config.pretraining_tp,
                dim=0,
            )
            logits = torch.cat(
                [
                    F.linear(hidden_states, head_slices[index])
                    for index in range(self.model.config.pretraining_tp)
                ],
                dim=-1,
            )
        else:
            logits = self.model.lm_head(hidden_states)
        return GatedSuffixOutput(
            logits=logits.float(),
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
        """Run independent queries sequentially with one shared sampled gate."""

        if len(suffix_input_ids) != len(suffix_labels):
            raise ValueError("suffix inputs and labels must have equal length")
        if not suffix_input_ids:
            raise ValueError("at least one suffix branch is required")
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
    """Suffix runner with one prompt-block gate per KV head."""

    def __init__(
        self,
        model: LlamaForCausalLM,
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
    """Suffix runner with one prompt-block gate per Q head."""

    def __init__(
        self,
        model: LlamaForCausalLM,
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
