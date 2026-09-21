"""Model-independent cache, output and runner contracts for Q-head training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from ...core.layout import BlockLayout


@dataclass(frozen=True, slots=True)
class PromptKVCache:
    """Detached, read-only prompt K/V tensors in legacy per-layer form."""

    key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    sequence_length: int
    batch_size: int
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_huggingface(
        cls,
        past_key_values: Cache | Sequence[Sequence[torch.Tensor]],
        *,
        expected_layers: int,
    ) -> "PromptKVCache":
        if isinstance(past_key_values, Cache):
            legacy = past_key_values.to_legacy_cache()
        else:
            legacy = tuple(past_key_values)
        if len(legacy) != expected_layers:
            raise ValueError(
                f"prompt cache has {len(legacy)} layers, expected {expected_layers}"
            )

        detached: list[tuple[torch.Tensor, torch.Tensor]] = []
        reference_shape: tuple[int, int, int, int] | None = None
        for layer_index, layer_cache in enumerate(legacy):
            if len(layer_cache) < 2:
                raise ValueError(f"cache layer {layer_index} has no K/V pair")
            key, value = layer_cache[:2]
            if key.ndim != 4 or value.ndim != 4:
                raise ValueError("prompt cache K/V tensors must be rank four")
            if key.shape != value.shape:
                raise ValueError("prompt cache key and value shapes must match")
            current_shape = tuple(key.shape)
            if reference_shape is None:
                reference_shape = current_shape
            elif current_shape != reference_shape:
                raise ValueError("all prompt cache layers must share one shape")
            detached.append((key.detach(), value.detach()))

        if reference_shape is None:
            raise ValueError("prompt cache is empty")
        batch_size, num_kv_heads, sequence_length, head_dim = reference_shape
        return cls(
            key_values=tuple(detached),
            sequence_length=sequence_length,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    def layer(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_values[layer_index]


@dataclass(frozen=True, slots=True)
class GatedSuffixOutput:
    """Logits and final hidden states for one teacher-forced suffix."""

    logits: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(frozen=True, slots=True)
class GatedBranchesOutput:
    """Independent suffix branches sharing one prompt cache and one gate."""

    logits: tuple[torch.Tensor, ...]
    branch_losses: torch.Tensor
    task_loss: torch.Tensor


def causal_value_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Apply the causal-language-model shift while ignoring unsupervised labels."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if labels.ndim != 2 or labels.shape != logits.shape[:2]:
        raise ValueError("labels must match the logits batch and sequence axes")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(logits.device)
    supervised = shift_labels.ne(-100)
    if not torch.any(supervised):
        raise ValueError("suffix labels contain no supervised target tokens")
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class QHeadSuffixRunner(Protocol):
    """Structural interface consumed by model-independent training code."""

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    layout: BlockLayout
    gate_granularity: str

    def dense_prefill(
        self,
        prompt_input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> PromptKVCache: ...

    def forward_suffix(
        self,
        suffix_input_ids: torch.Tensor,
        *,
        prompt_cache: PromptKVCache,
        learned_gate: torch.Tensor,
        binary_gate: torch.Tensor | None = None,
        checkpoint_layers: bool = False,
    ) -> GatedSuffixOutput: ...
