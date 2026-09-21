"""Official PyTorch DDP helpers for Q-head pattern training.

The frozen language model is intentionally not wrapped in DDP. Only the
Hard Concrete module owns trainable parameters, so DDP synchronizes that
small module while each rank processes a disjoint subset of prompt contexts.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .hard_concrete import QHeadHardConcreteGates
from .runner_types import (
    PromptKVCache,
    QHeadSuffixRunner,
    causal_value_cross_entropy,
)
from .training_step import BranchwiseBackwardResult


@dataclass(frozen=True, slots=True)
class ContextShard:
    """The exact global context offsets assigned to one rank."""

    rank: int
    world_size: int
    global_offsets: tuple[int, ...]
    slot_offsets: tuple[int | None, ...]


def _effective_q_head_gates(
    relaxed_gate: torch.Tensor,
    binary_gate: torch.Tensor | None,
    *,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Expand relaxed and binary shared-KV unions back to their Q heads."""

    group_size = relaxed_gate.shape[-2] // num_kv_heads
    if group_size == 1:
        return relaxed_gate, binary_gate

    grouped_shape = (
        *relaxed_gate.shape[:-2],
        num_kv_heads,
        group_size,
        relaxed_gate.shape[-1],
    )
    grouped_relaxed = relaxed_gate.reshape(grouped_shape)
    relaxed_union = 1.0 - torch.prod(1.0 - grouped_relaxed, dim=-2)
    effective_relaxed = relaxed_union.repeat_interleave(group_size, dim=-2)
    if binary_gate is None:
        return effective_relaxed, None

    grouped_binary = binary_gate.reshape(grouped_shape)
    binary_union = grouped_binary.any(dim=-2)
    effective_binary = binary_union.repeat_interleave(group_size, dim=-2)
    return effective_relaxed, effective_binary


def context_offsets_for_rank(
    num_contexts: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, ...]:
    """Round-robin contexts without padding, duplication, or dropping."""

    if num_contexts <= 0:
        raise ValueError("num_contexts must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must lie inside [0, world_size)")
    if world_size > num_contexts:
        raise ValueError("world_size cannot exceed num_contexts")
    return tuple(range(rank, num_contexts, world_size))


def build_context_shards(
    num_contexts: int,
    *,
    world_size: int,
) -> tuple[ContextShard, ...]:
    """Build a complete partition with equal DDP synchronization slots.

    ``global_offsets`` contains real contexts only.  ``slot_offsets`` pads
    shorter round-robin shards with ``None`` so every rank executes the same
    number of DDP forward/backward calls.  The training loop treats a padded
    slot as a zero-loss dummy backward, preserving collective ordering without
    duplicating any real context.
    """

    real_offsets = tuple(
        context_offsets_for_rank(
            num_contexts,
            rank=rank,
            world_size=world_size,
        )
        for rank in range(world_size)
    )
    slots_per_rank = max(map(len, real_offsets))
    shards = tuple(
        ContextShard(
            rank=rank,
            world_size=world_size,
            global_offsets=offsets,
            slot_offsets=offsets + (None,) * (slots_per_rank - len(offsets)),
        )
        for rank, offsets in enumerate(real_offsets)
    )
    flattened = [offset for shard in shards for offset in shard.global_offsets]
    if len(flattened) != num_contexts:
        raise AssertionError("distributed partition changed the context count")
    if sorted(flattened) != list(range(num_contexts)):
        raise AssertionError("distributed partition duplicated or dropped contexts")
    if len({len(shard.slot_offsets) for shard in shards}) != 1:
        raise AssertionError("DDP shards do not have equal synchronization slots")
    if any(
        tuple(offset for offset in shard.slot_offsets if offset is not None)
        != shard.global_offsets
        for shard in shards
    ):
        raise AssertionError("padded DDP slots changed real context order")
    return shards


def ddp_task_loss_scale(world_size: int) -> float:
    """Compensate for DDP's gradient averaging of globally weighted losses."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return float(world_size)


def unwrap_q_head_gates(module: nn.Module) -> QHeadHardConcreteGates:
    """Return the underlying gate module from plain or DDP ownership."""

    candidate = module.module if isinstance(module, DistributedDataParallel) else module
    if not isinstance(candidate, QHeadHardConcreteGates):
        raise TypeError("module must own QHeadHardConcreteGates")
    return candidate


def backward_distributed_q_head_branches(
    runner: QHeadSuffixRunner,
    prompt_cache: PromptKVCache,
    ddp_gates: DistributedDataParallel,
    suffix_input_ids: Sequence[torch.Tensor],
    suffix_labels: Sequence[torch.Tensor],
    *,
    branch_weights: Sequence[float] | torch.Tensor,
    uniform: torch.Tensor,
    binary_gate: torch.Tensor | None = None,
    synchronize_gradients: bool,
    final_regularization_loss: torch.Tensor | None = None,
    checkpoint_layers: bool = True,
) -> BranchwiseBackwardResult:
    """Accumulate one context under official DDP synchronization semantics.

    Every non-final context runs entirely under ``DDP.no_sync``. For the final
    local context, only its final query leaves ``no_sync``; that first synced
    backward reduces all task gradients accumulated on the rank. Query losses
    are multiplied by ``world_size`` because their branch weights are already
    normalized globally while DDP averages gradients across ranks.
    """

    gates = unwrap_q_head_gates(ddp_gates)
    if runner.gate_granularity != "q_head":
        raise ValueError("runner must use Q-head gate semantics")
    if tuple(gates.shape) != (
        runner.num_layers,
        runner.num_q_heads,
        runner.layout.num_learnable_blocks,
    ):
        raise ValueError("Q-head gate shape differs from the suffix runner")
    if len(suffix_input_ids) != len(suffix_labels):
        raise ValueError("suffix inputs and labels must have equal length")
    num_branches = len(suffix_input_ids)
    if num_branches == 0:
        raise ValueError("at least one suffix branch is required")

    weights = torch.as_tensor(branch_weights, dtype=torch.float32, device="cpu")
    if weights.shape != (num_branches,):
        raise ValueError(
            f"branch_weights has shape {tuple(weights.shape)}, "
            f"expected {(num_branches,)}"
        )
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("branch_weights must be finite and non-negative")
    if not torch.any(weights > 0):
        raise ValueError("at least one branch weight must be positive")

    uniform = torch.as_tensor(
        uniform,
        dtype=torch.float32,
        device=gates.log_alpha.device,
    ).detach()
    if tuple(uniform.shape) != gates.shape:
        raise ValueError(
            f"uniform has shape {tuple(uniform.shape)}, expected {gates.shape}"
        )

    if binary_gate is not None:
        binary_gate = torch.as_tensor(
            binary_gate,
            device=gates.log_alpha.device,
        ).detach()
        if tuple(binary_gate.shape) != gates.shape:
            raise ValueError(
                f"binary_gate has shape {tuple(binary_gate.shape)}, "
                f"expected {gates.shape}"
            )

    task_scale = ddp_task_loss_scale(dist.get_world_size(ddp_gates.process_group))
    detached_losses: list[torch.Tensor] = []
    weighted_task_loss = 0.0
    for branch_index, (input_ids, labels, weight) in enumerate(
        zip(suffix_input_ids, suffix_labels, weights, strict=True)
    ):
        sync_this_branch = synchronize_gradients and branch_index == num_branches - 1
        sync_context = nullcontext() if sync_this_branch else ddp_gates.no_sync()
        with sync_context:
            relaxed_gate = ddp_gates(uniform)
            learned_gate, effective_binary_gate = _effective_q_head_gates(
                relaxed_gate,
                binary_gate,
                num_kv_heads=runner.num_kv_heads,
            )
            output = runner.forward_suffix(
                input_ids,
                prompt_cache=prompt_cache,
                learned_gate=learned_gate,
                binary_gate=effective_binary_gate,
                checkpoint_layers=checkpoint_layers,
            )
            branch_loss = causal_value_cross_entropy(output.logits, labels)
            backward_loss = branch_loss * float(weight) * task_scale
            if sync_this_branch and final_regularization_loss is not None:
                backward_loss = backward_loss + final_regularization_loss
            backward_loss.backward()

        detached = branch_loss.detach().float().cpu()
        detached_losses.append(detached)
        weighted_task_loss += float(weight) * float(detached)
        del (
            branch_loss,
            backward_loss,
            output,
            learned_gate,
            effective_binary_gate,
            relaxed_gate,
        )

    return BranchwiseBackwardResult(
        branch_losses=torch.stack(detached_losses),
        branch_weights=weights.clone(),
        weighted_task_loss=weighted_task_loss,
    )
