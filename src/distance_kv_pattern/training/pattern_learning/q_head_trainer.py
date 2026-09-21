"""Pure planning utilities for Q-head Hard Concrete training.

The GPU training entry point deliberately keeps data ordering, step-level loss
normalization, Hard Concrete seeds, and L0 scheduling in this CPU-testable
module. The formal trainer always passes the complete manifest length as
``contexts_per_optimizer_step``; the generic context-stream helper still
accepts smaller windows because the gradient-audit tests use it to compare
partial prefixes with a full-coverage step. The previous A100 experiment's
learning-rate and L0 values are not safe defaults, so optimization values must
be selected explicitly for each run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Mapping, Sequence

import torch

from ...core.randomness import derive_seed
from .training_step import distance_balanced_branch_weights


FORMAL_NEEDLE_COUNT = 4
FORMAL_CONTEXTS_PER_OPTIMIZER_STEP = 254
FORMAL_LEARNING_RATE: float | None = None
FORMAL_L0_LAMBDA: float | None = None
FORMAL_MAX_OPTIMIZER_STEPS: int | None = None
FORMAL_L0_START_STEP = 0
FORMAL_L0_RAMP_STEPS = 0
FORMAL_MILESTONE_STEPS = (1, 5, 10, 20, 30)


def exact_topk_mask(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Return an exact-cardinality stable global Top-K training mask."""

    values = torch.as_tensor(scores, dtype=torch.float32, device="cpu")
    if values.ndim != 3 or values.numel() == 0:
        raise ValueError("scores must be a non-empty [layers, heads, distances] tensor")
    if not torch.isfinite(values).all():
        raise ValueError("scores contain NaN or infinity")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must lie inside (0, 1]")

    count = max(1, min(values.numel(), round(values.numel() * keep_ratio)))
    order = torch.argsort(values.flatten(), descending=True, stable=True)
    mask = torch.zeros(values.numel(), dtype=torch.bool)
    mask[order[:count]] = True
    return mask.view_as(values)


@dataclass(frozen=True, slots=True)
class QHeadTrainingHyperparameters:
    """Structural defaults plus explicitly selected optimization values."""

    contexts_per_optimizer_step: int | None = FORMAL_CONTEXTS_PER_OPTIMIZER_STEP
    learning_rate: float | None = FORMAL_LEARNING_RATE
    l0_lambda: float | None = FORMAL_L0_LAMBDA
    max_optimizer_steps: int | None = FORMAL_MAX_OPTIMIZER_STEPS
    l0_start_step: int = FORMAL_L0_START_STEP
    l0_ramp_steps: int = FORMAL_L0_RAMP_STEPS
    gradient_clip_norm: float = 1.0

    def unresolved(self) -> tuple[str, ...]:
        names = (
            "contexts_per_optimizer_step",
            "learning_rate",
            "l0_lambda",
            "max_optimizer_steps",
        )
        return tuple(name for name in names if getattr(self, name) is None)

    def validate(self, *, num_contexts: int, require_resolved: bool) -> None:
        if num_contexts <= 0:
            raise ValueError("num_contexts must be positive")
        if self.l0_start_step < 0:
            raise ValueError("l0_start_step cannot be negative")
        if self.l0_ramp_steps < 0:
            raise ValueError("l0_ramp_steps cannot be negative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")

        if self.contexts_per_optimizer_step is not None:
            if not 1 <= self.contexts_per_optimizer_step <= num_contexts:
                raise ValueError(
                    "contexts_per_optimizer_step must lie inside "
                    f"[1, {num_contexts}]"
                )
        if self.learning_rate is not None:
            if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
                raise ValueError("learning_rate must be finite and positive")
        if self.l0_lambda is not None:
            if not math.isfinite(self.l0_lambda) or self.l0_lambda < 0:
                raise ValueError("l0_lambda must be finite and non-negative")
        if self.max_optimizer_steps is not None and self.max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive")

        if require_resolved and self.unresolved():
            raise ValueError(
                "real training requires explicit values for: "
                + ", ".join(self.unresolved())
            )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "contexts_per_optimizer_step": self.contexts_per_optimizer_step,
            "learning_rate": self.learning_rate,
            "l0_lambda": self.l0_lambda,
            "max_optimizer_steps": self.max_optimizer_steps,
            "l0_start_step": self.l0_start_step,
            "l0_ramp_steps": self.l0_ramp_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
        }


def resolve_optimizer_step_run_limit(
    *,
    completed_optimizer_steps: int,
    max_optimizer_steps: int,
    optimizer_steps_this_run: int | None,
) -> int:
    """Return the absolute optimizer-step limit for one process invocation."""

    if max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    if not 0 <= completed_optimizer_steps <= max_optimizer_steps:
        raise ValueError("completed_optimizer_steps lies outside the training plan")
    if optimizer_steps_this_run is None:
        return max_optimizer_steps
    if optimizer_steps_this_run <= 0:
        raise ValueError("optimizer_steps_this_run must be positive")
    return min(
        max_optimizer_steps,
        completed_optimizer_steps + optimizer_steps_this_run,
    )


def resolve_milestone_steps(
    requested: Sequence[int],
    *,
    max_optimizer_steps: int,
) -> tuple[int, ...]:
    """Validate and canonicalize absolute completed-step milestones."""

    if max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    milestones = tuple(sorted({int(step) for step in requested}))
    if not milestones:
        raise ValueError("at least one milestone step is required")
    if milestones[0] <= 0:
        raise ValueError("milestone steps must be positive")
    if milestones[-1] > max_optimizer_steps:
        raise ValueError(
            "milestone step exceeds max_optimizer_steps: "
            f"{milestones[-1]} > {max_optimizer_steps}"
        )
    return milestones


def validate_continuation_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    target_signature: Mapping[str, Any],
    expected_completed_optimizer_steps: int | None = None,
    allow_manifest_change: bool = False,
) -> dict[str, Any]:
    """Validate a one-time import into a longer, otherwise identical plan.

    A continuation may extend ``max_optimizer_steps`` and may start an
    explicitly recorded polarization phase.  Target-only provenance and
    polarization fields are excluded from the protected comparison because an
    older source checkpoint cannot contain its own import metadata or the new
    phase configuration.  Absolute input locations are informational: model
    and retrieval identities are compared by their logical names, while the
    manifest collection is protected by ``manifest_id`` unless the target plan
    explicitly records that this continuation changes training data.
    """

    if int(checkpoint.get("schema_version", -1)) != 3:
        raise ValueError("unsupported continuation checkpoint schema")
    source_signature = checkpoint.get("signature")
    if not isinstance(source_signature, Mapping):
        raise ValueError("continuation checkpoint has no training signature")

    completed = checkpoint.get("completed_optimizer_steps")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise ValueError("continuation checkpoint has invalid completed step")
    if completed <= 0:
        raise ValueError("continuation checkpoint must contain completed training")
    if (
        expected_completed_optimizer_steps is not None
        and completed != expected_completed_optimizer_steps
    ):
        raise ValueError(
            "continuation checkpoint completed step differs from the expected "
            f"step: {completed} != {expected_completed_optimizer_steps}"
        )

    source_max = source_signature.get("max_optimizer_steps")
    target_max = target_signature.get("max_optimizer_steps")
    if isinstance(source_max, bool) or not isinstance(source_max, int):
        raise ValueError("continuation source has invalid max_optimizer_steps")
    if isinstance(target_max, bool) or not isinstance(target_max, int):
        raise ValueError("continuation target has invalid max_optimizer_steps")
    if completed > source_max:
        raise ValueError("continuation source completed step exceeds its plan")
    if completed >= target_max:
        raise ValueError("continuation target must extend beyond the source step")

    ignored = {
        "max_optimizer_steps",
        "continuation_checkpoint",
        "continuation_expected_step",
        "polarization_lambda",
        "polarization_start_step",
        "polarization_ramp_steps",
        "polarization_candidate_audit_only",
        "polarization_candidate_audit_lambdas",
        "gate_forward_mode",
        "st_topk_keep_ratios",
        "st_topk_assignment_policy",
        "manifests",
        "manifest_group_schedule",
        "allow_manifest_change_on_continuation",
    }
    if allow_manifest_change:
        ignored.add("manifest_id")
    source_comparable = {
        key: value for key, value in source_signature.items() if key not in ignored
    }
    target_comparable = {
        key: value for key, value in target_signature.items() if key not in ignored
    }
    for comparable in (source_comparable, target_comparable):
        if "model_name_or_path" in comparable:
            comparable["model_name_or_path"] = PurePath(
                comparable["model_name_or_path"]
            ).name
        if "retrieval_checkpoint" in comparable:
            retrieval_path = PurePath(comparable["retrieval_checkpoint"])
            comparable["retrieval_checkpoint"] = tuple(retrieval_path.parts[-2:])
    if source_comparable != target_comparable:
        differing = sorted(
            key
            for key in source_comparable.keys() | target_comparable.keys()
            if source_comparable.get(key) != target_comparable.get(key)
        )
        raise ValueError(
            "continuation checkpoint signature differs in protected fields: "
            + ", ".join(differing)
        )

    required_payload = ("log_alpha", "optimizer_state")
    missing = [key for key in required_payload if key not in checkpoint]
    if missing:
        raise ValueError(
            "continuation checkpoint is missing payload fields: "
            + ", ".join(missing)
        )
    return {
        "source_completed_optimizer_steps": completed,
        "source_max_optimizer_steps": source_max,
        "target_max_optimizer_steps": target_max,
        "manifest_change_allowed": allow_manifest_change,
        "source_manifest_id": source_signature.get("manifest_id"),
        "target_manifest_id": target_signature.get("manifest_id"),
    }


@dataclass(frozen=True, slots=True)
class ContextOccurrence:
    """One deterministic occurrence in the infinite manifest stream."""

    dataset_index: int
    epoch: int
    position_in_epoch: int
    position_in_step: int


def formal_manifest_id(plans: Sequence[Any]) -> str:
    """Validate one homogeneous formal N=4 manifest and return its identity."""

    if not plans:
        raise ValueError("manifest contains no plans")
    first = plans[0]
    identity = (
        first.split,
        first.value_type,
        first.needle_count,
        first.template_id,
        first.coverage_round,
    )
    for plan in plans:
        candidate = (
            plan.split,
            plan.value_type,
            plan.needle_count,
            plan.template_id,
            plan.coverage_round,
        )
        if candidate != identity:
            raise ValueError("formal training requires one homogeneous manifest")
    if first.needle_count != FORMAL_NEEDLE_COUNT:
        raise ValueError(
            "formal Q-head training requires "
            f"needle_count={FORMAL_NEEDLE_COUNT}, got {first.needle_count}"
        )
    return (
        f"{first.split}:{first.value_type}:n{first.needle_count}:"
        f"{first.template_id}:r{first.coverage_round:03d}"
    )


def formal_manifest_collection_id(
    manifests: Sequence[Sequence[Any]],
) -> str:
    """Return a deterministic identity for one or more formal manifests."""

    if not manifests:
        raise ValueError("manifest collection contains no manifests")
    identifiers = tuple(formal_manifest_id(plans) for plans in manifests)
    return "collection:" + "|".join(identifiers)


def planned_gate_indices(plans: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    """Return distance indices in independent-query branch order."""

    groups: list[tuple[int, ...]] = []
    for plan in plans:
        order = tuple(int(index) for index in plan.query_order)
        if sorted(order) != list(range(len(plan.needles))):
            raise ValueError("query_order must be a permutation of needle indices")
        group = tuple(
            int(plan.needles[needle_index].gate_index)
            for needle_index in order
        )
        if len(group) != FORMAL_NEEDLE_COUNT:
            raise ValueError(
                "every formal context must contain exactly four needles"
            )
        groups.append(group)
    return tuple(groups)


def _epoch_order(
    num_contexts: int,
    *,
    epoch: int,
    master_seed: int,
    shuffle: bool,
) -> tuple[int, ...]:
    if not shuffle:
        return tuple(range(num_contexts))
    seed = derive_seed(master_seed, "q_head_trainer_context_order", epoch)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(
        int(index)
        for index in torch.randperm(num_contexts, generator=generator).tolist()
    )


def context_occurrences_for_step(
    *,
    num_contexts: int,
    contexts_per_optimizer_step: int,
    optimizer_step: int,
    master_seed: int,
    shuffle: bool,
) -> tuple[ContextOccurrence, ...]:
    """Select one fixed-size step from a reproducible infinite context stream."""

    if num_contexts <= 0:
        raise ValueError("num_contexts must be positive")
    if not 1 <= contexts_per_optimizer_step <= num_contexts:
        raise ValueError(
            "contexts_per_optimizer_step must lie inside "
            f"[1, {num_contexts}]"
        )
    if optimizer_step < 0:
        raise ValueError("optimizer_step cannot be negative")

    start = optimizer_step * contexts_per_optimizer_step
    orders: dict[int, tuple[int, ...]] = {}
    occurrences: list[ContextOccurrence] = []
    for position_in_step in range(contexts_per_optimizer_step):
        stream_position = start + position_in_step
        epoch, position_in_epoch = divmod(stream_position, num_contexts)
        if epoch not in orders:
            orders[epoch] = _epoch_order(
                num_contexts,
                epoch=epoch,
                master_seed=master_seed,
                shuffle=shuffle,
            )
        occurrences.append(
            ContextOccurrence(
                dataset_index=orders[epoch][position_in_epoch],
                epoch=epoch,
                position_in_epoch=position_in_epoch,
                position_in_step=position_in_step,
            )
        )
    return tuple(occurrences)


def normalized_step_branch_weights(
    context_gate_indices: Sequence[Sequence[int]],
    *,
    coverage_count: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Normalize full-manifest distance weights over one optimizer step.

    This matches the prefix normalization used by the gradient audit. The
    returned branch weights sum to one even when the selected step contains
    only a subset of the complete manifest.
    """

    if not context_gate_indices:
        raise ValueError("an optimizer step must contain at least one context")
    base = tuple(
        distance_balanced_branch_weights(
            indices,
            coverage_count=coverage_count,
        )
        for indices in context_gate_indices
    )
    total = sum(float(weights.sum().item()) for weights in base)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("optimizer-step branch weights have invalid total")
    normalized = tuple(weights / total for weights in base)
    normalized_total = sum(float(weights.sum().item()) for weights in normalized)
    if not math.isclose(normalized_total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(
            f"normalized optimizer-step weights sum to {normalized_total}"
        )
    return normalized


def q_head_training_gate_sample_seed(
    master_seed: int,
    *,
    manifest_id: str,
    optimizer_step: int,
    position_in_step: int,
    sample_id: str,
) -> int:
    """Return a reproducible but update-specific Hard Concrete noise seed."""

    if not manifest_id:
        raise ValueError("manifest_id cannot be empty")
    if optimizer_step < 0 or position_in_step < 0:
        raise ValueError("optimizer step and context position cannot be negative")
    if not sample_id:
        raise ValueError("sample_id cannot be empty")
    return derive_seed(
        master_seed,
        "q_head_formal_training",
        manifest_id,
        optimizer_step,
        position_in_step,
        sample_id,
    )


def st_topk_keep_ratio_for_context(
    keep_ratios: Sequence[float],
    *,
    optimizer_step: int,
    position_in_step: int,
) -> float:
    """Assign fixed budgets evenly and rotate them across optimizer steps."""

    ratios = tuple(float(value) for value in keep_ratios)
    if not ratios:
        raise ValueError("at least one ST Top-K keep ratio is required")
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in ratios):
        raise ValueError("ST Top-K keep ratios must lie inside (0, 1]")
    if len(set(ratios)) != len(ratios):
        raise ValueError("ST Top-K keep ratios must be unique")
    if optimizer_step < 0 or position_in_step < 0:
        raise ValueError("optimizer step and context position cannot be negative")
    return ratios[(optimizer_step + position_in_step) % len(ratios)]


def manifest_group_index_for_step(
    optimizer_step: int,
    *,
    num_manifest_groups: int,
) -> int:
    """Select the next manifest group in a deterministic round-robin cycle.

    The completed optimizer-step count is checkpointed, so this pure mapping
    also makes resume select exactly the group that an uninterrupted run would
    have used.
    """

    if optimizer_step < 0:
        raise ValueError("optimizer_step cannot be negative")
    if num_manifest_groups <= 0:
        raise ValueError("num_manifest_groups must be positive")
    return optimizer_step % num_manifest_groups


def scheduled_l0_coefficient(
    *,
    optimizer_step: int,
    target_lambda: float,
    start_step: int,
    ramp_steps: int,
) -> float:
    """Return a zero-then-linear expected-L0 coefficient."""

    if optimizer_step < 0:
        raise ValueError("optimizer_step cannot be negative")
    if not math.isfinite(target_lambda) or target_lambda < 0:
        raise ValueError("target_lambda must be finite and non-negative")
    if start_step < 0 or ramp_steps < 0:
        raise ValueError("L0 schedule steps cannot be negative")
    if optimizer_step < start_step or target_lambda == 0:
        return 0.0
    if ramp_steps == 0:
        return float(target_lambda)
    progress = min(1.0, (optimizer_step - start_step + 1) / ramp_steps)
    return float(target_lambda * progress)
