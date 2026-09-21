"""Reusable training primitives for Q-head distance-pattern learning.

This module intentionally does not own an optimizer step. A formal trainer
must accumulate task gradients over one complete distance-coverage manifest
before applying the optimizer. The helpers here operate on one shared prompt
and its independent query branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Literal, Sequence

import torch

from .hard_concrete import HardConcreteConfig, QHeadHardConcreteGates
from ..retrieval_initialization.retrieval_init import load_retrieval_head_scores
from .runner_types import (
    PromptKVCache,
    QHeadSuffixRunner,
    causal_value_cross_entropy,
)


@dataclass(frozen=True, slots=True)
class QHeadGateInitialization:
    """Retrieval-derived Q-head gate initialization and its audit tensors."""

    gates: QHeadHardConcreteGates
    retrieval_score: torch.Tensor
    coverage_count: torch.Tensor
    initial_keep_probability: torch.Tensor


@dataclass(frozen=True, slots=True)
class BranchwiseBackwardResult:
    """Detached diagnostics from one shared-prompt branchwise backward."""

    branch_losses: torch.Tensor
    branch_weights: torch.Tensor
    weighted_task_loss: float


def gradient_competition_statistics(
    task_gradient: torch.Tensor,
    l0_gradient: torch.Tensor,
) -> dict[str, float | None]:
    """Summarize gate-wise task/L0 competition before gradient clipping.

    A positive gradient closes a gate under gradient descent because the
    optimizer subtracts it from ``log_alpha``.  Gate-wise fractions are more
    informative than a single global L2 ratio: V1 failed even though its L0
    coefficient had been selected from global norms.
    """

    task = torch.as_tensor(task_gradient).detach().float()
    l0 = torch.as_tensor(l0_gradient).detach().float()
    if task.shape != l0.shape or task.numel() == 0:
        raise ValueError("task_gradient and l0_gradient must have the same non-empty shape")
    if not torch.isfinite(task).all() or not torch.isfinite(l0).all():
        raise ValueError("gradients must be finite")

    total = task + l0
    task_norm = task.norm()
    l0_norm = l0.norm()
    norm_ratio = (
        float((l0_norm / task_norm).item())
        if float(task_norm.item()) > 0.0
        else None
    )
    return {
        "l0_to_task_gradient_norm_ratio": norm_ratio,
        "task_gradient_close_fraction": float(task.gt(0).float().mean().item()),
        "task_gradient_zero_fraction": float(task.eq(0).float().mean().item()),
        "l0_dominates_task_fraction": float(
            l0.abs().gt(task.abs()).float().mean().item()
        ),
        "composite_gradient_close_fraction": float(
            total.gt(0).float().mean().item()
        ),
        "l0_flips_task_open_to_close_fraction": float(
            task.lt(0).logical_and(total.gt(0)).float().mean().item()
        ),
    }


def l0_candidate_first_step_statistics(
    task_gradient: torch.Tensor,
    unit_l0_gradient: torch.Tensor,
    log_alpha: torch.Tensor,
    *,
    candidates: Sequence[float],
    learning_rate: float,
    gradient_clip_norm: float = 1.0,
    adam_epsilon: float = 1e-8,
    hard_mask_threshold: float = 0.5,
    config: HardConcreteConfig | None = None,
) -> dict[str, object]:
    """Audit many L0 coefficients from one fixed-parameter task gradient.

    At fixed gates and Hard Concrete noise, the task gradient is independent
    of the scalar L0 coefficient and the analytic L0 gradient is exactly
    linear in that coefficient.  This helper therefore evaluates an entire
    coefficient grid without repeating the expensive language-model pass.

    The simulated update is the exact first AdamW update for a fresh optimizer
    with zero weight decay, including the same global gradient clipping used
    by the formal trainer.  Adam's bias corrections cancel on the first step,
    leaving ``-lr * g / (abs(g) + eps)`` after clipping.
    """

    task = torch.as_tensor(task_gradient).detach().float()
    unit_l0 = torch.as_tensor(unit_l0_gradient).detach().float()
    alpha = torch.as_tensor(log_alpha).detach().float()
    if task.shape != unit_l0.shape or task.shape != alpha.shape or task.numel() == 0:
        raise ValueError(
            "task_gradient, unit_l0_gradient, and log_alpha must share a non-empty shape"
        )
    if task.ndim != 3:
        raise ValueError("audit tensors must have shape [layers, heads, distances]")
    if not torch.isfinite(task).all() or not torch.isfinite(unit_l0).all():
        raise ValueError("audit gradients must be finite")
    if not torch.isfinite(alpha).all():
        raise ValueError("log_alpha must be finite")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    if not math.isfinite(adam_epsilon) or adam_epsilon <= 0:
        raise ValueError("adam_epsilon must be finite and positive")
    if not 0.0 <= hard_mask_threshold <= 1.0:
        raise ValueError("hard_mask_threshold must lie inside [0, 1]")

    values = tuple(float(value) for value in candidates)
    if not values:
        raise ValueError("at least one L0 candidate is required")
    if len(set(values)) != len(values):
        raise ValueError("L0 candidates must be unique")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("L0 candidates must be finite and non-negative")

    hard_concrete = config if config is not None else HardConcreteConfig()
    offset = hard_concrete.beta * math.log(
        -hard_concrete.gamma / hard_concrete.zeta
    )
    probability_before = torch.sigmoid(alpha - offset)
    hard_before = probability_before.ge(hard_mask_threshold)

    records: list[dict[str, object]] = []
    for coefficient in values:
        l0_gradient = unit_l0 * coefficient
        total_gradient = task + l0_gradient
        pre_clip_norm = total_gradient.norm()
        clip_scale = min(
            1.0,
            gradient_clip_norm / (float(pre_clip_norm.item()) + 1e-6),
        )
        clipped_gradient = total_gradient * clip_scale
        update = (
            -learning_rate
            * clipped_gradient
            / (clipped_gradient.abs() + adam_epsilon)
        )
        probability_after = torch.sigmoid(alpha + update - offset)
        hard_after = probability_after.ge(hard_mask_threshold)
        quantiles = torch.quantile(
            probability_after,
            torch.tensor(
                [0.1, 0.5, 0.9],
                dtype=torch.float32,
                device=probability_after.device,
            ),
        )
        records.append(
            {
                "l0_lambda": coefficient,
                "gradient_competition": gradient_competition_statistics(
                    task,
                    l0_gradient,
                ),
                "pre_clip_gradient_norm": float(pre_clip_norm.item()),
                "gradient_clip_scale": clip_scale,
                "simulated_mean_abs_log_alpha_update": float(
                    update.abs().mean().item()
                ),
                "simulated_max_abs_log_alpha_update": float(
                    update.abs().max().item()
                ),
                "simulated_keep_probability_after_update": {
                    "min": float(probability_after.min().item()),
                    "p10": float(quantiles[0].item()),
                    "median": float(quantiles[1].item()),
                    "mean": float(probability_after.mean().item()),
                    "p90": float(quantiles[2].item()),
                    "max": float(probability_after.max().item()),
                },
                "simulated_hard_keep_ratio_after_update": float(
                    hard_after.float().mean().item()
                ),
                "simulated_hard_mask_change_ratio": float(
                    hard_after.ne(hard_before).float().mean().item()
                ),
                "simulated_near_threshold_fraction": float(
                    probability_after.ge(hard_mask_threshold - 0.05)
                    .logical_and(
                        probability_after.le(hard_mask_threshold + 0.05)
                    )
                    .float()
                    .mean()
                    .item()
                ),
            }
        )

    return {
        "learning_rate": float(learning_rate),
        "gradient_clip_norm": float(gradient_clip_norm),
        "adam_epsilon": float(adam_epsilon),
        "hard_mask_threshold": float(hard_mask_threshold),
        "expected_keep_ratio_before_update": float(
            probability_before.mean().item()
        ),
        "hard_keep_ratio_before_update": float(
            hard_before.float().mean().item()
        ),
        "task_gradient_norm": float(task.norm().item()),
        "unit_l0_gradient_norm": float(unit_l0.norm().item()),
        "candidates": records,
    }


def polarization_candidate_continuation_statistics(
    task_gradient: torch.Tensor,
    unit_l0_gradient: torch.Tensor,
    unit_polarization_gradient: torch.Tensor,
    log_alpha: torch.Tensor,
    adam_exp_avg: torch.Tensor,
    adam_exp_avg_sq: torch.Tensor,
    *,
    adam_step: int,
    candidates: Sequence[float],
    l0_lambda: float,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    gradient_clip_norm: float = 1.0,
    adam_epsilon: float = 1e-8,
    config: HardConcreteConfig | None = None,
) -> dict[str, object]:
    """Simulate the next AdamW update for several interior-mass penalties.

    Unlike :func:`l0_candidate_first_step_statistics`, this audit starts from
    an existing AdamW state.  It is therefore suitable for deciding whether a
    new polarization phase can safely continue from a formal checkpoint.
    Weight decay is intentionally unsupported because the gate optimizer uses
    zero weight decay.
    """

    task = torch.as_tensor(task_gradient).detach().float()
    unit_l0 = torch.as_tensor(unit_l0_gradient).detach().float()
    unit_polar = torch.as_tensor(unit_polarization_gradient).detach().float()
    alpha = torch.as_tensor(log_alpha).detach().float()
    exp_avg = torch.as_tensor(adam_exp_avg).detach().float()
    exp_avg_sq = torch.as_tensor(adam_exp_avg_sq).detach().float()
    tensors = (task, unit_l0, unit_polar, alpha, exp_avg, exp_avg_sq)
    if task.ndim != 3 or task.numel() == 0:
        raise ValueError("audit tensors must have a non-empty [layers, heads, distances] shape")
    if any(tensor.shape != task.shape for tensor in tensors[1:]):
        raise ValueError("all polarization audit tensors must have the same shape")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("polarization audit tensors must be finite")
    if isinstance(adam_step, bool) or not isinstance(adam_step, int) or adam_step < 0:
        raise ValueError("adam_step must be a non-negative integer")
    values = tuple(float(value) for value in candidates)
    if not values:
        raise ValueError("at least one polarization candidate is required")
    if len(set(values)) != len(values):
        raise ValueError("polarization candidates must be unique")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("polarization candidates must be finite and non-negative")
    for name, value in (
        ("l0_lambda", l0_lambda),
        ("learning_rate", learning_rate),
        ("gradient_clip_norm", gradient_clip_norm),
        ("adam_epsilon", adam_epsilon),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if learning_rate == 0 or gradient_clip_norm == 0 or adam_epsilon == 0:
        raise ValueError("learning_rate, gradient_clip_norm, and adam_epsilon must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("Adam beta values must lie inside [0, 1)")

    hard_concrete = config if config is not None else HardConcreteConfig()
    nonzero_offset = hard_concrete.beta * math.log(
        -hard_concrete.gamma / hard_concrete.zeta
    )
    one_offset = hard_concrete.beta * math.log(
        (1.0 - hard_concrete.gamma) / (hard_concrete.zeta - 1.0)
    )

    def distribution(current_alpha: torch.Tensor) -> tuple[torch.Tensor, ...]:
        nonzero = torch.sigmoid(current_alpha - nonzero_offset)
        one = torch.sigmoid(current_alpha - one_offset)
        zero = 1.0 - nonzero
        interior = nonzero - one
        endpoint_mask = one.gt(zero)
        return nonzero, zero, interior, one, endpoint_mask

    nonzero_before, zero_before, interior_before, one_before, endpoint_before = (
        distribution(alpha)
    )
    l0_gradient = unit_l0 * float(l0_lambda)
    base_gradient = task + l0_gradient
    next_step = adam_step + 1
    records: list[dict[str, object]] = []
    for coefficient in values:
        polarization_gradient = unit_polar * coefficient
        total_gradient = base_gradient + polarization_gradient
        pre_clip_norm = total_gradient.norm()
        clip_scale = min(
            1.0,
            gradient_clip_norm / (float(pre_clip_norm.item()) + 1e-6),
        )
        clipped_gradient = total_gradient * clip_scale
        next_exp_avg = beta1 * exp_avg + (1.0 - beta1) * clipped_gradient
        next_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * clipped_gradient.square()
        bias_correction1 = 1.0 - beta1**next_step
        bias_correction2 = 1.0 - beta2**next_step
        denominator = next_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
        denominator = denominator + adam_epsilon
        update = -(learning_rate / bias_correction1) * next_exp_avg / denominator
        alpha_after = alpha + update
        nonzero_after, zero_after, interior_after, one_after, endpoint_after = (
            distribution(alpha_after)
        )
        quantiles = torch.quantile(
            nonzero_after,
            torch.tensor(
                [0.1, 0.5, 0.9],
                dtype=torch.float32,
                device=nonzero_after.device,
            ),
        )
        base_norm = base_gradient.norm()
        polarization_norm = polarization_gradient.norm()
        records.append(
            {
                "polarization_lambda": coefficient,
                "polarization_to_task_gradient_norm_ratio": float(
                    polarization_norm / task.norm()
                ) if float(task.norm().item()) > 0 else None,
                "polarization_to_task_plus_l0_gradient_norm_ratio": float(
                    polarization_norm / base_norm
                ) if float(base_norm.item()) > 0 else None,
                "polarization_flips_base_open_to_close_fraction": float(
                    base_gradient.lt(0)
                    .logical_and(total_gradient.gt(0))
                    .float()
                    .mean()
                    .item()
                ),
                "polarization_flips_base_close_to_open_fraction": float(
                    base_gradient.gt(0)
                    .logical_and(total_gradient.lt(0))
                    .float()
                    .mean()
                    .item()
                ),
                "pre_clip_gradient_norm": float(pre_clip_norm.item()),
                "gradient_clip_scale": clip_scale,
                "simulated_mean_abs_log_alpha_update": float(update.abs().mean().item()),
                "simulated_max_abs_log_alpha_update": float(update.abs().max().item()),
                "simulated_keep_probability_after_update": {
                    "min": float(nonzero_after.min().item()),
                    "p10": float(quantiles[0].item()),
                    "median": float(quantiles[1].item()),
                    "mean": float(nonzero_after.mean().item()),
                    "p90": float(quantiles[2].item()),
                    "max": float(nonzero_after.max().item()),
                },
                "simulated_endpoint_mass_after_update": {
                    "zero": float(zero_after.mean().item()),
                    "interior": float(interior_after.mean().item()),
                    "one": float(one_after.mean().item()),
                },
                "simulated_endpoint_keep_ratio_after_update": float(
                    endpoint_after.float().mean().item()
                ),
                "simulated_endpoint_mask_change_ratio": float(
                    endpoint_after.ne(endpoint_before).float().mean().item()
                ),
                "simulated_interior_mass_change": float(
                    interior_after.mean().sub(interior_before.mean()).item()
                ),
            }
        )

    return {
        "learning_rate": float(learning_rate),
        "gradient_clip_norm": float(gradient_clip_norm),
        "adam_epsilon": float(adam_epsilon),
        "adam_betas": [float(beta1), float(beta2)],
        "adam_step_before_update": adam_step,
        "l0_lambda": float(l0_lambda),
        "task_gradient_norm": float(task.norm().item()),
        "l0_gradient_norm": float(l0_gradient.norm().item()),
        "unit_polarization_gradient_norm": float(unit_polar.norm().item()),
        "endpoint_mass_before_update": {
            "zero": float(zero_before.mean().item()),
            "interior": float(interior_before.mean().item()),
            "one": float(one_before.mean().item()),
        },
        "endpoint_keep_ratio_before_update": float(endpoint_before.float().mean().item()),
        "candidates": records,
    }


def retrieval_scores_to_keep_probability(
    retrieval_score: torch.Tensor,
    *,
    keep_floor: float = 0.95,
    keep_gain: float = 0.04,
    mapping: Literal["linear", "logistic"] = "linear",
    target_mean: float = 0.80,
    score_slope: float = 4.0,
    probability_min: float = 1e-3,
    probability_max: float = 1.0 - 1e-3,
) -> torch.Tensor:
    """Map normalized retrieval scores to non-saturated keep probabilities.

    ``linear`` is the historical, backward-compatible mapping.  ``logistic``
    is an audit candidate that centers the score distribution at
    ``target_mean`` and controls contrast with ``score_slope``.  Neither
    mapping is a budget constraint: the resulting probabilities are only an
    initialization and deployment still uses an explicit threshold/quantile.
    """

    score = torch.as_tensor(retrieval_score, dtype=torch.float32)
    if score.ndim != 3:
        raise ValueError("retrieval_score must have shape [layers, q_heads, distances]")
    if not torch.isfinite(score).all():
        raise ValueError("retrieval_score contains NaN or infinity")
    if float(score.min()) < -1e-6 or float(score.max()) > 1.0 + 1e-6:
        raise ValueError("retrieval_score must lie inside [0, 1]")
    if mapping not in ("linear", "logistic"):
        raise ValueError("mapping must be 'linear' or 'logistic'")
    if not 0.0 < keep_floor < 1.0:
        raise ValueError("keep_floor must lie strictly inside (0, 1)")
    if keep_gain < 0.0:
        raise ValueError("keep_gain cannot be negative")
    if keep_floor + keep_gain >= 1.0:
        raise ValueError("keep_floor + keep_gain must remain below 1")
    if not 0.0 < target_mean < 1.0:
        raise ValueError("target_mean must lie strictly inside (0, 1)")
    if score_slope < 0.0 or not torch.isfinite(torch.tensor(score_slope)):
        raise ValueError("score_slope must be finite and non-negative")
    if (
        not 0.0 < probability_min < probability_max < 1.0
        or not torch.isfinite(torch.tensor(probability_min))
        or not torch.isfinite(torch.tensor(probability_max))
    ):
        raise ValueError("probability bounds must satisfy 0 < min < max < 1")
    score = score.clamp(0.0, 1.0)
    if mapping == "linear":
        return keep_floor + keep_gain * score
    centered = score - score.mean()
    logits = torch.logit(torch.tensor(target_mean, dtype=score.dtype))
    probability = torch.sigmoid(logits + score_slope * centered)
    return probability.clamp(probability_min, probability_max)


def initialize_q_head_gates_from_retrieval(
    checkpoint: str | Path,
    *,
    metric: str,
    expected_shape: tuple[int, int, int],
    keep_floor: float = 0.95,
    keep_gain: float = 0.04,
    mapping: Literal["linear", "logistic"] = "linear",
    target_mean: float = 0.80,
    score_slope: float = 4.0,
    probability_min: float = 1e-3,
    probability_max: float = 1.0 - 1e-3,
    config: HardConcreteConfig | None = None,
    device: torch.device | str | None = None,
) -> QHeadGateInitialization:
    """Load complete Q-head scores and initialize one distribution per gate."""

    score, coverage = load_retrieval_head_scores(
        Path(checkpoint),
        metric=metric,
        head_granularity="q_head",
        expected_shape=expected_shape,
    )
    probability = retrieval_scores_to_keep_probability(
        score,
        keep_floor=keep_floor,
        keep_gain=keep_gain,
        mapping=mapping,
        target_mean=target_mean,
        score_slope=score_slope,
        probability_min=probability_min,
        probability_max=probability_max,
    )
    gates = QHeadHardConcreteGates(
        *expected_shape,
        initial_keep_probability=probability,
        config=config,
    )
    if device is not None:
        # Do not pass a model dtype: log_alpha must remain FP32.
        gates.to(device=torch.device(device))
    return QHeadGateInitialization(
        gates=gates,
        retrieval_score=score,
        coverage_count=coverage,
        initial_keep_probability=probability,
    )


def distance_balanced_branch_weights(
    gate_indices: Sequence[int],
    *,
    coverage_count: torch.Tensor,
) -> torch.Tensor:
    """Return exact full-manifest weights, 1 / (D * count[d]), per branch.

    When every branch in the manifest uses these weights, every one of the D
    distances contributes exactly 1/D even if a few distances repeat.
    """

    counts = torch.as_tensor(coverage_count, dtype=torch.long, device="cpu")
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("coverage_count must be a non-empty rank-one tensor")
    if torch.any(counts <= 0):
        raise ValueError("coverage_count must be positive at every distance")
    if not gate_indices:
        raise ValueError("at least one gate index is required")
    indices = torch.tensor(tuple(gate_indices), dtype=torch.long)
    if torch.any(indices < 0) or torch.any(indices >= counts.numel()):
        raise ValueError("gate index lies outside coverage_count")
    return 1.0 / (float(counts.numel()) * counts.index_select(0, indices).float())


def backward_independent_q_head_branches(
    runner: QHeadSuffixRunner,
    prompt_cache: PromptKVCache,
    gates: QHeadHardConcreteGates,
    suffix_input_ids: Sequence[torch.Tensor],
    suffix_labels: Sequence[torch.Tensor],
    *,
    branch_weights: Sequence[float] | torch.Tensor | None = None,
    uniform: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    checkpoint_layers: bool = True,
) -> BranchwiseBackwardResult:
    """Backward independent suffixes one at a time under one sampled pattern.

    Reusing the same uniform noise makes all branches observe exactly the same
    Hard Concrete sample. The gate is reparameterized again for each branch,
    producing a fresh small graph so each large suffix graph can be released
    immediately after its backward call.
    """

    if runner.gate_granularity != "q_head":
        raise ValueError("runner must use Q-head gate semantics")
    if gates.head_granularity != "q_head":
        raise ValueError("gates must use Q-head semantics")
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
    if uniform is not None and generator is not None:
        raise ValueError("provide either uniform or generator, not both")

    if branch_weights is None:
        weights = torch.full(
            (num_branches,),
            1.0 / num_branches,
            dtype=torch.float32,
        )
    else:
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

    if uniform is None:
        uniform = torch.rand(
            gates.shape,
            dtype=torch.float32,
            device=gates.log_alpha.device,
            generator=generator,
        )
    else:
        uniform = torch.as_tensor(
            uniform,
            dtype=torch.float32,
            device=gates.log_alpha.device,
        ).detach()
        if tuple(uniform.shape) != gates.shape:
            raise ValueError(
                f"uniform has shape {tuple(uniform.shape)}, expected {gates.shape}"
            )

    detached_losses: list[torch.Tensor] = []
    weighted_task_loss = 0.0
    for input_ids, labels, weight in zip(
        suffix_input_ids,
        suffix_labels,
        weights,
        strict=True,
    ):
        # Same random sample numerically, independent reparameterization graph.
        learned_gate = gates.sample(uniform=uniform).gate
        output = runner.forward_suffix(
            input_ids,
            prompt_cache=prompt_cache,
            learned_gate=learned_gate,
            checkpoint_layers=checkpoint_layers,
        )
        branch_loss = causal_value_cross_entropy(output.logits, labels)
        (branch_loss * float(weight)).backward()
        detached = branch_loss.detach().float().cpu()
        detached_losses.append(detached)
        weighted_task_loss += float(weight) * float(detached)
        del branch_loss, output, learned_gate

    return BranchwiseBackwardResult(
        branch_losses=torch.stack(detached_losses),
        branch_weights=weights.clone(),
        weighted_task_loss=weighted_task_loss,
    )
