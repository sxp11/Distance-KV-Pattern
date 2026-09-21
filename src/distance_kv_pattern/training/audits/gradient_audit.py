"""Utilities for auditing partial-coverage Q-head gate gradients.

The real 128K audit accumulates task gradients at fixed ``log_alpha`` and
records a few prefix snapshots.  This module contains only deterministic,
device-agnostic bookkeeping and comparison logic so it can be tested without
loading a language model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from ...core.randomness import derive_seed


DEFAULT_CONTEXT_CHECKPOINTS = (8, 16, 32, 64, 128)


@dataclass(frozen=True, slots=True)
class GradientSnapshot:
    """One raw accumulated gradient and its exact data-loss weight."""

    context_count: int
    query_count: int
    covered_distance_count: int
    cumulative_weight: float
    cumulative_weighted_loss: float
    gradient: torch.Tensor

    def __post_init__(self) -> None:
        if self.context_count <= 0:
            raise ValueError("context_count must be positive")
        if self.query_count <= 0:
            raise ValueError("query_count must be positive")
        if self.covered_distance_count <= 0:
            raise ValueError("covered_distance_count must be positive")
        if not math.isfinite(self.cumulative_weight) or self.cumulative_weight <= 0:
            raise ValueError("cumulative_weight must be finite and positive")
        if not math.isfinite(self.cumulative_weighted_loss):
            raise ValueError("cumulative_weighted_loss must be finite")
        if self.gradient.ndim != 3:
            raise ValueError("gradient must have shape [layers, heads, distances]")
        if not torch.isfinite(self.gradient).all():
            raise ValueError("gradient contains NaN or infinity")

    def normalized_gradient(self) -> torch.Tensor:
        """Return the average task gradient represented by this prefix."""

        return self.gradient.float() / self.cumulative_weight

    def metadata(self) -> dict[str, int | float]:
        values = asdict(self)
        values.pop("gradient")
        return values


def q_head_gate_sample_seed(
    master_seed: int,
    *,
    manifest_id: str,
    sample_id: str,
) -> int:
    """Return a stable context-specific seed for Hard Concrete noise."""

    if not manifest_id:
        raise ValueError("manifest_id cannot be empty")
    if not sample_id:
        raise ValueError("sample_id cannot be empty")
    return derive_seed(
        master_seed,
        "q_head_gradient_audit",
        manifest_id,
        sample_id,
    )


def resolve_context_checkpoints(
    requested: Iterable[int],
    *,
    num_contexts: int,
) -> tuple[int, ...]:
    """Validate prefix checkpoints and always include the full manifest."""

    if num_contexts <= 0:
        raise ValueError("num_contexts must be positive")
    checkpoints = {int(value) for value in requested}
    if any(value <= 0 for value in checkpoints):
        raise ValueError("context checkpoints must be positive")
    if any(value > num_contexts for value in checkpoints):
        invalid = sorted(value for value in checkpoints if value > num_contexts)
        raise ValueError(
            f"context checkpoints exceed manifest length {num_contexts}: {invalid}"
        )
    checkpoints.add(num_contexts)
    return tuple(sorted(checkpoints))


def manifest_distance_coverage(
    gate_index_groups: Sequence[Sequence[int]],
    *,
    num_distances: int,
) -> torch.Tensor:
    """Count how often every distance occurs in a compact manifest."""

    if num_distances <= 0:
        raise ValueError("num_distances must be positive")
    counts = torch.zeros(num_distances, dtype=torch.long)
    if not gate_index_groups:
        raise ValueError("manifest contains no context groups")
    for indices in gate_index_groups:
        if not indices:
            raise ValueError("a context contains no gate indices")
        tensor = torch.as_tensor(tuple(indices), dtype=torch.long)
        if torch.any(tensor < 0) or torch.any(tensor >= num_distances):
            raise ValueError("manifest gate index lies outside the distance range")
        counts.scatter_add_(0, tensor, torch.ones_like(tensor))
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        preview = missing[:16]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ValueError(
            f"manifest does not cover all distances; missing {preview}{suffix}"
        )
    return counts


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(F.cosine_similarity(left, right, dim=0).item())


def _top_fraction_sign_agreement(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    fraction: float,
) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top fraction must lie inside (0, 1]")
    candidate = candidate.float().reshape(-1)
    reference = reference.float().reshape(-1)
    count = max(1, math.ceil(reference.numel() * fraction))
    indices = reference.abs().topk(count, sorted=False).indices
    candidate_sign = torch.sign(candidate.index_select(0, indices))
    reference_sign = torch.sign(reference.index_select(0, indices))
    return float(candidate_sign.eq(reference_sign).float().mean().item())


def compare_gradient_snapshot(
    partial: GradientSnapshot,
    full: GradientSnapshot,
    *,
    top_fraction: float = 0.10,
) -> dict[str, object]:
    """Compare one normalized prefix gradient with the complete gradient.

    In addition to the nested prefix-vs-full comparison, the function reports
    prefix-vs-complement cosine.  The latter is a stricter measure because the
    prefix is not contained in the complement.
    """

    if partial.gradient.shape != full.gradient.shape:
        raise ValueError("partial and full gradients have different shapes")
    if partial.context_count > full.context_count:
        raise ValueError("partial snapshot cannot exceed the full snapshot")
    if partial.cumulative_weight > full.cumulative_weight + 1e-8:
        raise ValueError("partial weight cannot exceed full weight")

    partial_gradient = partial.normalized_gradient()
    full_gradient = full.normalized_gradient()
    full_norm = float(full_gradient.norm().item())
    partial_norm = float(partial_gradient.norm().item())
    if full_norm == 0.0:
        relative_l2_error = float("nan")
        norm_ratio = float("nan")
    else:
        relative_l2_error = float(
            (partial_gradient - full_gradient).norm().item() / full_norm
        )
        norm_ratio = partial_norm / full_norm

    result: dict[str, object] = {
        **partial.metadata(),
        "cosine_with_full": _cosine(partial_gradient, full_gradient),
        "relative_l2_error": relative_l2_error,
        "norm_ratio": norm_ratio,
        "top_fraction": top_fraction,
        "top_fraction_sign_agreement": _top_fraction_sign_agreement(
            partial_gradient,
            full_gradient,
            fraction=top_fraction,
        ),
        "per_layer_cosine_with_full": [
            _cosine(partial_gradient[layer], full_gradient[layer])
            for layer in range(full_gradient.shape[0])
        ],
    }

    remaining_weight = full.cumulative_weight - partial.cumulative_weight
    if remaining_weight > 1e-8:
        complement_gradient = (
            full.gradient.float() - partial.gradient.float()
        ) / remaining_weight
        result["cosine_with_complement"] = _cosine(
            partial_gradient,
            complement_gradient,
        )
        result["per_layer_cosine_with_complement"] = [
            _cosine(partial_gradient[layer], complement_gradient[layer])
            for layer in range(full_gradient.shape[0])
        ]
    else:
        result["cosine_with_complement"] = None
        result["per_layer_cosine_with_complement"] = None
    return result


def calibrate_l0_coefficients(
    task_gradient: torch.Tensor,
    l0_gradient: torch.Tensor,
    *,
    target_ratios: Sequence[float] = (0.05, 0.10),
) -> dict[str, float]:
    """Choose coefficients giving target L0/task gradient-norm ratios."""

    if task_gradient.shape != l0_gradient.shape:
        raise ValueError("task and L0 gradients have different shapes")
    if not torch.isfinite(task_gradient).all() or not torch.isfinite(l0_gradient).all():
        raise ValueError("gradient contains NaN or infinity")
    task_norm = float(task_gradient.float().norm().item())
    l0_norm = float(l0_gradient.float().norm().item())
    if task_norm == 0.0:
        raise ValueError("task gradient norm is zero")
    if l0_norm == 0.0:
        raise ValueError("L0 gradient norm is zero")
    coefficients: dict[str, float] = {}
    for ratio in target_ratios:
        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("target ratios must be finite and positive")
        coefficients[f"{ratio:.6g}"] = ratio * task_norm / l0_norm
    return coefficients


def calibrate_gatewise_l0(
    task_gradient: torch.Tensor,
    unit_l0_gradient: torch.Tensor,
    *,
    target_ratios: Sequence[float] = (0.05, 0.10),
    eps: float = 1e-8,
) -> dict[str, object]:
    """Report per-gate L0/task ratios and scalar-L0 calibration quantiles.

    ``unit_l0_gradient`` must be the gradient of ``expected_l0('mean')``
    (that is, the gradient for a coefficient of one).  A single scalar L0
    coefficient cannot satisfy every gate when task gradients are sparse, so
    this report exposes the full distribution instead of silently selecting a
    norm-only coefficient.  ``lambda_for_target`` is the coefficient that
    would make each gate's absolute L0 gradient equal to the requested
    fraction of its task gradient, with ``eps`` suppressing unstable ratios
    for gates whose task gradient is effectively zero.
    """

    if task_gradient.shape != unit_l0_gradient.shape:
        raise ValueError("task and unit L0 gradients have different shapes")
    if task_gradient.ndim != 3:
        raise ValueError("gradients must have shape [layers, heads, distances]")
    if not torch.isfinite(task_gradient).all() or not torch.isfinite(unit_l0_gradient).all():
        raise ValueError("gradients contain NaN or infinity")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    task_abs = task_gradient.float().abs()
    l0_abs = unit_l0_gradient.float().abs()
    flat_task = task_abs.reshape(-1)
    flat_l0 = l0_abs.reshape(-1)
    quantile_points = torch.tensor(
        [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0],
        dtype=torch.float32,
    )

    def _stats(values: torch.Tensor) -> dict[str, float]:
        quantiles = torch.quantile(values, quantile_points)
        return {
            "min": float(quantiles[0].item()),
            "p01": float(quantiles[1].item()),
            "p10": float(quantiles[2].item()),
            "p25": float(quantiles[3].item()),
            "median": float(quantiles[4].item()),
            "p75": float(quantiles[5].item()),
            "p90": float(quantiles[6].item()),
            "p99": float(quantiles[7].item()),
            "max": float(quantiles[8].item()),
            "mean": float(values.mean().item()),
        }

    result: dict[str, object] = {
        "shape": list(task_gradient.shape),
        "eps": eps,
        "task_abs_gradient": _stats(flat_task),
        "unit_l0_abs_gradient": _stats(flat_l0),
        "target_ratios": [float(ratio) for ratio in target_ratios],
        "by_target_ratio": {},
    }
    by_target: dict[str, object] = {}
    for ratio in target_ratios:
        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("target ratios must be finite and positive")
        lambda_for_target = ratio * task_abs / l0_abs.clamp_min(eps)
        gate_ratio_at_median = (
            float(lambda_for_target.median().item()) * l0_abs / task_abs.clamp_min(eps)
        )
        by_target[str(ratio)] = {
            "lambda_for_target_quantiles": _stats(lambda_for_target.reshape(-1)),
            "median_lambda": float(lambda_for_target.median().item()),
            "p10_lambda": float(torch.quantile(lambda_for_target, 0.10).item()),
            "p90_lambda": float(torch.quantile(lambda_for_target, 0.90).item()),
            "fraction_l0_over_task_at_median_lambda": float(
                gate_ratio_at_median.gt(1.0).float().mean().item()
            ),
            "fraction_l0_below_target_at_median_lambda": float(
                gate_ratio_at_median.lt(ratio).float().mean().item()
            ),
        }
    result["by_target_ratio"] = by_target
    return result
