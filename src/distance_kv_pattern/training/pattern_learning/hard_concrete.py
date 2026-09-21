"""Vectorized Hard Concrete gates for static KV-retention patterns.

The trainable object is one FP32 log_alpha tensor. A training forward draws a
relaxed gate tensor; deployment exports a deterministic binary mask from the
analytic non-zero probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


Reduction = Literal["none", "mean", "sum"]
HeadGranularity = Literal["q_head", "kv_head"]


@dataclass(frozen=True, slots=True)
class HardConcreteConfig:
    """Fixed distribution hyperparameters from the standard formulation."""

    beta: float = 2.0 / 3.0
    gamma: float = -0.1
    zeta: float = 1.1
    uniform_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if not self.gamma < 0 < 1 < self.zeta:
            raise ValueError("Hard Concrete requires gamma < 0 < 1 < zeta")
        if not 0 < self.uniform_epsilon < 0.5:
            raise ValueError("uniform_epsilon must lie inside (0, 0.5)")


@dataclass(frozen=True, slots=True)
class HardConcreteSample:
    """One reparameterized sample and its analytic non-zero probabilities."""

    gate: torch.Tensor
    keep_probability: torch.Tensor


@dataclass(frozen=True, slots=True)
class HardConcreteEndpointMass:
    """Analytic probability mass at zero, inside ``(0, 1)``, and at one."""

    zero: torch.Tensor
    interior: torch.Tensor
    one: torch.Tensor


def hard_concrete_log_alpha_from_keep_probability(
    keep_probability: torch.Tensor,
    *,
    beta: float = 2.0 / 3.0,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> torch.Tensor:
    """Invert the analytic P(z > 0) expression and return an FP32 tensor."""

    config = HardConcreteConfig(beta=beta, gamma=gamma, zeta=zeta)
    probability = torch.as_tensor(keep_probability, dtype=torch.float32)
    if torch.any(~torch.isfinite(probability)):
        raise ValueError("keep probabilities must be finite")
    if torch.any((probability <= 0) | (probability >= 1)):
        raise ValueError("keep probabilities must lie strictly inside (0, 1)")
    return torch.logit(probability) + config.beta * math.log(
        -config.gamma / config.zeta
    )


class HardConcreteGates(nn.Module):
    """Learn one Hard Concrete distribution per layer/head/distance gate.

    This class defaults to KV-head semantics for backward compatibility. New
    code should prefer the explicit Q-head or KV-head wrappers below.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_distances: int,
        *,
        initial_keep_probability: float | torch.Tensor = 0.9,
        config: HardConcreteConfig | None = None,
        head_granularity: HeadGranularity = "kv_head",
    ) -> None:
        super().__init__()
        if num_layers <= 0 or num_heads <= 0 or num_distances <= 0:
            raise ValueError("gate dimensions must be positive")
        if head_granularity not in ("q_head", "kv_head"):
            raise ValueError(
                "head_granularity must be either 'q_head' or 'kv_head'"
            )
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_distances = int(num_distances)
        self.head_granularity: HeadGranularity = head_granularity
        self.config = config if config is not None else HardConcreteConfig()

        shape = (self.num_layers, self.num_heads, self.num_distances)
        probability = torch.as_tensor(
            initial_keep_probability,
            dtype=torch.float32,
        )
        try:
            probability = torch.broadcast_to(probability, shape).clone()
        except RuntimeError as exc:
            raise ValueError(
                "initial_keep_probability is not broadcastable to "
                f"{shape}"
            ) from exc
        log_alpha = hard_concrete_log_alpha_from_keep_probability(
            probability,
            beta=self.config.beta,
            gamma=self.config.gamma,
            zeta=self.config.zeta,
        )
        self.log_alpha = nn.Parameter(log_alpha.contiguous())

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.num_layers, self.num_heads, self.num_distances

    @property
    def num_kv_heads(self) -> int:
        """Compatibility accessor for the original KV-head-default class."""

        if self.head_granularity != "kv_head":
            raise AttributeError("Q-head gates do not define num_kv_heads")
        return self.num_heads

    def _checked_log_alpha(self) -> torch.Tensor:
        if self.log_alpha.dtype != torch.float32:
            raise RuntimeError(
                "HardConcreteGates.log_alpha must remain FP32; move the module "
                "with .to(device=...) without passing a model dtype"
            )
        return self.log_alpha

    def keep_probability(self) -> torch.Tensor:
        """Return the analytic probability that each sampled gate is non-zero."""

        log_alpha = self._checked_log_alpha()
        offset = self.config.beta * math.log(
            -self.config.gamma / self.config.zeta
        )
        return torch.sigmoid(log_alpha - offset)

    def expected_l0(self, reduction: Reduction = "mean") -> torch.Tensor:
        """Return the differentiable expected number or ratio of active gates."""

        probability = self.keep_probability()
        if reduction == "none":
            return probability
        if reduction == "mean":
            return probability.mean()
        if reduction == "sum":
            return probability.sum()
        raise ValueError(f"unsupported reduction: {reduction!r}")

    def endpoint_mass(self) -> HardConcreteEndpointMass:
        """Return the three analytic masses of the Hard Concrete mixture.

        ``keep_probability`` is :math:`P(z > 0)`, not :math:`P(z = 1)`.
        Keeping these quantities separate is essential when a relaxed gate is
        eventually exported as a physical binary KV-retention decision.
        """

        log_alpha = self._checked_log_alpha()
        nonzero = self.keep_probability()
        one_offset = self.config.beta * math.log(
            (1.0 - self.config.gamma) / (self.config.zeta - 1.0)
        )
        one = torch.sigmoid(log_alpha - one_offset)
        zero = 1.0 - nonzero
        interior = nonzero - one
        return HardConcreteEndpointMass(
            zero=zero,
            interior=interior,
            one=one,
        )

    def expected_interior_mass(self, reduction: Reduction = "mean") -> torch.Tensor:
        """Return expected probability mass that remains strictly between endpoints."""

        interior = self.endpoint_mass().interior
        if reduction == "none":
            return interior
        if reduction == "mean":
            return interior.mean()
        if reduction == "sum":
            return interior.sum()
        raise ValueError(f"unsupported reduction: {reduction!r}")

    def sample(
        self,
        *,
        generator: torch.Generator | None = None,
        uniform: torch.Tensor | None = None,
    ) -> HardConcreteSample:
        """Draw one vectorized reparameterized sample.

        Supply uniform for exact tests or to share one sampled mask across
        several query branches. Sample outside checkpointed regions so backward
        recomputation never draws a different mask.
        """

        log_alpha = self._checked_log_alpha()
        if uniform is not None and generator is not None:
            raise ValueError("provide either uniform or generator, not both")
        if uniform is None:
            uniform = torch.rand(
                self.shape,
                dtype=torch.float32,
                device=log_alpha.device,
                generator=generator,
            )
        else:
            if uniform.shape != self.shape:
                raise ValueError(
                    f"uniform has shape {tuple(uniform.shape)}, expected {self.shape}"
                )
            uniform = uniform.to(device=log_alpha.device, dtype=torch.float32)

        epsilon = self.config.uniform_epsilon
        uniform = uniform.clamp(epsilon, 1.0 - epsilon)
        concrete = torch.sigmoid(
            (torch.logit(uniform) + log_alpha) / self.config.beta
        )
        stretched = concrete * (
            self.config.zeta - self.config.gamma
        ) + self.config.gamma
        gate = stretched.clamp(0.0, 1.0)
        return HardConcreteSample(
            gate=gate,
            keep_probability=self.keep_probability(),
        )

    def forward(self, uniform: torch.Tensor) -> torch.Tensor:
        """Return one sample through the standard ``nn.Module`` call path.

        DistributedDataParallel installs its reducer from ``forward`` calls.
        This delegates to :meth:`sample`, so distributed and single-GPU
        training use exactly the same Hard Concrete formula.
        """

        return self.sample(uniform=uniform).gate


    def deterministic_relaxed_gate(self) -> torch.Tensor:
        """Return the median-noise relaxed gate, primarily for diagnostics."""

        log_alpha = self._checked_log_alpha()
        concrete = torch.sigmoid(log_alpha / self.config.beta)
        stretched = concrete * (
            self.config.zeta - self.config.gamma
        ) + self.config.gamma
        return stretched.clamp(0.0, 1.0)

    @torch.no_grad()
    def export_hard_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Export a deterministic boolean mask from analytic keep probability."""

        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie inside [0, 1]")
        return self.keep_probability().ge(threshold)

    @torch.no_grad()
    def export_endpoint_mask(self) -> torch.Tensor:
        """Choose the more likely exact endpoint without imposing a keep budget."""

        endpoint = self.endpoint_mass()
        return endpoint.one.gt(endpoint.zero)

    def extra_repr(self) -> str:
        return (
            f"shape={self.shape}, head_granularity={self.head_granularity}, "
            f"beta={self.config.beta}, "
            f"gamma={self.config.gamma}, zeta={self.config.zeta}"
        )


class KVHeadHardConcreteGates(HardConcreteGates):
    """Hard Concrete gates indexed by layer, KV head and distance."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        num_distances: int,
        *,
        initial_keep_probability: float | torch.Tensor = 0.9,
        config: HardConcreteConfig | None = None,
    ) -> None:
        super().__init__(
            num_layers,
            num_kv_heads,
            num_distances,
            initial_keep_probability=initial_keep_probability,
            config=config,
            head_granularity="kv_head",
        )

    @property
    def num_kv_heads(self) -> int:
        return self.num_heads


class QHeadHardConcreteGates(HardConcreteGates):
    """Hard Concrete gates indexed by layer, Q head and distance."""

    def __init__(
        self,
        num_layers: int,
        num_q_heads: int,
        num_distances: int,
        *,
        initial_keep_probability: float | torch.Tensor = 0.9,
        config: HardConcreteConfig | None = None,
    ) -> None:
        super().__init__(
            num_layers,
            num_q_heads,
            num_distances,
            initial_keep_probability=initial_keep_probability,
            config=config,
            head_granularity="q_head",
        )

    @property
    def num_q_heads(self) -> int:
        return self.num_heads
