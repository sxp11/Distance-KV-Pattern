#!/usr/bin/env python3
"""Compare two retrieval initialization checkpoints and save a JSON report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-floor", type=float, default=0.95)
    parser.add_argument("--keep-gain", type=float, default=0.04)
    return parser.parse_args()


def pearson(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    x = reference.double().reshape(-1)
    y = candidate.double().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float(torch.dot(x, y).div(denominator).item())


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    array = values.detach().cpu().double().reshape(-1).numpy()
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    boundaries = np.concatenate(
        (
            np.array([0]),
            np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1,
            np.array([array.size]),
        )
    )
    ranks = np.empty(array.size, dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
    return torch.from_numpy(ranks)


def tensor_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("comparison tensors must be finite")
    difference = (candidate.double() - reference.double()).abs().reshape(-1)
    return {
        "pearson": pearson(reference, candidate),
        "mae": float(difference.mean().item()),
        "p99_absolute_error": float(torch.quantile(difference, 0.99).item()),
        "maximum_absolute_error": float(difference.max().item()),
        "reference_mean": float(reference.double().mean().item()),
        "candidate_mean": float(candidate.double().mean().item()),
    }


def top_head_overlap(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    k: int,
) -> dict[str, float]:
    reference_top = reference.topk(k, dim=1).indices
    candidate_top = candidate.topk(k, dim=1).indices
    matches = (
        reference_top.unsqueeze(2).eq(candidate_top.unsqueeze(1)).any(dim=2).sum(dim=1)
    )
    return {
        "mean_fraction": float(matches.float().mean().div(k).item()),
        "exact_set_fraction": float(matches.eq(k).float().mean().item()),
    }


def mean_distance_top_head_overlap(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    k: int,
) -> dict[str, float]:
    return top_head_overlap(
        reference.mean(dim=2, keepdim=True),
        candidate.mean(dim=2, keepdim=True),
        k=k,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("coverage_count", "q_head_scores", "kv_head_scores"):
        if key not in payload:
            raise KeyError(f"{path} lacks {key}")
    return payload


def main() -> None:
    args = parse_args()
    reference = load_checkpoint(args.reference)
    candidate = load_checkpoint(args.candidate)
    reference_coverage = reference["coverage_count"].long()
    candidate_coverage = candidate["coverage_count"].long()
    if reference_coverage.shape != candidate_coverage.shape:
        raise ValueError("coverage shapes differ")

    report: dict[str, Any] = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "coverage": {
            "exact_match": bool(torch.equal(reference_coverage, candidate_coverage)),
            "reference_minimum": int(reference_coverage.min().item()),
            "reference_maximum": int(reference_coverage.max().item()),
            "candidate_minimum": int(candidate_coverage.min().item()),
            "candidate_maximum": int(candidate_coverage.max().item()),
        },
        "scores": {},
    }
    for granularity in ("q_head", "kv_head"):
        score_key = f"{granularity}_scores"
        reference_metrics = reference[score_key]
        candidate_metrics = candidate[score_key]
        if set(reference_metrics) != set(candidate_metrics):
            raise ValueError(f"{granularity} metric names differ")
        metric_reports: dict[str, Any] = {}
        for metric in sorted(reference_metrics):
            ref_tensor = reference_metrics[metric].float()
            candidate_tensor = candidate_metrics[metric].float()
            metric_report = tensor_stats(ref_tensor, candidate_tensor)
            if metric == "retrieval_score_at_k":
                metric_report["spearman"] = pearson(
                    average_ranks(ref_tensor), average_ranks(candidate_tensor)
                )
            metric_reports[metric] = metric_report
        report["scores"][granularity] = metric_reports

    q_reference = reference["q_head_scores"]["retrieval_score_at_k"].float()
    q_candidate = candidate["q_head_scores"]["retrieval_score_at_k"].float()
    report["q_head_top_overlap_by_layer_distance"] = {
        f"top_{k}": top_head_overlap(q_reference, q_candidate, k=k)
        for k in (1, 2, 4, 8)
    }
    report["q_head_top_overlap_after_distance_mean"] = {
        f"top_{k}": mean_distance_top_head_overlap(
            q_reference,
            q_candidate,
            k=k,
        )
        for k in (1, 2, 4, 8)
    }
    reference_keep = args.keep_floor + args.keep_gain * q_reference
    candidate_keep = args.keep_floor + args.keep_gain * q_candidate
    report["candidate_keep_probability_mapping"] = {
        "keep_floor": args.keep_floor,
        "keep_gain": args.keep_gain,
        **tensor_stats(reference_keep, candidate_keep),
    }

    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
