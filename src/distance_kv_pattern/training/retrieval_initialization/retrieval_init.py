"""Utilities for collecting distance-wise retrieval initialization scores.

The formal initializer is deliberately computed only from suffix query rows.
The 128K document is prefetched with FlashAttention and never requests a full
attention matrix.  For every supervised value token we inspect the preceding
single-token query row, which is exactly the row whose logits predict that
value token under causal language-model teacher forcing.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch

from ..pattern_learning.hard_concrete import (
    hard_concrete_log_alpha_from_keep_probability,
)


RAW_SCHEMA_VERSION = 1
ATTENTION_METRICS = (
    "retrieval_score_at_k",
    "aligned_value_attention",
    "value_span_attention",
    "needle_span_attention",
    "block_attention",
)


def prediction_query_target_pairs(labels: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Map supervised target indices to the causal rows that predict them."""
    target_indices = tuple(index for index, label in enumerate(labels) if label != -100)
    if not target_indices:
        raise ValueError("suffix has no supervised target tokens")
    if target_indices[0] == 0:
        raise ValueError("the first suffix token cannot be supervised")
    if target_indices != tuple(range(target_indices[0], target_indices[-1] + 1)):
        raise ValueError("the formal independent value span must be contiguous")
    return tuple((target_index - 1, target_index) for target_index in target_indices)


def _candidate_token_span(
    tokenizer: Any,
    *,
    text: str,
    char_start: int,
    char_end: int,
    token_ids: Sequence[int],
) -> tuple[int, ...]:
    """Find a short-text token span without relying on global offset mappings."""
    target = text[char_start:char_end]
    approximate_start = len(
        tokenizer.encode(text[:char_start], add_special_tokens=False)
    )
    approximate_end = len(
        tokenizer.encode(text[:char_end], add_special_tokens=False)
    )
    candidates: list[tuple[tuple[int, int, int], tuple[int, ...]]] = []
    for start in range(len(token_ids)):
        for stop in range(start + 1, len(token_ids) + 1):
            decoded = tokenizer.decode(
                token_ids[start:stop],
                skip_special_tokens=False,
            )
            if target not in decoded:
                continue
            score = (
                stop - start,
                len(decoded) - len(target),
                abs(start - approximate_start) + abs(stop - approximate_end),
            )
            candidates.append((score, tuple(range(start, stop))))
    if not candidates:
        raise ValueError(
            f"cannot map character span [{char_start}, {char_end}) in {text!r}"
        )
    return min(candidates, key=lambda item: item[0])[1]


def source_value_token_positions(
    tokenizer: Any,
    *,
    needle_text: str,
    value: str,
    absolute_needle_start: int,
) -> tuple[int, ...]:
    """Return the exact prompt-token positions occupied by a stored value."""
    if needle_text.count(value) != 1:
        raise ValueError("needle text must contain its value exactly once")
    char_start = needle_text.index(value)
    char_end = char_start + len(value)
    needle_ids = tuple(tokenizer.encode(needle_text, add_special_tokens=False))
    local_positions = _candidate_token_span(
        tokenizer,
        text=needle_text,
        char_start=char_start,
        char_end=char_end,
        token_ids=needle_ids,
    )
    return tuple(absolute_needle_start + index for index in local_positions)


def attention_step_metrics(
    attentions: Sequence[torch.Tensor],
    *,
    aligned_source_position: int | Sequence[int],
    value_positions: Sequence[int],
    needle_span: tuple[int, int],
    block_span: tuple[int, int],
    retrieval_k: int,
) -> dict[str, torch.Tensor]:
    """Reduce one incremental attention row to small layer-by-Q-head matrices."""
    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be positive")
    if not attentions:
        raise ValueError("model returned no attention tensors")
    value_index = torch.tensor(value_positions, device=attentions[0].device)
    aligned_positions = (
        (aligned_source_position,)
        if isinstance(aligned_source_position, int)
        else tuple(aligned_source_position)
    )
    if not aligned_positions:
        raise ValueError("aligned source positions must not be empty")
    aligned_index = torch.tensor(
        aligned_positions,
        device=attentions[0].device,
    )
    per_metric: dict[str, list[torch.Tensor]] = {
        metric: [] for metric in ATTENTION_METRICS
    }
    needle_start, needle_stop = needle_span
    block_start, block_stop = block_span
    for layer_attention in attentions:
        if layer_attention.ndim != 4 or layer_attention.shape[0] != 1:
            raise ValueError(
                "expected incremental attention shaped [1, heads, 1, keys]"
            )
        row = layer_attention[0, :, -1, :].float()
        if any(
            not 0 <= source_position < row.shape[-1]
            for source_position in aligned_positions
        ):
            raise ValueError("aligned source position lies outside attention keys")
        if block_stop > row.shape[-1] or needle_stop > row.shape[-1]:
            raise ValueError("source span lies outside attention keys")
        top_indices = torch.topk(
            row,
            k=min(retrieval_k, row.shape[-1]),
            dim=-1,
        ).indices
        per_metric["retrieval_score_at_k"].append(
            top_indices.unsqueeze(-1)
            .eq(aligned_index)
            .any(dim=(-1, -2))
            .float()
        )
        per_metric["aligned_value_attention"].append(
            row.index_select(-1, aligned_index).sum(dim=-1)
        )
        per_metric["value_span_attention"].append(
            row.index_select(-1, value_index).sum(dim=-1)
        )
        per_metric["needle_span_attention"].append(
            row[:, needle_start:needle_stop].sum(dim=-1)
        )
        per_metric["block_attention"].append(
            row[:, block_start:block_stop].sum(dim=-1)
        )
    return {
        metric: torch.stack(layer_values, dim=0).detach().cpu()
        for metric, layer_values in per_metric.items()
    }


def mean_step_metrics(
    steps: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not steps:
        raise ValueError("cannot average zero attention steps")
    return {
        metric: torch.stack([step[metric] for step in steps], dim=0).mean(dim=0)
        for metric in ATTENTION_METRICS
    }


def aggregate_q_heads_to_kv(
    q_head_scores: torch.Tensor,
    *,
    num_kv_heads: int,
    top_q: int = 2,
) -> torch.Tensor:
    """Aggregate stable Q-head scores within each contiguous GQA group.

    ``q_head_scores`` must be ``[layers, q_heads, distances]``.  Aggregation is
    intentionally performed after averaging repeated observations, so a fixed
    KV gate is initialized from Q heads that are consistently strong.
    """
    if q_head_scores.ndim != 3:
        raise ValueError("q_head_scores must have shape [layers, q_heads, distances]")
    num_q_heads = q_head_scores.shape[1]
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("Q-head count must be divisible by KV-head count")
    group_size = num_q_heads // num_kv_heads
    if not 1 <= top_q <= group_size:
        raise ValueError(f"top_q must be in [1, {group_size}]")
    grouped = q_head_scores.reshape(
        q_head_scores.shape[0],
        num_kv_heads,
        group_size,
        q_head_scores.shape[2],
    )
    return grouped.topk(top_q, dim=2).values.mean(dim=2)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


@dataclass(frozen=True, slots=True)
class RetrievalAggregate:
    q_head_scores: dict[str, torch.Tensor]
    kv_head_scores: dict[str, torch.Tensor]
    coverage_count: torch.Tensor
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_distances: int
    top_q: int


def aggregate_raw_records(
    records: Iterable[Mapping[str, Any]],
    *,
    num_distances: int,
    top_q: int = 2,
) -> RetrievalAggregate:
    """Aggregate successful branch records into Q- and KV-head score tensors."""
    successful = [record for record in records if record.get("error") is None]
    if not successful:
        raise ValueError("no successful retrieval records to aggregate")
    first = successful[0]
    num_layers = int(first["num_layers"])
    num_q_heads = int(first["num_q_heads"])
    num_kv_heads = int(first["num_kv_heads"])
    sums = {
        metric: torch.zeros(num_layers, num_q_heads, num_distances, dtype=torch.float64)
        for metric in ATTENTION_METRICS
    }
    coverage = torch.zeros(num_distances, dtype=torch.int64)
    for record in successful:
        dimensions = (
            int(record["num_layers"]),
            int(record["num_q_heads"]),
            int(record["num_kv_heads"]),
        )
        if dimensions != (num_layers, num_q_heads, num_kv_heads):
            raise ValueError("raw records mix incompatible model dimensions")
        gate_index = int(record["gate_index"])
        if not 0 <= gate_index < num_distances:
            raise ValueError(f"invalid gate index in raw record: {gate_index}")
        for metric in ATTENTION_METRICS:
            matrix = torch.tensor(record["q_head_metrics"][metric], dtype=torch.float64)
            if matrix.shape != (num_layers, num_q_heads):
                raise ValueError(
                    f"{metric} has shape {tuple(matrix.shape)}, expected "
                    f"{(num_layers, num_q_heads)}"
                )
            sums[metric][:, :, gate_index] += matrix
        coverage[gate_index] += 1

    denominator = coverage.clamp_min(1).to(torch.float64).view(1, 1, -1)
    q_scores = {metric: (value / denominator).float() for metric, value in sums.items()}
    missing = coverage.eq(0).view(1, 1, -1)
    for value in q_scores.values():
        value.masked_fill_(missing, float("nan"))
    kv_scores = {
        metric: aggregate_q_heads_to_kv(
            value.nan_to_num(float("-inf")),
            num_kv_heads=num_kv_heads,
            top_q=top_q,
        ).masked_fill(coverage.eq(0).view(1, 1, -1), float("nan"))
        for metric, value in q_scores.items()
    }
    return RetrievalAggregate(
        q_head_scores=q_scores,
        kv_head_scores=kv_scores,
        coverage_count=coverage,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_distances=num_distances,
        top_q=top_q,
    )


def aggregate_raw_records_condition_balanced(
    records: Iterable[Mapping[str, Any]],
    *,
    num_distances: int,
    top_q: int = 2,
    condition_fields: Sequence[str] = (
        "value_type",
        "template_id",
        "coverage_round",
    ),
) -> RetrievalAggregate:
    """Average within condition-distance cells, then equally across conditions.

    Full-coverage manifests occasionally assign a second branch to a distance.
    Direct branch averaging would give the condition owning that extra slot more
    weight.  This reducer first averages repeated slots inside each
    ``(condition, distance)`` cell and then averages the available condition
    means with equal weight.
    """
    iterator = (
        record for record in records if record.get("error") is None
    )
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("no successful retrieval records to aggregate") from exc

    num_layers = int(first["num_layers"])
    num_q_heads = int(first["num_q_heads"])
    num_kv_heads = int(first["num_kv_heads"])
    coverage = torch.zeros(num_distances, dtype=torch.int64)
    condition_sums: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
    condition_counts: dict[tuple[Any, ...], torch.Tensor] = {}

    def add(record: Mapping[str, Any]) -> None:
        dimensions = (
            int(record["num_layers"]),
            int(record["num_q_heads"]),
            int(record["num_kv_heads"]),
        )
        if dimensions != (num_layers, num_q_heads, num_kv_heads):
            raise ValueError("raw records mix incompatible model dimensions")
        gate_index = int(record["gate_index"])
        if not 0 <= gate_index < num_distances:
            raise ValueError(f"invalid gate index in raw record: {gate_index}")
        try:
            condition = tuple(record[field] for field in condition_fields)
        except KeyError as exc:
            raise ValueError(
                f"raw record lacks condition field: {exc.args[0]}"
            ) from exc
        if condition not in condition_sums:
            condition_sums[condition] = {
                metric: torch.zeros(
                    num_layers,
                    num_q_heads,
                    num_distances,
                    dtype=torch.float64,
                )
                for metric in ATTENTION_METRICS
            }
            condition_counts[condition] = torch.zeros(
                num_distances, dtype=torch.int64
            )
        for metric in ATTENTION_METRICS:
            matrix = torch.tensor(
                record["q_head_metrics"][metric], dtype=torch.float64
            )
            if matrix.shape != (num_layers, num_q_heads):
                raise ValueError(
                    f"{metric} has shape {tuple(matrix.shape)}, expected "
                    f"{(num_layers, num_q_heads)}"
                )
            condition_sums[condition][metric][:, :, gate_index] += matrix
        condition_counts[condition][gate_index] += 1
        coverage[gate_index] += 1

    add(first)
    for record in iterator:
        add(record)

    equal_sums = {
        metric: torch.zeros(
            num_layers, num_q_heads, num_distances, dtype=torch.float64
        )
        for metric in ATTENTION_METRICS
    }
    contributing_conditions = torch.zeros(num_distances, dtype=torch.int64)
    for condition, sums in condition_sums.items():
        counts = condition_counts[condition]
        present = counts.gt(0)
        contributing_conditions += present
        denominator = counts.clamp_min(1).to(torch.float64).view(1, 1, -1)
        for metric in ATTENTION_METRICS:
            mean = sums[metric] / denominator
            equal_sums[metric][:, :, present] += mean[:, :, present]

    denominator = contributing_conditions.clamp_min(1).to(torch.float64)
    q_scores = {
        metric: (value / denominator.view(1, 1, -1)).float()
        for metric, value in equal_sums.items()
    }
    missing = contributing_conditions.eq(0).view(1, 1, -1)
    for value in q_scores.values():
        value.masked_fill_(missing, float("nan"))
    kv_scores = {
        metric: aggregate_q_heads_to_kv(
            value.nan_to_num(float("-inf")),
            num_kv_heads=num_kv_heads,
            top_q=top_q,
        ).masked_fill(missing, float("nan"))
        for metric, value in q_scores.items()
    }
    return RetrievalAggregate(
        q_head_scores=q_scores,
        kv_head_scores=kv_scores,
        coverage_count=coverage,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_distances=num_distances,
        top_q=top_q,
    )


HeadGranularity = Literal["q_head", "kv_head"]


def load_retrieval_head_scores(
    checkpoint: Path,
    *,
    metric: str,
    head_granularity: HeadGranularity,
    expected_shape: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load complete Q- or KV-head initialization scores from a checkpoint."""

    if head_granularity not in ("q_head", "kv_head"):
        raise ValueError("head_granularity must be 'q_head' or 'kv_head'")
    payload = torch.load(checkpoint, map_location="cpu")
    score_key = f"{head_granularity}_scores"
    try:
        score = payload[score_key][metric].float()
        coverage = payload["coverage_count"].long()
    except KeyError as exc:
        raise KeyError(
            f"{checkpoint} does not contain {head_granularity} metric {metric!r}"
        ) from exc
    if score.ndim != 3:
        raise ValueError("retrieval score must have shape [layers, heads, distances]")
    if expected_shape is not None and tuple(score.shape) != expected_shape:
        raise ValueError(
            f"score shape {tuple(score.shape)} differs from {expected_shape}"
        )
    if coverage.shape != (score.shape[-1],):
        raise ValueError("coverage_count has incompatible shape")
    if torch.any(coverage <= 0):
        raise ValueError("retrieval initialization has uncovered distances")
    if not torch.isfinite(score).all():
        raise ValueError("retrieval score contains NaN or infinity")
    return score, coverage


def save_aggregate(
    aggregate: RetrievalAggregate,
    *,
    output_dir: Path,
    metadata: Mapping[str, Any],
    min_distance: int,
) -> None:
    """Write the tensor checkpoint, inspectable Q/KV CSVs and summary metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RAW_SCHEMA_VERSION,
        "metadata": dict(metadata),
        "top_q": aggregate.top_q,
        "coverage_count": aggregate.coverage_count,
        "q_head_scores": aggregate.q_head_scores,
        "kv_head_scores": aggregate.kv_head_scores,
    }
    torch.save(payload, output_dir / "retrieval_scores.pt")

    q_csv_path = output_dir / "q_head_scores.csv"
    q_fieldnames = [
        "gate_index",
        "relative_distance",
        "layer",
        "q_head",
        "coverage_count",
        *ATTENTION_METRICS,
    ]
    with q_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=q_fieldnames)
        writer.writeheader()
        for gate_index in range(aggregate.num_distances):
            for layer in range(aggregate.num_layers):
                for q_head in range(aggregate.num_q_heads):
                    writer.writerow(
                        {
                            "gate_index": gate_index,
                            "relative_distance": min_distance + gate_index,
                            "layer": layer,
                            "q_head": q_head,
                            "coverage_count": int(
                                aggregate.coverage_count[gate_index].item()
                            ),
                            **{
                                metric: float(
                                    aggregate.q_head_scores[metric][
                                        layer, q_head, gate_index
                                    ].item()
                                )
                                for metric in ATTENTION_METRICS
                            },
                        }
                    )

    csv_path = output_dir / "kv_head_scores.csv"
    fieldnames = [
        "gate_index",
        "relative_distance",
        "layer",
        "kv_head",
        "coverage_count",
        *ATTENTION_METRICS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for gate_index in range(aggregate.num_distances):
            for layer in range(aggregate.num_layers):
                for kv_head in range(aggregate.num_kv_heads):
                    writer.writerow(
                        {
                            "gate_index": gate_index,
                            "relative_distance": min_distance + gate_index,
                            "layer": layer,
                            "kv_head": kv_head,
                            "coverage_count": int(
                                aggregate.coverage_count[gate_index].item()
                            ),
                            **{
                                metric: float(
                                    aggregate.kv_head_scores[metric][
                                        layer, kv_head, gate_index
                                    ].item()
                                )
                                for metric in ATTENTION_METRICS
                            },
                        }
                    )

    coverage = aggregate.coverage_count
    summary = {
        "schema_version": RAW_SCHEMA_VERSION,
        "num_layers": aggregate.num_layers,
        "num_q_heads": aggregate.num_q_heads,
        "num_kv_heads": aggregate.num_kv_heads,
        "num_distances": aggregate.num_distances,
        "top_q": aggregate.top_q,
        "num_successful_branches": int(coverage.sum().item()),
        "num_covered_distances": int(coverage.gt(0).sum().item()),
        "missing_gate_indices": torch.nonzero(
            coverage.eq(0), as_tuple=False
        ).flatten().tolist(),
        "minimum_coverage": int(coverage.min().item()),
        "maximum_coverage": int(coverage.max().item()),
        "metadata": dict(metadata),
        "files": {
            "tensor": "retrieval_scores.pt",
            "q_csv": "q_head_scores.csv",
            "kv_csv": "kv_head_scores.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
