#!/usr/bin/env python3
"""Reaggregate retrieval raw records with equal condition-round weighting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


METHOD_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from distance_kv_pattern import BlockLayout  # noqa: E402
from distance_kv_pattern.training.retrieval_initialization.retrieval_init import (  # noqa: E402
    RAW_SCHEMA_VERSION,
    aggregate_raw_records_condition_balanced,
    save_aggregate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--top-q", type=int, default=2)
    parser.add_argument("--layout-config", type=Path)
    parser.add_argument(
        "--source-label",
        help="Short source name stored in checkpoint metadata.",
    )
    return parser.parse_args()


def layout_from_config(path: Path | None) -> BlockLayout:
    if path is None:
        return BlockLayout()
    config = json.loads(path.read_text(encoding="utf-8"))
    return BlockLayout(**config["layout"])


def iter_records(
    path: Path,
    *,
    stats: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    branch_ids: set[str] = set()
    conditions: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            stats["num_records"] += 1
            branch_id = str(record["branch_id"])
            if branch_id in branch_ids:
                raise ValueError(f"duplicate branch_id: {branch_id}")
            branch_ids.add(branch_id)
            if record.get("error") is None:
                stats["num_successful_records"] += 1
                conditions.add(
                    (
                        str(record["value_type"]),
                        str(record["template_id"]),
                        int(record["coverage_round"]),
                    )
                )
            else:
                stats["num_failed_records"] += 1
            yield record
    stats["conditions"] = [
        {
            "value_type": value_type,
            "template_id": template_id,
            "coverage_round": coverage_round,
        }
        for value_type, template_id, coverage_round in sorted(conditions)
    ]


def main() -> None:
    args = parse_args()
    raw_path = args.raw_jsonl.resolve()
    output_dir = args.output_dir.resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if output_dir == raw_path.parent:
        raise ValueError("output_dir must not overwrite the source aggregation")
    output_files = (
        output_dir / "retrieval_scores.pt",
        output_dir / "q_head_scores.csv",
        output_dir / "kv_head_scores.csv",
        output_dir / "summary.json",
    )
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")

    layout = layout_from_config(args.layout_config)
    stats: dict[str, Any] = {
        "num_records": 0,
        "num_successful_records": 0,
        "num_failed_records": 0,
        "conditions": [],
    }
    aggregate = aggregate_raw_records_condition_balanced(
        iter_records(raw_path, stats=stats),
        num_distances=layout.num_learnable_blocks,
        top_q=args.top_q,
    )
    if stats["num_failed_records"]:
        raise RuntimeError(
            f"raw input contains {stats['num_failed_records']} failed records"
        )
    if int(aggregate.coverage_count.min().item()) <= 0:
        raise RuntimeError("raw input does not cover every learnable distance")

    metadata = {
        "schema_version": RAW_SCHEMA_VERSION,
        "source_label": args.source_label,
        "source_raw_jsonl": str(raw_path),
        **stats,
        "aggregation": (
            "mean repeated slots within "
            "(value_type, template_id, coverage_round, distance), then equal "
            "mean over condition-round cells"
        ),
        "condition_fields": ["value_type", "template_id", "coverage_round"],
    }
    save_aggregate(
        aggregate,
        output_dir=output_dir,
        metadata=metadata,
        min_distance=layout.min_learnable_distance,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_records": stats["num_records"],
                "conditions": stats["conditions"],
                "minimum_coverage": int(aggregate.coverage_count.min().item()),
                "maximum_coverage": int(aggregate.coverage_count.max().item()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
