#!/usr/bin/env python3
"""Generate compact, deterministic, full-coverage training manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from distance_kv_pattern.core import BlockLayout  # noqa: E402
from distance_kv_pattern.training.data_pipeline import (  # noqa: E402
    ManifestPlanner,
    validate_plans,
    validate_tokenizer_contract,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-name-or-path",
        help="Override config model_name_or_path.",
    )
    parser.add_argument(
        "--value-types",
        nargs="+",
        help="Generate only these value types; defaults to the config.",
    )
    parser.add_argument(
        "--needle-counts",
        nargs="+",
        type=int,
        help="Generate only these needle counts; defaults to the config.",
    )
    parser.add_argument(
        "--template-ids",
        nargs="+",
        help="Generate only these templates; defaults to the config.",
    )
    parser.add_argument(
        "--coverage-rounds",
        nargs="+",
        type=int,
        help=(
            "Explicit coverage rounds to generate, such as 1 2 3; defaults "
            "to range(config train.coverage_rounds)."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="Summary path; defaults to OUTPUT_DIR/generation_summary.json.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main(
    *,
    tokenizer_loader: Callable[..., Any] = AutoTokenizer.from_pretrained,
    contract_validator: Callable[..., dict[str, int]] = validate_tokenizer_contract,
) -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_path = args.model_name_or_path or config["model_name_or_path"]

    tokenizer = tokenizer_loader(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    layout = BlockLayout(**config["layout"])
    contract = contract_validator(tokenizer, layout)
    planner = ManifestPlanner(
        tokenizer,
        layout=layout,
        master_seed=int(config["master_seed"]),
        split="train",
    )

    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    train_config = config["train"]
    value_types = args.value_types or train_config["value_types"]
    needle_counts = args.needle_counts or train_config["needle_counts"]
    template_ids = args.template_ids or train_config["template_ids"]
    coverage_rounds = args.coverage_rounds or list(
        range(int(train_config["coverage_rounds"]))
    )
    if any(coverage_round < 0 for coverage_round in coverage_rounds):
        raise ValueError("coverage rounds must be non-negative")
    selections = {
        "value_types": list(value_types),
        "needle_counts": [int(value) for value in needle_counts],
        "template_ids": list(template_ids),
        "coverage_rounds": [int(value) for value in coverage_rounds],
    }
    targets = [
        args.output_dir
        / "train"
        / value_type
        / f"n{needle_count}"
        / template_id.lower()
        / f"round_{coverage_round:03d}.jsonl"
        for value_type in value_types
        for needle_count in needle_counts
        for template_id in template_ids
        for coverage_round in coverage_rounds
    ]
    if len(set(targets)) != len(targets):
        raise ValueError("generation selectors contain duplicate targets")
    summary_path = args.summary_path or args.output_dir / "generation_summary.json"
    collisions = [
        path
        for target in targets
        for path in (target, target.with_suffix(target.suffix + ".meta.json"))
        if path.exists()
    ]
    if summary_path.exists():
        collisions.append(summary_path)
    if collisions:
        preview = ", ".join(str(path) for path in collisions[:8])
        suffix = " ..." if len(collisions) > 8 else ""
        raise FileExistsError(f"refusing to overwrite existing outputs: {preview}{suffix}")

    for value_type in value_types:
        for needle_count in needle_counts:
            for template_id in template_ids:
                for coverage_round in coverage_rounds:
                    plans = tuple(
                        planner.iter_plans(
                            value_type=value_type,
                            needle_count=int(needle_count),
                            template_id=template_id,
                            coverage_round=coverage_round,
                        )
                    )
                    coverage = validate_plans(
                        plans,
                        layout=layout,
                        value_type=value_type,
                        needle_count=int(needle_count),
                        template_id=template_id,
                        coverage_round=coverage_round,
                    )
                    output_path = (
                        args.output_dir
                        / "train"
                        / value_type
                        / f"n{needle_count}"
                        / template_id.lower()
                        / f"round_{coverage_round:03d}.jsonl"
                    )
                    metadata = write_manifest(
                        output_path,
                        plans,
                        metadata={
                            "split": "train",
                            "value_type": value_type,
                            "needle_count": int(needle_count),
                            "template_id": template_id,
                            "coverage_round": coverage_round,
                            "master_seed": int(config["master_seed"]),
                            "tokenizer_name_or_path": str(tokenizer.name_or_path),
                            "layout": layout.as_dict(),
                            "coverage": coverage,
                            "config_path": str(args.config.resolve()),
                        },
                    )
                    summaries.append(
                        {
                            "path": str(output_path),
                            **coverage,
                            "sha256": metadata["jsonl_sha256"],
                        }
                    )
                    print(
                        f"wrote {output_path}: samples={len(plans)}, "
                        f"slots={coverage['num_slots']}, "
                        f"extra={coverage['extra_slots']}"
                    )

    summary = {
        "schema_version": 2,
        "config": config,
        "selections": selections,
        "tokenizer_contract": contract,
        "layout": layout.as_dict(),
        "num_manifests": len(summaries),
        "num_samples": sum(int(item["num_groups"]) for item in summaries),
        "elapsed_seconds": time.perf_counter() - started,
        "manifests": summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"complete: manifests={summary['num_manifests']}, "
        f"samples={summary['num_samples']}, summary={summary_path}"
    )


if __name__ == "__main__":
    main()
